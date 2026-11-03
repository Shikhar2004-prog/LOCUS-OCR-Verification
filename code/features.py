"""Counterfactual-evidence extraction for CEG (Counterfactual Evidence Graph).

For each (word image x, OCR claim h) we query a *frozen* verifier recognizer V
(TrOCR-base) and produce, segmentation-free:

  per-character evidence (node features of the character graph)
    - token log-prob of h_t under V given x (spread over the chars of its BPE token)
    - token entropy at h_t
    - counterfactual margins: s(h|x) - max s(h'|x) over minimal edits h' of position t
      (confusable substitution, case flip, deletion, V-aligned substitution, insertion)
    - render-normalised evidence: lp_t(h|x) - lp_t(h|render(h))  (cancels the LM prior)
    - latent render distance: 1 - cos(dec_state_t(x), dec_state_t(render(h)))
    - edit-op of h_t w.r.t. V's own greedy reading g (match/sub/ins)
    - character class + relative position
  word-level evidence (global node)
    - s(h|x), s(h|render), s(h|x)-s(g|x), [h==g], length stats, |h|-|g|

Nothing here uses the ground truth. Output: one dict per sample, saved with torch.save.
"""
import json, os, sys, unicodedata
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = os.environ.get("LOCUS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERIFIER = {"FUNSD": "printed", "CORD": "printed", "SROIE": "printed", "IAM": "handwritten",
            "IIIT5K": "str", "OCR10K": "printed"}
FONTS = ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
MAXC = 32  # characters per claim considered for counterfactual edits

# visually confusable groups typical of OCR errors
_GROUPS = ["O0oQDCc", "Il1|!ij", "5Ss$", "8B&3", "2Zz", "6Gb", "9gq", "7T/", "uvnUV", "wW",
           ".,:;'`\"", "-_~=", "(){}[]", "aeo", "hbk", "rnm", "cC(e", "xX%", "yY", "pP", "fft"]
CONF = {}
for g in _GROUPS:
    for c in g:
        CONF.setdefault(c, [])
        CONF[c] += [d for d in g if d != c and d not in CONF[c]]


def char_class(c):
    if c.isdigit():
        return 0
    if c.isalpha():
        return 1 if c.isupper() else 2
    if c.isspace():
        return 3
    return 4


def levenshtein_ops(a, b):
    """Return per-position op of a w.r.t. b: 0 match, 1 substitution, 2 a-only (insertion
    in a), and the aligned character of b (or '' if none)."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    ops, al = [0] * n, [""] * n
    i, j = n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            ops[i - 1] = 0 if a[i - 1] == b[j - 1] else 1
            al[i - 1] = b[j - 1]
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            ops[i - 1] = 2
            i -= 1
        else:
            j -= 1
    while i > 0:
        ops[i - 1] = 2
        i -= 1
    return ops, al, int(D[n, m])


RENDER_SCALE = float(os.environ.get("RENDER_SCALE", "0.75"))  # glyph size / crop height (sensitivity study)


def render(text, h, font_path):
    h = max(32, min(int(h), 96))
    font = ImageFont.truetype(font_path, max(8, int(h * RENDER_SCALE)))
    text = text if text.strip() else " "
    l, t, r, b = font.getbbox(text)
    W = max(8, r - l + h // 2)
    img = Image.new("RGB", (W, h), "white")
    ImageDraw.Draw(img).text((h // 4 - l, (h - (b - t)) // 2 - t), text, fill="black", font=font)
    return img


def variants(h, g, al):
    """Minimal counterfactual edits of h. Returns list of (pos, kind, string).
    kind: 0 confusable-sub, 1 case-flip, 2 deletion, 3 V-aligned sub, 4 insertion-from-g"""
    out = []
    for t, c in enumerate(h[:MAXC]):
        for d in CONF.get(c, [])[:3]:
            out.append((t, 0, h[:t] + d + h[t + 1:]))
        if c.isalpha() and c.swapcase() != c:
            out.append((t, 1, h[:t] + c.swapcase() + h[t + 1:]))
        if len(h) > 1:
            out.append((t, 2, h[:t] + h[t + 1:]))
        if al[t] and al[t] != c:
            out.append((t, 3, h[:t] + al[t] + h[t + 1:]))
    seen, uniq = set(), []
    for v in out:
        if v[2] != h and (v[0], v[2]) not in seen:
            seen.add((v[0], v[2]))
            uniq.append(v)
    return uniq


class Verifier:
    def __init__(self, kind, device="cuda"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        name = f"microsoft/trocr-base-{kind}"
        self.proc = TrOCRProcessor.from_pretrained(name)
        self.tok = self.proc.tokenizer
        self.model = VisionEncoderDecoderModel.from_pretrained(name).to(device).eval().half()
        self.dev = device
        self.start = (self.model.config.decoder_start_token_id
                      or self.model.generation_config.decoder_start_token_id or self.tok.eos_token_id)
        self.eos = self.tok.eos_token_id
        self.pad = self.tok.pad_token_id

    @torch.no_grad()
    def encode(self, imgs, fast=False):
        if fast:  # GPU equivalent of the ViT processor: bilinear 384x384, (x/255-0.5)/0.5
            pv = torch.stack([F.interpolate(
                torch.from_numpy(np.asarray(i.convert("RGB"))).to(self.dev).permute(2, 0, 1)[None].float(),
                size=(384, 384), mode="bilinear", antialias=True, align_corners=False)[0] for i in imgs])
            pv = (pv / 255 - 0.5) / 0.5
        else:
            pv = self.proc(images=[i.convert("RGB") for i in imgs], return_tensors="pt").pixel_values
        return self.model.encoder(pixel_values=pv.to(self.dev).half()).last_hidden_state

    @torch.no_grad()
    def greedy(self, enc):
        from transformers.modeling_outputs import BaseModelOutput
        g = self.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=enc),
                                max_new_tokens=48, num_beams=1)
        return [t.strip() for t in self.proc.batch_decode(g, skip_special_tokens=True)]

    def tokenize(self, texts):
        ids = [self.tok(t, add_special_tokens=False)["input_ids"][:60] for t in texts]
        return ids

    @torch.no_grad()
    def score(self, enc, idx, texts, want_states=False, bs=384):
        """Teacher-forced scoring. enc: (N, S, D) encoder states; idx[i] picks enc row for
        texts[i]. Returns per-token logprobs, entropies, (optional) last hidden states."""
        ids = self.tokenize(texts)
        out_lp, out_ent, out_h = [None] * len(texts), [None] * len(texts), [None] * len(texts)
        order = np.argsort([len(i) for i in ids])
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            L = max(len(ids[i]) for i in sel) + 1
            inp = torch.full((len(sel), L), self.pad, dtype=torch.long)
            tgt = torch.full((len(sel), L), -1, dtype=torch.long)
            for r, i in enumerate(sel):
                seq = ids[i]
                inp[r, 0] = self.start
                inp[r, 1:len(seq) + 1] = torch.tensor(seq, dtype=torch.long)
                tgt[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                tgt[r, len(seq)] = self.eos
            inp, tgt = inp.to(self.dev), tgt.to(self.dev)
            att = (tgt != -1).long()
            e = enc[torch.as_tensor([idx[i] for i in sel], device=self.dev)]
            o = self.model.decoder(input_ids=inp, attention_mask=att, encoder_hidden_states=e,
                                   output_hidden_states=want_states)
            logp = torch.log_softmax(o.logits.float(), -1)
            tlp = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            ent = -(logp.exp() * logp).sum(-1)
            hs = o.hidden_states[-1].float() if want_states else None
            for r, i in enumerate(sel):
                n = len(ids[i]) + 1
                out_lp[i] = tlp[r, :n].cpu()
                out_ent[i] = ent[r, :n].cpu()
                if want_states:
                    out_h[i] = hs[r, :n].cpu()
        return ids, out_lp, out_ent, out_h


def char_token_map(tok, text, ids):
    """Map each char of text to the index of the BPE token that covers it."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"][:len(ids)]
    m = np.full(len(text), -1, dtype=np.int64)
    for k, (a, b) in enumerate(offs):
        m[a:b] = k
    # chars not covered (e.g. leading spaces folded into next token) -> nearest token
    last = 0
    for c in range(len(text)):
        if m[c] < 0:
            m[c] = last
        last = m[c]
    m = np.clip(m, 0, max(0, len(ids) - 1))
    return m


@torch.no_grad()
def extract(V, imgs, hyps, B=48):
    """imgs: list of PIL, hyps: list of claimed strings. Returns list of feature dicts."""
    results = []
    for s in range(0, len(imgs), B):
        bi, bh = imgs[s:s + B], [h if h.strip() else "?" for h in hyps[s:s + B]]
        n = len(bi)
        enc_x = V.encode(bi)
        g = V.greedy(enc_x)
        rend = [[render(h, im.size[1], f) for f in FONTS] for h, im in zip(bh, bi)]
        enc_r = V.encode([r for rr in rend for r in rr])  # n*len(FONTS)
        nf = len(FONTS)
        # claim scored on real image, with states; and on each rendering
        _, lp_x, ent_x, hs_x = V.score(enc_x, list(range(n)), bh, want_states=True)
        _, lp_r, _, hs_r = V.score(enc_r, [i * nf + f for i in range(n) for f in range(nf)],
                                   [h for h in bh for _ in range(nf)], want_states=True)
        _, lp_g, _, _ = V.score(enc_x, list(range(n)), [x if x else "?" for x in g])
        # counterfactual variants
        var_idx, var_txt, var_meta = [], [], []
        aligns = []
        for i, (h, gg) in enumerate(zip(bh, g)):
            ops, al, dist = levenshtein_ops(h, gg)
            aligns.append((ops, al, dist))
            for (t, kind, v) in variants(h, gg, al):
                var_idx.append(i)
                var_txt.append(v)
                var_meta.append((i, t, kind))
        _, lp_v, _, _ = V.score(enc_x, var_idx, var_txt) if var_txt else (None, [], None, None)
        s_x = [float(l.sum()) for l in lp_x]
        for i in range(n):
            h, ids_h = bh[i], V.tokenize([bh[i]])[0]
            L = len(h)
            cmap = char_token_map(V.tok, h, ids_h)
            ntok_chars = np.bincount(cmap, minlength=max(1, len(ids_h)))
            lpx = lp_x[i].numpy()
            lpr = np.mean([lp_r[i * nf + f].numpy() for f in range(nf)], 0)
            ent = ent_x[i].numpy()
            cos = np.mean([F.cosine_similarity(hs_x[i], hs_r[i * nf + f], dim=-1).numpy()
                           for f in range(nf)], 0)
            ops, al, dist = aligns[i]
            # margins per char & kind
            marg = np.full((L, 5), 10.0, dtype=np.float32)  # 10 = no competitor found
            for (vi, t, kind), l in zip(var_meta, lp_v):
                if vi == i:
                    m = s_x[i] - float(l.sum())
                    marg[t, kind] = min(marg[t, kind], m)
            node = np.zeros((L, 18), dtype=np.float32)
            for t in range(L):
                k = cmap[t] if len(ids_h) else 0
                share = 1.0 / max(1, ntok_chars[k]) if len(ids_h) else 1.0
                node[t, 0] = lpx[k] * share
                node[t, 1] = ent[k]
                node[t, 2] = (lpx[k] - lpr[k]) * share
                node[t, 3] = 1 - cos[k]
                node[t, 4:9] = np.clip(marg[t], -20, 10)
                node[t, 9] = np.clip(marg[t].min(), -20, 10)
                node[t, 10 + ops[t]] = 1  # 10..12 edit op vs V reading
                node[t, 13 + char_class(h[t])] = 1  # 13..17 char class
            sr = float(lpr[:len(lpr)].sum())
            glob = np.array([s_x[i], sr, s_x[i] - sr, s_x[i] - float(lp_g[i].sum()),
                             float(h == g[i]), dist, dist / max(1, L), L, len(g[i]),
                             float(lpx.min()), float(lpx.mean()), float((lpx - lpr).min()),
                             float(np.clip(marg.min(), -20, 10)), float(ent.max()),
                             float(1 - cos.min())], dtype=np.float32)
            results.append({"node": node, "glob": glob, "g": g[i], "h": h})
    return results


def main(ds, engine):
    """engine in {tess, trocr} -> verify real hypotheses; 'pretrain' -> label-free
    agreement-bootstrapped corrupted pairs built from unlabeled train images."""
    man = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/manifest.jsonl")]
    V = Verifier(VERIFIER[ds])
    if engine in ("tess", "trocr", "given"):
        hyp = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/hyp_{engine}.jsonl"))}
        rows = [r for r in man if r["id"] in hyp]
        claims = [hyp[r["id"]]["hyp"] for r in rows]
    else:
        rows, claims = pretrain_pairs(ds, man)
    imgs = [Image.open(r["path"]).convert("RGB") for r in rows]
    feats = extract(V, imgs, claims)
    for r, f, c in zip(rows, feats, claims):
        f["id"], f["split"], f["claim"] = r["id"], r["split"], c
        if engine != "pretrain":
            f["gt"] = r["gt"]
            if "label" in r:  # OCR10K: label given by the dataset, no gt string
                f["label"] = r["label"]
        else:
            f["pseudo_label"] = r["pseudo_label"]
    torch.save(feats, f"{ROOT}/data/{ds}/feat_{engine}.pt")
    print(ds, engine, len(feats))


def corrupt(s, rng):
    """OCR-like corruption: confusable sub, case flip, deletion, insertion, merge/split."""
    s = list(s)
    for _ in range(rng.choice([1, 1, 1, 2])):
        if not s:
            break
        t = rng.randrange(len(s))
        op = rng.random()
        if op < 0.45 and CONF.get(s[t]):
            s[t] = rng.choice(CONF[s[t]])
        elif op < 0.6 and s[t].isalpha():
            s[t] = s[t].swapcase()
        elif op < 0.8 and len(s) > 1:
            del s[t]
        else:
            s.insert(t, rng.choice(CONF.get(s[t], [s[t]]) or [s[t]]))
    return "".join(s)


def pretrain_pairs(ds, man):
    """Label-free: keep TRAIN images where two independent engines agree (pseudo-correct,
    no human label used), then pair each with its agreed string (pos) or a corruption (neg)."""
    import random
    rng = random.Random(0)
    e1, e2 = ("trocr", "trocrB") if ds == "IAM" else ("tess", "trocr")  # Tesseract fails on handwriting
    T = {r["id"]: r["hyp"] for r in map(json.loads, open(f"{ROOT}/data/{ds}/hyp_{e1}.jsonl"))}
    S = {r["id"]: r["hyp"] for r in map(json.loads, open(f"{ROOT}/data/{ds}/hyp_{e2}.jsonl"))}
    agree = [r for r in man if r["split"] == "train" and T.get(r["id"]) and T[r["id"]] == S.get(r["id"])]
    rng.shuffle(agree)
    rows, claims = [], []
    for r in agree[:3000]:
        a = T[r["id"]]
        rows.append(dict(r, pseudo_label=1)); claims.append(a)
        c = corrupt(a, rng)
        if c != a:
            rows.append(dict(r, pseudo_label=0)); claims.append(c)
    return rows, claims


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

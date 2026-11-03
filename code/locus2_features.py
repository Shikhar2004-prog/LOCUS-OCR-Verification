"""LOCUS v2 evidence: dual-verifier counterfactual edit graphs.

Two frozen, complementary recognisers score every hypothesis v of the edit graph:
  V1 = TrOCR-base (BPE, transformer encoder-decoder; AAAI'23)
  V2 = PARSeq     (character-level permuted AR; ECCV'22), canonical left-to-right scoring
For each verifier: s(v|x), s(v|R(v)), rho(v) = s(v|x) - s(v|R(v)), token minima, latent
decoder distance between x and R(v)  -> 11 features each, 22 per node.
The neighbourhood N(h) contains minimal edits of h plus BOTH verifiers' readings and the
single edits aligned toward each reading. No ground truth is used.
Output: data/<DS>/locus2_<engine>.pt
"""
import json, sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import CONF, FONTS, MAXC, ROOT, VERIFIER, Verifier, levenshtein_ops, render
from locus_features import score_pairs

MAXN = 28
# rendering sensitivity study: font of R(v) (default Liberation Sans), optional subset and output tag
import os
RENDER_FONT = os.environ.get("RENDER_FONT", FONTS[0])
SUBSET = int(os.environ.get("SUBSET", "0"))      # >0: first N train and first N test crops only
TAG = os.environ.get("TAG", "")
KIND = {"claim": 0, "conf": 1, "case": 2, "del": 3, "vsub": 4, "ins": 5, "greedy": 6, "pgreedy": 7}


class ParseqVerifier:
    def __init__(self, dev="cuda"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pydeps"))
        self.m = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).eval().to(dev)
        self.tok = self.m.tokenizer
        self.dev = dev
        self.tf = T.Compose([T.Resize((32, 128), T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])
        self.charset = set(self.tok._itos)
        self.maxlen = self.m.model.max_label_length

    def clean(self, s):
        s = "".join(c for c in s if c in self.charset)[: self.maxlen]
        return s if s else "?"

    @torch.no_grad()
    def encode(self, imgs):
        x = torch.stack([self.tf(i.convert("RGB")) for i in imgs]).to(self.dev)
        return torch.cat([self.m.model.encode(x[j:j + 512]) for j in range(0, len(x), 512)])

    @torch.no_grad()
    def greedy(self, mem):
        # decode from precomputed memory (same AR loop as PARSeq.forward, refine_iters as configured)
        out = []
        for j in range(0, mem.shape[0], 512):
            out += self._decode_mem(mem[j:j + 512])
        return out

    def _decode_mem(self, memory):
        mdl = self.m.model
        bs = memory.shape[0]
        num_steps = self.maxlen + 1
        pos_queries = mdl.pos_queries[:, :num_steps].expand(bs, -1, -1)
        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self.dev), 1)
        tgt_in = torch.full((bs, num_steps), self.tok.pad_id, dtype=torch.long, device=self.dev)
        tgt_in[:, 0] = self.tok.bos_id
        logits = []
        for i in range(num_steps):
            j = i + 1
            o = mdl.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                           tgt_query_mask=query_mask[i:j, :j])
            p = mdl.head(o)
            logits.append(p)
            if j < num_steps:
                tgt_in[:, j] = p.squeeze(1).argmax(-1)
                if (tgt_in == self.tok.eos_id).any(-1).all():
                    break
        logits = torch.cat(logits, 1)
        if mdl.refine_iters:
            qm = query_mask.clone()
            qm[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self.dev), 2)] = 0
            bos = torch.full((bs, 1), self.tok.bos_id, dtype=torch.long, device=self.dev)
            for _ in range(mdl.refine_iters):
                ti = torch.cat([bos, logits[:, :-1].argmax(-1)], 1)
                pad = (ti == self.tok.eos_id).int().cumsum(-1) > 0
                o = mdl.decode(ti, memory, tgt_mask, pad, pos_queries, qm[:, : ti.shape[1]])
                logits = mdl.head(o)
        preds, _ = self.tok.decode(logits.softmax(-1))
        return preds

    @torch.no_grad()
    def score_pairs(self, mem_x, mem_r, idx_x, texts, bs=512):
        """Canonical L->R teacher-forced scoring on the real crop and on the rendering."""
        texts = [self.clean(t) for t in texts]
        out = np.zeros((len(texts), 11), dtype=np.float32)
        mdl = self.m.model
        for s in range(0, len(texts), bs):
            tt = texts[s:s + bs]
            tgt = self.tok.encode(tt, self.dev)
            tin, tout = tgt[:, :-1], tgt[:, 1:]
            L = tin.shape[1]
            causal = torch.triu(torch.ones((L, L), dtype=torch.bool, device=self.dev), 1)
            pad = (tin == self.tok.pad_id) | (tin == self.tok.eos_id)
            res = []
            for mem in (mem_x[torch.as_tensor(idx_x[s:s + bs], device=self.dev)], mem_r[s:s + bs]):
                o = mdl.decode(tin, mem, causal, pad, tgt_query_mask=causal)
                safe = torch.where(tout == self.tok.pad_id, torch.zeros_like(tout), tout)  # head excludes bos/pad
                lp = torch.log_softmax(mdl.head(o).float(), -1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
                res.append((lp, o.float()))
            (a, hx), (b, hr) = res
            msk = tout != self.tok.pad_id
            m = msk.float()
            n = m.sum(1)
            big = torch.where(msk, torch.zeros_like(a), torch.full_like(a, 1e4))
            d = a - b
            cos = F.cosine_similarity(hx, hr, dim=-1)
            f = torch.stack([(a * m).sum(1), (b * m).sum(1), (d * m).sum(1), (a * m).sum(1) / n,
                             (a + big).min(1).values, (d + big).min(1).values, (d * m).sum(1) / n,
                             1 - (cos * m).sum(1) / n, 1 - (cos + big).min(1).values,
                             torch.tensor([len(t) for t in tt], device=self.dev, dtype=torch.float), n], 1)
            out[s:s + len(tt)] = f.cpu().numpy()
        return out


def aligned_edits(h, g, kind_sub="vsub"):
    """Single edits of h that move it one step toward reading g (substitutions/insertions/deletions)."""
    out = []
    if not g or g == h:
        return out
    ops, al, _ = levenshtein_ops(h, g)
    for t, c in enumerate(h[:MAXC]):
        if al[t] and al[t] != c:
            out.append((h[:t] + al[t] + h[t + 1:], kind_sub, t))
        if ops[t] == 2 and len(h) > 1:
            out.append((h[:t] + h[t + 1:], "del", t))
    gops, _, _ = levenshtein_ops(g, h)
    p = 0
    for j, c in enumerate(g):
        if gops[j] == 2:
            out.append((h[:p] + c + h[p:], "ins", p))
        else:
            p += 1
    return out


def neighbours(h, g1, g2):
    pri = []
    if g2 and g2 != h:
        pri.append((g2, "pgreedy", -1))
    if g1 and g1 != h:
        pri.append((g1, "greedy", -1))
    pri += aligned_edits(h, g2) + aligned_edits(h, g1)
    rest = []
    for t, c in enumerate(h[:MAXC]):
        if c.isalpha() and c.swapcase() != c:
            rest.append((h[:t] + c.swapcase() + h[t + 1:], "case", t))
        for d in CONF.get(c, [])[:2]:
            rest.append((h[:t] + d + h[t + 1:], "conf", t))
        if len(h) > 1:
            rest.append((h[:t] + h[t + 1:], "del", t))
    seen, out = {h}, []
    for v in pri + rest:
        if v[0] and v[0] not in seen:
            seen.add(v[0]); out.append(v)
        if len(out) >= MAXN:
            break
    return out


@torch.no_grad()
def extract(V1, V2, imgs, claims, B=32):
    out = []
    for s in range(0, len(imgs), B):
        bi = imgs[s:s + B]
        bh = [h if h.strip() else "?" for h in claims[s:s + B]]
        e1 = V1.encode(bi, fast=True)
        g1 = V1.greedy(e1)
        e2 = V2.encode(bi)
        g2 = V2.greedy(e2)
        hyps, owner, meta = [], [], []
        for i, h in enumerate(bh):
            for v, kind, pos in [(h, "claim", -1)] + neighbours(h, g1[i], g2[i]):
                hyps.append(v); owner.append(i); meta.append((kind, pos))
        rend = [render(v, bi[o].size[1], RENDER_FONT) for v, o in zip(hyps, owner)]
        r1 = torch.cat([V1.encode(rend[j:j + 192], fast=True) for j in range(0, len(rend), 192)])
        f1 = score_pairs(V1, e1, r1, owner, hyps)
        del r1
        r2 = V2.encode(rend)
        f2 = V2.score_pairs(e2, r2, owner, hyps)
        Fx = np.concatenate([f1, f2], 1)
        for i in range(len(bi)):
            sel = [j for j, o in enumerate(owner) if o == i]
            Lh = max(1, len(bh[i]))
            kinds = np.zeros((len(sel), len(KIND)), dtype=np.float32)
            pos = np.zeros((len(sel), 1), dtype=np.float32)
            for r, j in enumerate(sel):
                kinds[r, KIND[meta[j][0]]] = 1
                pos[r, 0] = (meta[j][1] + 1) / Lh if meta[j][1] >= 0 else -1
            out.append({"hyps": [hyps[j] for j in sel], "x": Fx[sel], "kind": kinds, "pos": pos,
                        "edit_pos": [meta[j][1] for j in sel], "g": g1[i], "g2": g2[i], "h": bh[i]})
        if s % (B * 25) == 0:
            print(f"  {s}/{len(imgs)}", flush=True)
    return out


def main(ds, engine):
    man = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/manifest.jsonl")]
    V1 = Verifier(VERIFIER[ds])
    V2 = ParseqVerifier()
    if engine == "pretrain":
        from features import pretrain_pairs
        rows, claims = pretrain_pairs(ds, man)
    else:
        hyp = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/hyp_{engine}.jsonl"))}
        rows = [r for r in man if r["id"] in hyp]
        if SUBSET:
            rows = [r for r in rows if r["split"] == "train"][:SUBSET] + [r for r in rows if r["split"] == "test"][:SUBSET]
        claims = [hyp[r["id"]]["hyp"] for r in rows]
    imgs = [Image.open(r["path"]).convert("RGB") for r in rows]
    feats = extract(V1, V2, imgs, claims)
    for r, f, c in zip(rows, feats, claims):
        f["id"], f["split"], f["claim"] = r["id"], r["split"], c
        if engine == "pretrain":
            f["pseudo_label"] = r["pseudo_label"]
        else:
            f["gt"] = r["gt"]
    torch.save(feats, f"{ROOT}/data/{ds}/locus2_{engine}{TAG}.pt")
    print(ds, engine, len(feats))


if __name__ == "__main__":
    torch.set_num_threads(8)
    main(sys.argv[1], sys.argv[2])

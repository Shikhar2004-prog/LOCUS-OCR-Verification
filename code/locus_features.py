"""LOCUS evidence: a counterfactual edit graph around each OCR claim.

Nodes  = the claim h plus its counterfactual neighbours N(h): minimal OCR-style edits
         (confusable substitution, case flip, deletion, verifier-aligned substitution and
         insertion) and the frozen verifier's own reading g.
Node evidence (per hypothesis v, scored by a frozen recognizer V, no ground truth used):
         s(v|x)               log-likelihood of v given the real crop x
         s(v|R(v))            log-likelihood of v given a clean rendering of v
         rho(v) = s(v|x) - s(v|R(v))   render-normalised likelihood ratio (prior cancels)
         token-level minima, decoder-state distance between x and R(v), lengths
Edges  = claim-neighbour (typed by edit kind) and neighbour-neighbour when the two
         hypotheses edit the same position (competing explanations of one glyph).
The ground truth is used only to record which node (if any) equals it (repair target).
"""
import os
import json, sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import CONF, FONTS, MAXC, ROOT, VERIFIER, Verifier, levenshtein_ops, render

MAXN = 24  # neighbours per claim
KIND = {"claim": 0, "conf": 1, "case": 2, "del": 3, "vsub": 4, "ins": 5, "greedy": 6}


def neighbours(h, g):
    """Return list of (string, kind, position) in priority order (claim excluded)."""
    ops, al, _ = levenshtein_ops(h, g)
    pri = []
    # verifier-suggested edits first (most informative competitors)
    if g and g != h:
        pri.append((g, "greedy", -1))
    for t, c in enumerate(h[:MAXC]):
        if al[t] and al[t] != c:
            pri.append((h[:t] + al[t] + h[t + 1:], "vsub", t))
    gops, _, _ = levenshtein_ops(g, h)
    p = 0
    for j, c in enumerate(g):
        if gops[j] == 2:
            pri.append((h[:p] + c + h[p:], "ins", p))
        else:
            p += 1
    for t in range(min(len(h), MAXC)):
        if ops[t] == 2 and len(h) > 1:
            pri.append((h[:t] + h[t + 1:], "del", t))
    rest = []
    for t, c in enumerate(h[:MAXC]):
        if c.isalpha() and c.swapcase() != c:
            rest.append((h[:t] + c.swapcase() + h[t + 1:], "case", t))
        for d in CONF.get(c, [])[:2]:
            rest.append((h[:t] + d + h[t + 1:], "conf", t))
        if len(h) > 1 and ops[t] != 2:
            rest.append((h[:t] + h[t + 1:], "del", t))
    seen, out = {h}, []
    for v in pri + rest:
        if v[0] and v[0] not in seen:
            seen.add(v[0])
            out.append(v)
        if len(out) >= MAXN:
            break
    return out


@torch.no_grad()
def _tf(V, enc, inp, att):
    o = V.model.decoder(input_ids=inp, attention_mask=att, encoder_hidden_states=enc,
                        output_hidden_states=True)
    return torch.log_softmax(o.logits.float(), -1), o.hidden_states[-1].float()


@torch.no_grad()
def score_pairs(V, enc_x, enc_r, idx_x, texts, bs=256):
    """For hypothesis i: teacher-forced scoring on real crop enc_x[idx_x[i]] and on its
    own rendering enc_r[i]; all reductions on GPU. Returns (N, 11) features."""
    ids = V.tokenize(texts)
    out = np.zeros((len(texts), 11), dtype=np.float32)
    order = np.argsort([len(i) for i in ids])
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        Lm = max(len(ids[i]) for i in sel) + 1
        inp = torch.full((len(sel), Lm), V.pad, dtype=torch.long)
        tgt = torch.zeros((len(sel), Lm), dtype=torch.long)
        msk = torch.zeros((len(sel), Lm), dtype=torch.bool)
        for r, i in enumerate(sel):
            q = ids[i]
            inp[r, 0] = V.start
            inp[r, 1:len(q) + 1] = torch.tensor(q, dtype=torch.long)
            tgt[r, :len(q)] = torch.tensor(q, dtype=torch.long)
            tgt[r, len(q)] = V.eos
            msk[r, :len(q) + 1] = True
        inp, tgt, msk = inp.to(V.dev), tgt.to(V.dev), msk.to(V.dev)
        att = msk.long()
        sel_t = torch.as_tensor(sel, device=V.dev)
        ex = enc_x[torch.as_tensor([idx_x[i] for i in sel], device=V.dev)]
        lpx, hx = _tf(V, ex, inp, att)
        lpr, hr = _tf(V, enc_r[sel_t], inp, att)
        a = lpx.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        b = lpr.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        cos = F.cosine_similarity(hx, hr, dim=-1)
        m = msk.float()
        n = m.sum(1)
        big = torch.where(msk, torch.zeros_like(a), torch.full_like(a, 1e4))
        d = a - b
        feats = torch.stack([(a * m).sum(1), (b * m).sum(1), (d * m).sum(1), (a * m).sum(1) / n,
                             (a + big).min(1).values, (d + big).min(1).values, (d * m).sum(1) / n,
                             1 - (cos * m).sum(1) / n, 1 - (cos + big).min(1).values,
                             torch.tensor([len(texts[i]) for i in sel], device=V.dev, dtype=torch.float), n], 1)
        out[sel] = feats.cpu().numpy()
    return out


@torch.no_grad()
def extract(V, imgs, claims, B=32):
    out = []
    for s in range(0, len(imgs), B):
        bi = imgs[s:s + B]
        bh = [h if h.strip() else "?" for h in claims[s:s + B]]
        enc_x = V.encode(bi, fast=True)
        g = V.greedy(enc_x)
        hyps, owner, meta = [], [], []
        for i, (h, gg) in enumerate(zip(bh, g)):
            nb = [(h, "claim", -1)] + neighbours(h, gg)
            for v, kind, pos in nb:
                hyps.append(v); owner.append(i); meta.append((kind, pos))
        rend = [render(v, bi[o].size[1], FONTS[0]) for v, o in zip(hyps, owner)]
        enc_r = torch.cat([V.encode(rend[j:j + 192], fast=True) for j in range(0, len(rend), 192)])
        F_ = score_pairs(V, enc_x, enc_r, owner, hyps)
        for i in range(len(bi)):
            sel = [j for j, o in enumerate(owner) if o == i]
            L = max(1, len(bh[i]))
            kinds = np.zeros((len(sel), len(KIND)), dtype=np.float32)
            pos = np.zeros((len(sel), 1), dtype=np.float32)
            for r, j in enumerate(sel):
                kinds[r, KIND[meta[j][0]]] = 1
                pos[r, 0] = (meta[j][1] + 1) / L if meta[j][1] >= 0 else -1
            out.append({"hyps": [hyps[j] for j in sel], "x": F_[sel], "kind": kinds, "pos": pos,
                        "edit_pos": [meta[j][1] for j in sel], "g": g[i], "h": bh[i]})
        if s % (B * 20) == 0:
            print(f"  {s}/{len(imgs)}", flush=True)
    return out


def main(ds, engine):
    man = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/manifest.jsonl")]
    V = Verifier(VERIFIER[ds])
    if engine == "pretrain":
        from features import pretrain_pairs
        rows, claims = pretrain_pairs(ds, man)
    else:
        hyp = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/hyp_{engine}.jsonl"))}
        rows = [r for r in man if r["id"] in hyp]
        claims = [hyp[r["id"]]["hyp"] for r in rows]
    imgs = [Image.open(r["path"]).convert("RGB") for r in rows]
    feats = extract(V, imgs, claims)
    for r, f, c in zip(rows, feats, claims):
        f["id"], f["split"], f["claim"] = r["id"], r["split"], c
        if engine == "pretrain":
            f["pseudo_label"] = r["pseudo_label"]
        else:
            f["gt"] = r["gt"]
    torch.save(feats, f"{ROOT}/data/{ds}/locus_{engine}.pt")
    print(ds, engine, len(feats))


if __name__ == "__main__":
    torch.set_num_threads(8)  # shared machine: avoid CPU oversubscription
    main(sys.argv[1], sys.argv[2])

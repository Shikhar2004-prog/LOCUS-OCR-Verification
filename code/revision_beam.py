"""Revision [R1.4, R2.7, R2.8]: does widening candidate generation with beam search raise the share of real
errors whose truth lies in the counterfactual graph?  Coverage of G(h) U top-b TrOCR-base beam readings,
b in {1, 3, 5, 10}, on every test error.  usage: python code/revision_beam.py <DS>"""
import json, sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/storage/shikhar/ocrv_ext/code")
from common import CASE_INSENSITIVE, MAIN_ENGINE, ROOT, norm
from features import VERIFIER, Verifier, levenshtein_ops

ds = sys.argv[1]
ci = CASE_INSENSITIVE.get(ds, False)
F = torch.load(f"{ROOT}/data/{ds}/locus2_{MAIN_ENGINE[ds]}.pt", weights_only=False)
man = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/data/{ds}/manifest.jsonl")}
err = [f for f in F if f["split"] == "test" and norm(f["claim"], ci) != norm(f["gt"], ci)]
V = Verifier(VERIFIER[ds])
NB = 10
beams = []
from transformers.modeling_outputs import BaseModelOutput
for s in range(0, len(err), 32):
    imgs = [Image.open(man[f["id"]]["path"]).convert("RGB") for f in err[s:s + 32]]
    enc = V.encode(imgs, fast=True)
    with torch.no_grad():
        g = V.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=enc), max_new_tokens=48,
                             num_beams=NB, num_return_sequences=NB)
    txt = [t.strip() for t in V.proc.batch_decode(g, skip_special_tokens=True)]
    beams += [txt[i * NB:(i + 1) * NB] for i in range(len(imgs))]
res = {"ds": ds, "errors": len(err)}
ing = np.array([any(norm(h, ci) == norm(f["gt"], ci) for h in f["hyps"]) for f in err])
res["graph"] = float(ing.mean())
for b in (1, 3, 5, 10):
    inb = np.array([any(norm(h, ci) == norm(f["gt"], ci) for h in bm[:b]) for f, bm in zip(err, beams)])
    res[f"beam{b}_alone"] = float(inb.mean())
    res[f"graph+beam{b}"] = float((ing | inb).mean())
# edit distance between claim and truth (characters), to show how often errors are multi-character
ed = np.array([levenshtein_ops(norm(f["claim"], ci), norm(f["gt"], ci))[2] for f in err])
for lo, hi, n in ((1, 1, "ed1"), (2, 2, "ed2"), (3, 999, "ed3+")):
    m = (ed >= lo) & (ed <= hi)
    res[f"share_{n}"] = float(m.mean())
    res[f"graph_cov_{n}"] = float(ing[m].mean()) if m.any() else None
print(json.dumps(res))
open(f"{ROOT}/results/rev_beam.jsonl", "a").write(json.dumps(res) + "\n")

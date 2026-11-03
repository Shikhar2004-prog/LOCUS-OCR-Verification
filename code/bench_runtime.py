"""Measured throughput (words/s) on 256 FUNSD test words, one GPU, after a warm-up batch."""
import os
import json, sys, time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pydeps"))
from common import ROOT

man = [json.loads(l) for l in open(f"{ROOT}/data/FUNSD/manifest.jsonl") if '"test"' in l][:288]
hyp = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/FUNSD/hyp_tess.jsonl"))}
imgs = [Image.open(r["path"]).convert("RGB") for r in man]
claims = [hyp[r["id"]]["hyp"] for r in man]
out = {}


def timed(name, fn, params):
    fn(imgs[:32], claims[:32], man[:32])  # warm-up
    torch.cuda.synchronize()
    t = time.time()
    fn(imgs[32:], claims[32:], man[32:])
    torch.cuda.synchronize()
    out[name] = {"wps": 256 / (time.time() - t), "params": params}
    print(name, out[name], flush=True)


import locus2_features as L2
from features import Verifier
V1 = Verifier("printed"); V2 = L2.ParseqVerifier()
np1 = sum(p.numel() for p in V1.model.parameters()); np2 = sum(p.numel() for p in V2.m.parameters())
timed("LOCUS evidence extraction (TrOCR-base + PARSeq, both renderings)",
      lambda i, c, m: L2.extract(V1, V2, i, c), f"{(np1 + np2) / 1e6:.0f}M (frozen)")
import locus as LC
g = LC.LOCUS()
out["LOCUS GNN (trainable)"] = {"wps": float("nan"), "params": f"{sum(p.numel() for p in g.parameters()) / 1e6:.2f}M"}
import reocr_baselines as R
timed("PARSeq re-recognition", lambda i, c, m: R.parseq(m), f"{np2 / 1e6:.0f}M")
timed("Florence-2-base re-recognition", lambda i, c, m: R.florence(m), "0.23B")
import pytesseract
timed("Tesseract 5 (CPU, 1 thread)", lambda i, c, m: [pytesseract.image_to_string(x, config="--psm 7") for x in i], "--")
json.dump(out, open(f"{ROOT}/results/runtime.json", "w"), indent=1)

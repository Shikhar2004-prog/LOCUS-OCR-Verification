"""Revision [R1.8, R2.2, R2.3]: where does LOCUS spend its time, and how much do batching and graph pruning
buy?  Measured on 512 FUNSD test words (after a warm-up batch) on one RTX A6000.  Also times graph
construction (CPU) and the GNN forward pass separately."""
import json, sys, time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/storage/shikhar/ocrv_ext/code")
sys.path.insert(0, "/storage/shikhar/ocrv_ext/pydeps")
from common import ROOT

N = 512
man = [json.loads(l) for l in open(f"{ROOT}/data/FUNSD/manifest.jsonl") if '"test"' in l][:N + 64]
hyp = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/FUNSD/hyp_tess.jsonl"))}
imgs = [Image.open(r["path"]).convert("RGB") for r in man]
claims = [hyp[r["id"]]["hyp"] for r in man]
import locus2_features as L2
from features import Verifier
V1 = Verifier("printed"); V2 = L2.ParseqVerifier()
out = []


def run(M, B):
    L2.MAXN = M
    L2.extract(V1, V2, imgs[:64], claims[:64], B=B)  # warm-up
    torch.cuda.synchronize(); t = time.time()
    feats = L2.extract(V1, V2, imgs[64:], claims[64:], B=B)
    torch.cuda.synchronize()
    r = {"M": M, "batch": B, "wps": N / (time.time() - t), "nodes": float(np.mean([len(f["hyps"]) for f in feats]))}
    out.append(r); print(r, flush=True)
    return feats


for M, B in ((28, 32), (28, 128), (16, 128), (8, 128), (4, 128)):
    feats = run(M, B)
# graph construction + GNN forward (full graphs)
L2.MAXN = 28
feats = L2.extract(V1, V2, imgs[64:], claims[64:], B=128)
import locus as LC
for f in feats:
    f["agree"] = np.zeros(LC.N_AGREE, np.float32); f["y"] = 1
nz = LC.Norm(feats)
t = time.time(); G = [LC.build(f, nz) for f in feats]; tb = time.time() - t
m = LC.L()().to("cuda").eval()
LC.predict(m, G)
torch.cuda.synchronize(); t = time.time()
for _ in range(10):
    LC.predict(m, G)
torch.cuda.synchronize(); tg = (time.time() - t) / 10
edges = float(np.mean([g.edge_index.shape[1] for g in G]))
out.append({"graph_build_wps": N / tb, "gnn_wps": N / tg, "edges_per_graph": edges,
            "gnn_params": sum(p.numel() for p in m.parameters())})
print(out[-1])
json.dump(out, open(f"{ROOT}/results/rev_bench.json", "w"), indent=1)

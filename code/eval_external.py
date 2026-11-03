"""Metrics for external / top-venue baselines on the SAME test words as LOCUS.

  Engine MSP confidence      Hendrycks & Gimpel, ICLR'17 (threshold fitted on the k-shot support)
  ConfidNet-style learned confidence   Corbiere et al., NeurIPS'19 (MLP on recogniser evidence)
  Re-OCR agreement: TrOCR-base (AAAI'23), PARSeq (ECCV'22; binary + confidence-graded),
                    Florence-2 (CVPR'24)
  Two-engine voting
  VLM judge: Qwen2.5-VL-7B-Instruct zero-shot (P(yes)); oracle-threshold upper bound
Output: results/external.jsonl
"""
import json, os, sys

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CASE_INSENSITIVE, MAIN_ENGINE, ROOT, load_hyp, metrics, norm
from run import best_threshold, sample_support

SEEDS = [0, 1, 2, 3, 4]
KS = [5, 10, 25, 50, 100, "full"]


def log(row):
    with open(f"{ROOT}/results/external.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def run(ds):
    E = MAIN_ENGINE[ds]
    ci = CASE_INSENSITIVE.get(ds, False)
    F_ = torch.load(f"{ROOT}/data/{ds}/locus_{E}.pt", weights_only=False)
    C = {f["id"]: f for f in torch.load(f"{ROOT}/data/{ds}/feat_{E}.pt", weights_only=False)}
    tr = [f for f in F_ if f["split"] == "train"]
    te = [f for f in F_ if f["split"] == "test"]
    lab = lambda f: int(norm(f["claim"], ci) == norm(f["gt"], ci))
    ytr, yte = np.array([lab(f) for f in tr]), np.array([lab(f) for f in te])
    same = lambda a, b: float(norm(a, ci) == norm(b, ci))
    # ---- label-free agreement baselines
    log({"ds": ds, "method": "ReOCR TrOCR-base agree", "k": 0, **metrics(np.array([same(f["claim"], f["g"]) for f in te]), yte)})
    other = {"tess": "trocr", "trocr": "trocrB"}[E]
    H2 = load_hyp(ds, other)
    log({"ds": ds, "method": "Two-engine voting", "k": 0, **metrics(np.array([same(f["claim"], H2[f["id"]]["hyp"]) for f in te]), yte)})
    ro = f"{ROOT}/results/reocr_{ds}.json"
    if os.path.exists(ro):
        R = json.load(open(ro))
        if all("parseq" in R.get(f["id"], {}) for f in te):
            a = np.array([same(f["claim"], R[f["id"]]["parseq"]) for f in te])
            c = np.array([R[f["id"]]["parseq_conf"] for f in te])
            log({"ds": ds, "method": "ReOCR PARSeq agree", "k": 0, **metrics(a, yte)})
            log({"ds": ds, "method": "ReOCR PARSeq graded", "k": 0, **metrics(np.where(a > 0, c, -c), yte, thr=0.0)})
        if all("florence" in R.get(f["id"], {}) for f in te):
            log({"ds": ds, "method": "ReOCR Florence-2 agree", "k": 0,
                 **metrics(np.array([same(f["claim"], R[f["id"]]["florence"]) for f in te]), yte)})
    vp = f"{ROOT}/results/vlm_{ds}.json"
    if os.path.exists(vp):
        Vs = json.load(open(vp))
        v = np.array([Vs[f["id"]] for f in te])
        log({"ds": ds, "method": "VLM judge Qwen2.5-VL-7B", "k": 0, **metrics(v, yte, thr=0.5)})
        log({"ds": ds, "method": "VLM judge Qwen2.5-VL-7B (oracle thr)", "k": 0,
             **metrics(v, yte, thr=best_threshold(v, yte))})
    # ---- k-shot baselines (same supports as LOCUS)
    H = load_hyp(ds, E)
    ctr = np.array([H[f["id"]]["conf"] for f in tr]); cte = np.array([H[f["id"]]["conf"] for f in te])
    lik = lambda f: C[f["id"]]["glob"][[0, 9, 10, 13, 7]]  # recogniser claim likelihood statistics
    Ltr = np.stack([np.r_[lik(f), H[f["id"]]["conf"]] for f in tr])
    Lte = np.stack([np.r_[lik(f), H[f["id"]]["conf"]] for f in te])
    mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-6
    for k in KS:
        for s in SEEDS:
            if k == "full":
                sup = np.random.RandomState(s).permutation(len(tr))[int(0.15 * len(tr)):]
            else:
                sup = sample_support(ytr, k, s)
            t = best_threshold(ctr[sup], ytr[sup])
            log({"ds": ds, "method": "Engine MSP confidence", "k": k, "seed": s, **metrics(cte, yte, thr=t)})
            mlp = MLPClassifier((32,), alpha=1e-2, max_iter=2000, random_state=s).fit((Ltr[sup] - mu) / sd, ytr[sup])
            log({"ds": ds, "method": "ConfidNet-style learned conf.", "k": k, "seed": s,
                 **metrics(mlp.predict_proba((Lte - mu) / sd)[:, 1], yte)})
    print(ds, "done")


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        run(ds)

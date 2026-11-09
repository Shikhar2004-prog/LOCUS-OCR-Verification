"""Export a deployable LOCUS verifier per domain for the demo application.

For each dataset this script
  1. pretrains the LOCUS GNN label-free (seed 0, same recipe as the paper: 15 epochs on
     engine-agreement positives + synthetic OCR-style corruptions),
  2. fits the few-shot fusion head on k=25 labelled words per class (seed-0 support set);
     its inputs are the label-free priors [pretrained LOCUS logit, rho-margin rule,
     PARSeq graded agreement, TrOCR agreement],
  3. evaluates the exported model on the held-out test split and stores the numbers with it.

The fusion head is the head LOCUS selects most often at k=25 (Sec. 3 of the paper); the
anchored residual head additionally needs character-level evidence and is not exported.
Output: app/models/locus_<DS>.pt

usage: python app/export_models.py FUNSD CORD SROIE IAM IIIT5K
"""
import os, sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import locus as Lc  # noqa: E402
from common import CASE_INSENSITIVE, metrics  # noqa: E402
from features import VERIFIER  # noqa: E402
from run import sample_support  # noqa: E402

K, SEED = 25, 0


def export(ds):
    Lc.index_pretrain_truth(ds)
    B = Lc.Bundle(ds)
    state = Lc.pretrained(B, SEED, Lc.L())
    zs = Lc.zero_shot_logits(B, SEED, state=state)
    Ptr, Pte = Lc.priors(B, SEED, zs)
    sup = sample_support(B.ytr, K, SEED)
    head = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(Ptr[sup], B.ytr[sup])
    p = head.predict_proba(Pte)[:, 1]
    met = metrics(p, B.yte)
    zs_met = metrics(1 / (1 + np.exp(-zs[1])), B.yte)
    print(f"{ds}: exported model test macro-F1 {met['macro_f1']:.3f} AUROC {met['auroc']:.3f} "
          f"(zero-shot GNN {zs_met['macro_f1']:.3f}/{zs_met['auroc']:.3f})", flush=True)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    torch.save({
        "ds": ds, "verifier": VERIFIER[ds], "case_insensitive": CASE_INSENSITIVE.get(ds, False),
        "state": {k: v.cpu() for k, v in state.items()},
        "norm_m": B.nz.m.astype(np.float32), "norm_s": B.nz.s.astype(np.float32),
        "head_coef": head.coef_[0].astype(np.float32), "head_intercept": float(head.intercept_[0]),
        "k": K, "seed": SEED, "n_support": int(len(sup)),
        "test_metrics": {k: float(v) for k, v in met.items() if np.isscalar(v)},
        "zero_shot_test_metrics": {k: float(v) for k, v in zs_met.items() if np.isscalar(v)},
        "n_test": int(len(B.yte)), "test_error_rate": float(1 - B.yte.mean()),
    }, os.path.join(HERE, "models", f"locus_{ds}.pt"))


if __name__ == "__main__":
    torch.set_num_threads(8)
    for ds in sys.argv[1:]:
        export(ds)

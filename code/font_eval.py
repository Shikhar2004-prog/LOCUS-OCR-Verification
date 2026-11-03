"""Rendering sensitivity: how much do the font / glyph size of R(v) matter?
For each variant (same first 1000 train + 1000 test crops per dataset):
  rho(claim)    AUROC of the claim's summed render-normalised ratio (no training)
  rule          AUROC / macro-F1 of the training-free rho-margin local-optimality rule
  LOCUS         GNN trained on the 1000 labelled train crops (3 seeds, early stopping, no pretraining)
Output: results/font_sensitivity.jsonl"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import locus as LC
from common import MAIN_ENGINE, ROOT, metrics

N_EVAL = 1000  # every variant is evaluated on the same first 1000 train / 1000 test crops
VARIANTS = [("sans", "Liberation Sans (default)"), ("serif", "Liberation Serif"), ("mono", "Liberation Mono"),
            ("dejavu", "DejaVu Sans"), ("comic", "Comic Neue (handwriting-like)"),
            ("sans0.6", "Liberation Sans, glyphs 0.6x"), ("sans0.9", "Liberation Sans, glyphs 0.9x")]


def run(ds):
    E = MAIN_ENGINE[ds]
    for tag, name in VARIANTS:
        p = f"{ROOT}/data/{ds}/locus2_{E}_r{tag}.pt"
        if not os.path.exists(p):
            print("missing", p); continue
        Fs = torch.load(p, weights_only=False)
        for f in Fs:
            f["y"] = LC.label_of(f, ds); f["agree"] = LC.agree_vec(f, ds)
        tr = [f for f in Fs if f["split"] == "train"][:N_EVAL]; te = [f for f in Fs if f["split"] == "test"][:N_EVAL]
        nz = LC.Norm(tr)
        ytr = np.array([f["y"] for f in tr]); yte = np.array([f["y"] for f in te])
        rho = np.array([f["x"][0, 2] + f["x"][0, 13] for f in te])
        rule = np.array([LC.zero_shot_score(f) for f in te])
        gtr = [LC.build(f, nz, rep=0 if f["y"] == 1 else -1) for f in tr]
        gte = [LC.build(f, nz, rep=LC.repair_target(f, ds)) for f in te]
        P = []
        for s in (0, 1, 2):
            perm = np.random.RandomState(s).permutation(len(gtr))
            nv = int(0.15 * len(perm))
            m = LC.train(LC.L(), [gtr[i] for i in perm[nv:]], s, epochs=60, val=[gtr[i] for i in perm[:nv]])
            P.append(LC.predict(m, gte)[0])
        row = {"ds": ds, "variant": name, "tag": tag, "auroc_rho_claim": metrics(rho, yte)["auroc"],
               "auroc_rule": metrics(rule, yte)["auroc"], "f1_rule": metrics(rule, yte, thr=0.0)["macro_f1"],
               **{f"locus_{k}": v for k, v in metrics(np.mean(P, 0), yte).items()}}
        with open(f"{ROOT}/results/font_sensitivity.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(ds, name, round(row["auroc_rho_claim"], 3), round(row["auroc_rule"], 3), round(row["locus_macro_f1"], 3), flush=True)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        run(ds)

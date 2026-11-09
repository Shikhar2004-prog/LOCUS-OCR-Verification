"""Measure the demo pipeline on the synthetic documents, whose word-level truth is known.

For every Tesseract word: is the claim correct (exact match with the typeset word)?
Compares LOCUS's flags with Tesseract-confidence flags at the SAME review budget.
Output: app/examples/demo_eval.json (used by the README and the video).
usage: python app/evaluate_demo.py [MODEL ...]
"""
import json, os, sys, time

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from locus_engine import LocusEngine, verify_page  # noqa: E402
from make_demo_docs import match_truth  # noqa: E402

DOCS = ["invoice", "receipt", "claim_form"]


def evaluate(model):
    eng = LocusEngine(model)
    out = {}
    for name in DOCS:
        page = Image.open(os.path.join(HERE, "examples", f"{name}.png")).convert("RGB")
        gt = json.load(open(os.path.join(HERE, "examples", f"{name}.json")))["words"]
        t = time.time()
        res = verify_page(eng, page)
        dt = time.time() - t
        rows = []
        for r in res:
            g = match_truth(gt, r.box)
            rows.append({"claim": r.claim, "truth": g["text"] if g else None, "p": r.p_correct,
                         "conf": r.engine_conf, "repair": r.repair, "box": r.box, "parseq": r.parseq, "trocr": r.trocr,
                         "correct": int(g is not None and g["text"] == r.claim)})
        y = np.array([w["correct"] for w in rows])
        p = np.array([w["p"] for w in rows])
        c = np.array([w["conf"] for w in rows])
        flag = p < 0.5
        n_err = int((y == 0).sum())
        budget = int(flag.sum())
        conf_flag = np.zeros(len(c), bool)
        conf_flag[np.argsort(c)[:budget]] = True  # Tesseract confidence, same number of words reviewed
        pq_flag = np.array([w["parseq"] != w["claim"] for w in rows])  # re-OCR baseline: disagree with PARSeq
        rep_ok = sum(1 for w, f in zip(rows, flag) if f and not w["correct"] and w["truth"] and w["repair"] == w["truth"])
        out[name] = {
            "words": len(rows), "errors": n_err, "flagged": budget,
            "errors_caught": int((flag & (y == 0)).sum()), "false_alarms": int((flag & (y == 1)).sum()),
            "errors_caught_by_conf_same_budget": int((conf_flag & (y == 0)).sum()),
            "reocr_parseq_flagged": int(pq_flag.sum()), "reocr_parseq_caught": int((pq_flag & (y == 0)).sum()),
            "auroc_locus": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "auroc_tess_conf": float(roc_auc_score(y, c)) if 0 < y.sum() < len(y) else None,
            "repairs_exact": rep_ok, "seconds": round(dt, 1), "rows": rows}
        s = out[name]
        print(f"[{model}] {name}: {s['words']} words, {n_err} errors | LOCUS flags {budget}, catches "
              f"{s['errors_caught']} ({s['false_alarms']} false alarms), exact repairs {rep_ok} | "
              f"Tesseract conf at same budget catches {s['errors_caught_by_conf_same_budget']} | AUROC "
              f"{s['auroc_locus']:.3f} vs {s['auroc_tess_conf']:.3f} | PARSeq re-OCR flags {s['reocr_parseq_flagged']} catches {s['reocr_parseq_caught']} | {dt:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    allres = {m: evaluate(m) for m in (sys.argv[1:] or ["SROIE"])}
    p = os.path.join(HERE, "examples", "demo_eval.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(allres)
    json.dump(old, open(p, "w"), indent=1)

"""LaTeX table bodies for the revision experiments (results/rev_*.jsonl -> journal_revision/tables/rev_*.tex)."""
import json, os

import numpy as np

ROOT = "/storage/shikhar/ocrv_ext"
T = f"{ROOT}/journal_revision/tables"
DSS = ["FUNSD", "CORD", "SROIE", "IAM", "IIIT5K"]


def rd(name):
    p = f"{ROOT}/results/rev_{name}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def f3(v):
    return "--" if v is None else f"{v:.3f}".replace("0.", ".", 1)


def noise():
    R = rd("noise")
    lines, summ = [], {}
    for var, disp in (("full", "LOCUS (full), 3 independent re-runs"), ("no message passing", "no message passing, 3 re-runs")):
        cells = []
        for ds in DSS:
            v = [r["macro_f1"] for r in R if r["ds"] == ds and r["variant"] == var]
            a = [r["auroc"] for r in R if r["ds"] == ds and r["variant"] == var]
            summ[(var, ds)] = (v, a)
            cells.append(f"{f3(min(v))}--{f3(max(v))}" if v else "--")
        lines.append(f"{disp} & " + " & ".join(cells) + " & \\\\")
    open(f"{T}/rev_noise.tex", "w").write("\n".join(lines) + "\n")
    out = {}
    for ds in DSS:
        vf, af = summ[("full", ds)]; vn, an = summ[("no message passing", ds)]
        if vf and vn:
            out[ds] = {"full_range_f1": max(vf) - min(vf), "full_range_auc": max(af) - min(af),
                       "nomp_minus_full_f1": float(np.mean(vn) - np.mean(vf)), "nomp_minus_full_auc": float(np.mean(an) - np.mean(af))}
    return out


def gsize():
    R = rd("gsize")
    B = json.load(open(f"{ROOT}/results/rev_bench.json")) if os.path.exists(f"{ROOT}/results/rev_bench.json") else []
    wps = {f"M={b['M']}" + (" (default)" if b["M"] == 28 else ""): b["wps"] for b in B if "M" in b and b["batch"] == 128}
    names = ["M=2", "M=4", "M=8", "M=16", "M=28 (default)", "readings only", "edits only (no recogniser readings)"]
    lines = []
    for n in names:
        cells = []
        nodes = [r["nodes"] for r in R if r["variant"] == n and r["ds"] == "FUNSD"]
        for ds in DSS:
            r = [x for x in R if x["ds"] == ds and x["variant"] == n]
            cells.append(f"{f3(r[0]['macro_f1'])}/{f3(r[0]['auroc'])}" if r else "--")
        cov = [x for x in R if x["ds"] == "IAM" and x["variant"] == n]
        w = wps.get(n)
        label = n.replace("M=", "$M{=}$").replace(" (no recogniser readings)", " (no readings)")
        s_nodes = f"{nodes[0]:.1f}" if nodes else "--"
        s_w = f"{w:.0f}" if w else "--"
        s_cov = f"{100 * cov[0]['truth_in_graph']:.1f}" if cov else "--"
        lines.append(f"{label} & {s_nodes} & {s_w} & " + " & ".join(cells) + f" & {s_cov} \\\\")
    open(f"{T}/rev_gsize.tex", "w").write("\n".join(lines) + "\n")


def pseudo():
    R = rd("pseudo")
    names = ["agreement (default)", "confidence-thresholded (drop lowest 30%)", "confidence-weighted (rank weights)",
             "three-engine agreement (+PARSeq)"]
    disp = {"agreement (default)": "two-engine agreement (default)", "confidence-thresholded (drop lowest 30%)":
            "+ confidence threshold (drop lowest 30\\%)", "confidence-weighted (rank weights)": "+ confidence weighting",
            "three-engine agreement (+PARSeq)": "three-engine agreement"}
    lines = []
    for n in names:
        cells = []
        for ds in DSS:
            r = [x for x in R if x["ds"] == ds and x["variant"] == n]
            cells.append(f"{100 * r[0]['precision']:.1f} \\,|\\, {f3(r[0]['macro_f1'])}/{f3(r[0]['auroc'])}" if r else "--")
        lines.append(f"{disp[n]} & " + " & ".join(cells) + " \\\\")
    open(f"{T}/rev_pseudo.tex", "w").write("\n".join(lines) + "\n")


def xcal():
    R = rd("xcal")
    meth = ["transfer, threshold 0.5", "label-free prior-shift EM", "target threshold from 5 labels/class",
            "target threshold from 25 labels/class", "in-engine reference (k=25)"]
    lines = []
    for ds in ["FUNSD", "CORD", "SROIE", "IIIT5K"]:
        for sn, tn in (("tess", "trocr"), ("trocr", "tess")):
            rows = {x["method"]: x for x in R if x["ds"] == ds and x["train"] == sn}
            if not rows:
                continue
            nm = {"tess": "Tess.", "trocr": "TrOCR-s"}
            err = 100 * (1 - rows[meth[0]]["target_correct_rate"])
            cells = [f3(rows[m]["macro_f1"]) if m in rows else "--" for m in meth]
            e0, e1 = rows[meth[0]].get("ece"), rows[meth[1]].get("ece")
            lines.append(f"{ds} & {nm[sn]}$\\rightarrow${nm[tn]} & {err:.1f} & " + " & ".join(cells)
                         + f" & {f3(e0)}/{f3(e1)} \\\\")
    open(f"{T}/rev_xcal.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    for fn in (gsize, pseudo, xcal):
        try:
            fn(); print("ok", fn.__name__)
        except Exception as e:
            print("FAILED", fn.__name__, e)
    print(json.dumps(noise(), indent=1))

"""Revision analyses that need no new training (reviewer comments R1.3-R1.6, R1.8, R2.3-R2.8).
Reads the stored k=25 predictions (results/preds_<DS>.npy), the evidence files and the baseline outputs;
writes results/rev_analysis.json and LaTeX table bodies journal/tables/rev_*.tex."""
import hashlib, json, os, sys

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, "/storage/shikhar/ocrv_ext/code")
from common import CASE_INSENSITIVE, MAIN_ENGINE, ROOT, load_hyp, metrics, norm
from features import levenshtein_ops

DSS = ["FUNSD", "CORD", "SROIE", "IAM", "IIIT5K"]
KEY = "LOCUS-GNN (pretrained, fine-tuned)|25"
T = f"{ROOT}/journal/tables"
OUT = {}


def load(ds):
    st = np.load(f"{ROOT}/results/preds_{ds}.npy", allow_pickle=True).item()
    ids, y = st["ids"], np.asarray(st["y"])
    ci = CASE_INSENSITIVE.get(ds, False)
    F = {f["id"]: f for f in torch.load(f"{ROOT}/data/{ds}/locus2_{MAIN_ENGINE[ds]}.pt", weights_only=False)}
    R = json.load(open(f"{ROOT}/results/reocr_{ds}.json"))
    H = load_hyp(ds, MAIN_ENGINE[ds])
    f = [F[i] for i in ids]
    d = dict(ids=ids, y=y, ci=ci, f=f, P=st[KEY], p=st[KEY].mean(0))
    d["h"] = [norm(H[i]["hyp"], ci) for i in ids]
    d["gt"] = [norm(x["gt"], ci) for x in f]
    d["g1"] = [norm(x["g"], ci) for x in f]
    d["pq"] = [norm(R[i]["parseq"], ci) for i in ids]
    d["pqc"] = np.array([R[i]["parseq_conf"] for i in ids])
    d["a_pq"] = np.array([a == b for a, b in zip(d["h"], d["pq"])])
    d["a_tr"] = np.array([a == b for a, b in zip(d["h"], d["g1"])])
    d["pq_graded"] = np.where(d["a_pq"], d["pqc"], -d["pqc"])
    d["in_graph"] = np.array([any(norm(v, ci) == g for v in x["hyps"]) for x, g in zip(f, d["gt"])])
    d["ed"] = np.array([levenshtein_ops(a, b)[2] for a, b in zip(d["h"], d["gt"])])
    V = json.load(open(f"{ROOT}/results/vlm_{ds}.json"))
    d["vlm"] = np.array([V[i] for i in ids])
    return d


def pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")


# --------------------------------------------------------------------------------- (a) graph coverage
def coverage(D):
    lines, res = [], {}
    for ds in DSS:
        d = D[ds]
        e = d["y"] == 0
        rej = d["p"] < 0.5
        r = {"errors": int(e.sum()), "in_graph": float(d["in_graph"][e].mean())}
        for nm, m in (("in", e & d["in_graph"]), ("out", e & ~d["in_graph"])):
            r[f"locus_recall_{nm}"] = float(rej[m].mean()) if m.any() else None
            r[f"pq_recall_{nm}"] = float((~d["a_pq"])[m].mean()) if m.any() else None
            r[f"n_{nm}"] = int(m.sum())
        # ranking: correct words vs errors whose truth is NOT in the graph
        m = (d["y"] == 1) | (e & ~d["in_graph"])
        r["auroc_out"] = auc(d["y"][m], d["p"][m]); r["auroc_out_pq"] = auc(d["y"][m], d["pq_graded"][m])
        m = (d["y"] == 1) | (e & d["in_graph"])
        r["auroc_in"] = auc(d["y"][m], d["p"][m])
        for nm, lo, hi in (("ed1", 1, 1), ("ed2", 2, 2), ("ed3", 3, 10 ** 6)):
            mm = e & (d["ed"] >= lo) & (d["ed"] <= hi)
            r[f"share_{nm}"] = float(mm.sum() / e.sum())
            r[f"cov_{nm}"] = float(d["in_graph"][mm].mean()) if mm.any() else None
            r[f"locus_recall_{nm}"] = float(rej[mm].mean()) if mm.any() else None
        res[ds] = r
        lines.append(f"{ds} & {r['errors']} & {pct(r['in_graph'])} & {pct(r['locus_recall_in'])} & {pct(r['locus_recall_out'])} & "
                     f"{pct(r['pq_recall_out'])} & {r['auroc_in']:.3f} & {r['auroc_out']:.3f} & {r['auroc_out_pq']:.3f} & "
                     f"{pct(r['share_ed1'])}/{pct(r['share_ed2'])}/{pct(r['share_ed3'])} & "
                     f"{pct(r['locus_recall_ed1'])}/{pct(r['locus_recall_ed2'])}/{pct(r['locus_recall_ed3'])} \\\\")
    open(f"{T}/rev_coverage.tex", "w").write("\n".join(lines) + "\n")
    OUT["coverage"] = res


# --------------------------------------------------------------------------------- (b) correlated failures
def correlated(D):
    lines, res = [], {}
    for ds in DSS:
        d = D[ds]
        g1w = np.array([a != b for a, b in zip(d["g1"], d["gt"])])
        pqw = np.array([a != b for a, b in zip(d["pq"], d["gt"])])
        same = np.array([a == b for a, b in zip(d["g1"], d["pq"])])
        claim_same = np.array([h == a for h, a in zip(d["h"], d["pq"])])
        dec = d["p"] >= 0.5
        strata = [("both wrong, same reading", g1w & pqw & same),
                  ("\\quad of which claim = that reading", g1w & pqw & same & claim_same),
                  ("both wrong, different readings", g1w & pqw & ~same)]
        res[ds] = {}
        first = True
        for nm, m in strata:
            n = int(m.sum())
            if n < 10:
                cells = [str(n), "--", "--", "--", "--", "--"]
            else:
                err = 1 - d["y"][m].mean()
                cells = [str(n), pct(err), pct((dec == d["y"].astype(bool))[m].mean()),
                         pct((d["a_pq"] == d["y"].astype(bool))[m].mean()),
                         pct((d["a_tr"] == d["y"].astype(bool))[m].mean()),
                         f"{auc(d['y'][m], d['p'][m]):.3f}" if len(np.unique(d['y'][m])) == 2 else "--"]
            res[ds][nm] = cells
            lines.append(f"{ds if first else ''} & {nm} & " + " & ".join(cells) + " \\\\")
            first = False
        lines.append("\\midrule")
    open(f"{T}/rev_correlated.tex", "w").write("\n".join(lines[:-1]) + "\n")
    OUT["correlated"] = res


# --------------------------------------------------------------------------------- (c) significance robustness
def boot(y, a, b, groups, B=2000, thr_a=0.5, thr_b=0.0, seed=0):
    ug = np.unique(groups)
    mem = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.RandomState(seed)
    dA, dF = [], []
    for _ in range(B):
        ix = np.concatenate([mem[g] for g in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y[ix])) < 2:
            continue
        dA.append(roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]))
        dF.append(f1_score(y[ix], a[ix] >= thr_a, average="macro") - f1_score(y[ix], b[ix] >= thr_b, average="macro"))
    return np.array(dA), np.array(dF)


def significance(D):
    """Design effect: how much wider are document-level than word-level intervals on FUNSD/CORD?
    The implied intra-document correlation is used to inflate the word-level SROIE/IAM intervals."""
    res = {"deff": {}, "inflated": {}, "seeds": {}}
    icc = []
    for ds in ("FUNSD", "CORD"):
        d = D[ds]
        Dm = json.load(open(f"{ROOT}/data/{ds}/doc_ids.json"))
        groups = np.array([Dm[i] for i in d["ids"]])
        m = len(groups) / len(np.unique(groups))
        for bn, b, tb in (("PARSeq graded", d["pq_graded"], 0.0), ("TrOCR re-OCR", d["a_tr"].astype(float), 0.5)):
            wA, wF = boot(d["y"], d["p"], b, np.arange(len(groups)), thr_b=tb)
            cA, cF = boot(d["y"], d["p"], b, groups, thr_b=tb)
            for nm, w, c in (("AUROC", wA, cA), ("F1", wF, cF)):
                deff = (c.std() / w.std()) ** 2
                rho = max(0.0, (deff - 1) / (m - 1))
                icc.append(rho)
                res["deff"][f"{ds}|{bn}|{nm}"] = {"deff": float(deff), "m": float(m), "icc": float(rho)}
    rho_max = max(icc)
    res["icc_max"] = float(rho_max)
    # cluster sizes: SROIE test lines per receipt (347 official test receipts); IAM words per writer (657 writers)
    msz = {"SROIE": 4000 / 347, "IAM": 4000 / 657}
    lines = []
    for ds in ("SROIE", "IAM"):
        d = D[ds]
        deff = 1 + (msz[ds] - 1) * rho_max
        for bn, b, tb in (("PARSeq re-OCR (graded)", d["pq_graded"], 0.0), ("TrOCR-base re-OCR", d["a_tr"].astype(float), 0.5)):
            wA, wF = boot(d["y"], d["p"], b, np.arange(len(d["y"])), thr_b=tb)
            cells = []
            for w in (wA, wF):
                mu, half = w.mean(), 1.96 * w.std() * np.sqrt(deff)
                cells += [f"{mu:+.3f}", f"[{mu - half:+.3f}, {mu + half:+.3f}]", "yes" if (mu - half > 0 or mu + half < 0) else "no"]
            res["inflated"][f"{ds}|{bn}"] = cells
            lines.append(f"{ds} & {deff:.2f} & {bn} & " + " & ".join(cells) + " \\\\")
    open(f"{T}/rev_deff.tex", "w").write("\n".join(lines) + "\n")
    # seed-level consistency: each of the five k=25 support sets separately vs the strongest baseline
    best = {"FUNSD": "pq", "CORD": "pq", "SROIE": "tr", "IAM": "pq", "IIIT5K": "pq"}
    for ds in DSS:
        d = D[ds]
        b = d["pq_graded"] if best[ds] == "pq" else d["a_tr"].astype(float)
        tb = 0.0 if best[ds] == "pq" else 0.5
        dA = [auc(d["y"], P) - auc(d["y"], b) for P in d["P"]]
        dF = [f1_score(d["y"], P >= 0.5, average="macro") - f1_score(d["y"], b >= tb, average="macro") for P in d["P"]]
        res["seeds"][ds] = {"dAUROC": [float(x) for x in dA], "dF1": [float(x) for x in dF]}
    OUT["significance"] = res


# --------------------------------------------------------------------------------- (d) render normalisation by subgroup
def render_groups(D):
    lines, res = [], {}
    for ds in DSS:
        d = D[ds]
        h = [x["claim"] for x in d["f"]]
        x = np.stack([f["x"][0] for f in d["f"]])
        groups = {"digits": np.array([any(c.isdigit() for c in s) for s in h]),
                  "upper-case": np.array([s.isalpha() and s.isupper() and len(s) > 1 for s in h]),
                  "lower/mixed words": np.array([s.isalpha() and not s.isupper() for s in h])}
        res[ds] = {}
        cells = []
        for gn, m in groups.items():
            if m.sum() < 50 or len(np.unique(d["y"][m])) < 2:
                cells += ["--", "--"]; continue
            a_raw, a_rho = auc(d["y"][m], x[m, 0]), auc(d["y"][m], x[m, 2])
            res[ds][gn] = {"n": int(m.sum()), "raw": a_raw, "rho": a_rho}
            cells += [f"{a_raw:.3f}", f"{a_rho:.3f}"]
        lines.append(f"{ds} & " + " & ".join(cells) + " \\\\")
    open(f"{T}/rev_render_groups.tex", "w").write("\n".join(lines) + "\n")
    OUT["render_groups"] = res


# --------------------------------------------------------------------------------- (e) cascade
def cascade(D, wps_locus=17.9, wps_pq=372.3):
    lines, res = [], {}
    for ds in DSS:
        d = D[ds]
        res[ds] = {}
        base = metrics(d["p"], d["y"])
        res[ds]["LOCUS on all"] = {"routed": 1.0, **base}
        for tau in (0.99, 0.95, 0.9):
            early = d["a_pq"] & (d["pqc"] >= tau)
            s = np.where(early, 1.0 + d["pqc"], d["p"])
            m = metrics(s, d["y"])
            frac = 1 - early.mean()
            wps = 1 / (1 / wps_pq + frac / wps_locus)
            res[ds][f"tau={tau}"] = {"routed": float(frac), "wps": float(wps), "early_err": float(1 - d["y"][early].mean()), **m}
        a, b = res[ds]["LOCUS on all"], res[ds]["tau=0.95"]
        lines.append(f"{ds} & {a['macro_f1']:.3f} & {a['auroc']:.3f} & {pct(b['routed'])} & {pct(b['early_err'])} & "
                     f"{b['macro_f1']:.3f} & {b['auroc']:.3f} & {b['wps']:.0f} \\\\")
    open(f"{T}/rev_cascade.tex", "w").write("\n".join(lines) + "\n")
    OUT["cascade"] = res


# --------------------------------------------------------------------------------- (f) claim length
def length_bins(D):
    lines, res = [], {}
    for ds in DSS:
        d = D[ds]
        L = np.array([len(s) for s in d["h"]])
        res[ds] = {}
        cells = []
        for nm, lo, hi in (("1-3", 0, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16+", 16, 10 ** 6)):
            m = (L >= lo) & (L <= hi)
            if m.sum() < 30 or len(np.unique(d["y"][m])) < 2:
                cells += ["--"]; continue
            e = m & (d["y"] == 0)
            r = {"n": int(m.sum()), "err": float(1 - d["y"][m].mean()),
                 "f1": float(f1_score(d["y"][m], d["p"][m] >= 0.5, average="macro")),
                 "f1_pq": float(f1_score(d["y"][m], d["a_pq"][m], average="macro")),
                 "cov": float(d["in_graph"][e].mean()) if e.any() else None,
                 "nodes": float(np.mean([len(d["f"][i]["hyps"]) for i in np.where(m)[0]]))}
            res[ds][nm] = r
            cells += [f"{r['f1']:.3f} ({r['n']})"]
        lines.append(f"{ds} & " + " & ".join(cells) + " \\\\")
    open(f"{T}/rev_length.tex", "w").write("\n".join(lines) + "\n")
    OUT["length"] = res


# --------------------------------------------------------------------------------- (g) leakage audit
def leakage():
    res = {}
    for ds in DSS:
        man = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/manifest.jsonl")]
        split = {r["id"]: r["split"] for r in man}
        P = torch.load(f"{ROOT}/data/{ds}/locus2_pretrain.pt", weights_only=False)
        pre_ids = {f["id"] for f in P}
        hs = {}
        for r in man:
            hs.setdefault(hashlib.md5(open(r["path"], "rb").read()).hexdigest(), set()).add(r["split"])
        dup = sum(1 for v in hs.values() if len(v) > 1)
        res[ds] = {"pretrain_images": len(pre_ids), "pretrain_from_test": sum(split.get(i) == "test" for i in pre_ids),
                   "pretrain_not_in_train": sum(split.get(i) != "train" for i in pre_ids),
                   "train_test_identical_crops": dup, "test": sum(v == "test" for v in split.values())}
    OUT["leakage"] = res


if __name__ == "__main__":
    D = {ds: load(ds) for ds in DSS}
    for fn in (coverage, correlated, significance, render_groups, cascade, length_bins):
        fn(D); print("ok", fn.__name__, flush=True)
    leakage(); print("ok leakage")
    json.dump(OUT, open(f"{ROOT}/results/rev_analysis.json", "w"), indent=1, default=float)

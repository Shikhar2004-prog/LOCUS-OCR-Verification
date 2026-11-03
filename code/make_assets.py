"""Generate every table (paper/tables/*.tex) and figure (paper/figs/*.pdf) from result files.
Nothing in the paper is typed by hand: re-running this script regenerates all numbers."""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CASE_INSENSITIVE, MAIN_ENGINE, ROOT, load_hyp, metrics, norm

P = f"{ROOT}/paper"
os.makedirs(f"{P}/tables", exist_ok=True)
os.makedirs(f"{P}/figs", exist_ok=True)
DSS = ["FUNSD", "CORD", "SROIE", "IAM", "IIIT5K"]
DSNAME = {"FUNSD": "FUNSD", "CORD": "CORD", "SROIE": "SROIE", "IAM": "IAM", "IIIT5K": "IIIT5K"}
# validated categorical slots (dataviz reference palette, light mode)
C_OURS, C_B1, C_B2, C_B3, C_MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "legend.frameon": False, "pdf.fonttype": 42})


def rd(name):
    p = f"{ROOT}/results/{name}.jsonl"
    return pd.read_json(p, lines=True) if os.path.exists(p) else pd.DataFrame()


def fmt(v, best=False, und=False, rank=0, color=True):
    s = f"{v:.3f}" if pd.notna(v) else "--"
    s = s.replace("0.", ".", 1) if s.startswith("0.") else s
    if best or rank == 1:
        s = (r"\best{" if color else r"\textbf{") + s + "}"
    elif und or rank == 2:
        s = (r"\second{" if color else r"\underline{") + s + "}"
    return s


def ranks(values, lower=False, eligible=None):
    """1 = best, 2 = second best, 0 = other. Ties (equal at 3 decimals) share a rank; ineligible or
    missing entries are never marked."""
    eligible = eligible or [True] * len(values)
    rv = [round(float(v), 3) if (e and pd.notna(v)) else None for v, e in zip(values, eligible)]
    uniq = sorted({v for v in rv if v is not None}, reverse=not lower)
    out = []
    for v in rv:
        if v is None:
            out.append(0)
        elif v == uniq[0]:
            out.append(1)
        elif len(uniq) > 1 and v == uniq[1]:
            out.append(2)
        else:
            out.append(0)
    return out


def summarise():
    """One tidy frame: method x ds x k -> metrics (ensemble rows for learned methods; seed means
    for threshold baselines; k=0 rows for label-free baselines)."""
    rows = []
    M = rd("locus_main")
    if len(M):
        M["k"] = M["k"].astype(str)
        M["seed"] = M["seed"].astype(str)
        ens = M[M.seed == "ens"]
        rows.append(ens)
        sm = M[(M.seed != "ens") & (M.seed != "-1")]
        thr_methods = set(sm.method) - set(ens.method)
        g = sm[sm.method.isin(thr_methods)].groupby(["ds", "method", "k"]).mean(numeric_only=True).reset_index()
        rows.append(g)
        rows.append(M[M.seed == "-1"])
    X = rd("external")
    if len(X):
        X["k"] = X["k"].astype(str)
        X["seed"] = X["seed"].astype(str) if "seed" in X else "nan"
        rows.append(X.groupby(["ds", "method", "k"]).mean(numeric_only=True).reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def seed_std(method, ds, k, metric="macro_f1"):
    M = rd("locus_main")
    if not len(M):
        return np.nan
    M["k"], M["seed"] = M["k"].astype(str), M["seed"].astype(str)
    v = M[(M.method == method) & (M.ds == ds) & (M.k == str(k)) & (~M.seed.isin(["ens", "-1"]))][metric]
    return v.std() if len(v) > 1 else np.nan


# ------------------------------------------------------------------ dataset table
def table_datasets():
    lines = []
    for ds in DSS:
        man = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/manifest.jsonl")]
        H = load_hyp(ds, MAIN_ENGINE[ds])
        ci = CASE_INSENSITIVE.get(ds, False)
        err = {}
        for sp in ("train", "test"):
            rr = [r for r in man if r["split"] == sp]
            err[sp] = np.mean([norm(H[r["id"]]["hyp"], ci) != norm(r["gt"], ci) for r in rr])
        n_tr = sum(r["split"] == "train" for r in man)
        n_te = len(man) - n_tr
        L = np.mean([len(r["gt"]) for r in man])
        dom = {"FUNSD": "scanned forms", "CORD": "camera receipts", "SROIE": "scanned receipts",
               "IAM": "handwriting", "IIIT5K": "scene text"}[ds]
        unit = "line" if ds == "SROIE" else "word"
        eng = {"tess": "Tesseract 5", "trocr": "TrOCR-small"}[MAIN_ENGINE[ds]]
        lines.append(f"{ds} & {dom} & {unit} & {n_tr:,} & {n_te:,} & {L:.1f} & {eng} & {100*err['train']:.1f} & {100*err['test']:.1f} \\\\")
    open(f"{P}/tables/datasets.tex", "w").write("\n".join(lines) + "\n")


def table_truth_in_graph():
    M = rd("locus_main")
    lines = []
    for ds in DSS:
        F = f"{ROOT}/data/{ds}/locus2_{MAIN_ENGINE[ds]}.pt"
        if not os.path.exists(F):
            continue
        Fs = [f for f in torch.load(F, weights_only=False) if f["split"] == "test"]
        ci = CASE_INSENSITIVE.get(ds, False)
        err = [f for f in Fs if norm(f["claim"], ci) != norm(f["gt"], ci)]
        tig = np.mean([any(norm(h, ci) == norm(f["gt"], ci) for h in f["hyps"]) for f in err])
        rg1 = np.mean([norm(f["g"], ci) == norm(f["gt"], ci) for f in err])
        rg2 = np.mean([norm(f["g2"], ci) == norm(f["gt"], ci) for f in err])
        nn = np.mean([len(f["hyps"]) for f in Fs])
        lines.append(f"{ds} & {nn:.1f} & {100*rg1:.1f} & {100*rg2:.1f} & {100*tig:.1f} \\\\")
    open(f"{P}/tables/truth_in_graph.tex", "w").write("\n".join(lines) + "\n")


def table_pretrain_stats():
    lines = []
    for ds in DSS:
        pp = f"{ROOT}/data/{ds}/locus2_pretrain.pt"
        if not os.path.exists(pp):
            continue
        Pt = torch.load(pp, weights_only=False)
        man = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/manifest.jsonl"))}
        ci = CASE_INSENSITIVE.get(ds, False)
        pos = [f for f in Pt if f["pseudo_label"] == 1]
        truth = {f["id"]: f["claim"] for f in pos}
        prec = np.mean([norm(f["claim"], ci) == norm(man[f["id"]]["gt"], ci) for f in pos])
        neg = [f for f in Pt if f["pseudo_label"] == 0]
        tig = np.mean([truth.get(f["id"]) in f["hyps"] for f in neg]) if neg else np.nan
        lines.append(f"{ds} & {len(pos):,} & {len(Pt):,} & {100*prec:.1f}\\% & {100*tig:.1f}\\% \\\\")
    open(f"{P}/tables/pretrain_stats.tex", "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ main comparison
BASELINES = [  # (display name, method key, k used)
    ("Engine MSP confidence~\\cite{hendrycks2017baseline}", "Engine MSP confidence", "K"),
    ("ConfidNet-style learned conf.~\\cite{corbiere2019confidnet}", "ConfidNet-style learned conf.", "K"),
    ("Two-engine voting", "Two-engine voting", "0"),
    ("Re-OCR: TrOCR-base~\\cite{li2023trocr}", "ReOCR TrOCR-base agree", "0"),
    ("Re-OCR: Florence-2~\\cite{xiao2024florence}", "ReOCR Florence-2 agree", "0"),
    ("Re-OCR: PARSeq~\\cite{bautista2022parseq}", "ReOCR PARSeq agree", "0"),
    ("Re-OCR: PARSeq, graded", "ReOCR PARSeq graded", "0"),
    ("VLM judge: Qwen2.5-VL-7B~\\cite{bai2025qwen25vl}", "VLM judge Qwen2.5-VL-7B", "0"),
    ("VLM judge, oracle threshold$^\\dagger$", "VLM judge Qwen2.5-VL-7B (oracle thr)", "0"),
]
OURS_ABL = [
    ("LR on flattened graph evidence", "Flat-LR", "K"),
    ("LR on character evidence", "CharEv-LR", "K"),
    ("GBDT on all evidence", "GBDT (all evidence)", "K"),
    ("DeepSets over rivals (no edges)", "DeepSets-rivals", "K"),
    ("LOCUS-GNN from scratch", "LOCUS-GNN (scratch)", "K"),
]
OURS_KEY = "LOCUS-GNN (pretrained, fine-tuned)"  # the final LOCUS model
OURS = [("LOCUS zero-shot (no labels)", "LOCUS-ZeroShot", "0"),
        ("LOCUS ($k$-shot)", OURS_KEY, "K")]


def table_main(k="25", metric="macro_f1", name="main_f1"):
    S = summarise()
    if not len(S):
        return
    groups = [("Baselines", BASELINES), ("Strong learned baselines on our evidence", OURS_ABL), ("Ours", OURS)]
    vals = {}
    for _, items in groups:
        for disp, key, kk in items:
            kq = k if kk == "K" else "0"
            for ds in DSS:
                r = S[(S.method == key) & (S.ds == ds) & (S.k == kq)]
                vals[(key, ds)] = r[metric].values[0] if len(r) else np.nan
    lower = metric == "aurc"
    rows = [(gname, disp, key) for gname, it in groups for disp, key, _ in it]
    elig = [("oracle" not in key) for _, _, key in rows]
    avgs = {}
    for _, _, key in rows:
        v = [vals[(key, ds)] for ds in DSS]
        avgs[key] = np.nanmean(v) if all(pd.notna(x) for x in v) else np.nan
    rk = {}
    for ds in DSS + ["avg"]:
        col = [avgs[key] if ds == "avg" else vals[(key, ds)] for _, _, key in rows]
        for (_, _, key), r in zip(rows, ranks(col, lower, elig)):
            rk[(key, ds)] = r
    lines = []
    # hindsight oracle: the best published-baseline row chosen separately for every dataset (not ranked)
    base_keys = [key for _, key, _ in BASELINES if "oracle" not in key]
    orc = []
    for ds in DSS:
        v = [vals[(key, ds)] for key in base_keys if pd.notna(vals[(key, ds)])]
        orc.append((min if lower else max)(v) if v else np.nan)
    for gname, items in groups:
        if gname == "Strong learned baselines on our evidence":
            avg = np.nanmean(orc) if all(pd.notna(x) for x in orc) else np.nan
            lines.append("\\emph{Best baseline per dataset, chosen in hindsight}$^\\ddagger$ & "
                         + " & ".join(fmt(x) for x in orc) + f" & {fmt(avg)} \\\\")
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{gname}}}}}\\\\")
        for disp, key, kk in items:
            cells = [fmt(vals[(key, ds)], rank=rk[(key, ds)]) for ds in DSS]
            lines.append(f"{disp} & {' & '.join(cells)} & {fmt(avgs[key], rank=rk[(key, 'avg')])} \\\\")
        lines.append("\\midrule")
    open(f"{P}/tables/{name}.tex", "w").write("\n".join(lines[:-1]) + "\n")


def table_all_metrics(k="25"):
    """Appendix: every metric for every method at k, one table per dataset."""
    S = summarise()
    metrics_ = [("macro_f1", "Macro-F1", False), ("f1_incorrect", "F1$_{err}$", False), ("acc", "Acc.", False),
                ("auroc", "AUROC", False), ("auprc_err", "AUPRC$_{err}$", False), ("aurc", "AURC$\\downarrow$", True),
                ("cov@95", "Cov@95", False)]
    items = BASELINES + OURS_ABL + OURS + [("LOCUS + CV-selected head", "LOCUS", "K"),
                                           ("LOCUS linear probe", "LOCUS-probe (un-anchored)", "K"),
                                           ("Fusion of label-free priors (LR)", "Priors-only LR", "K"),
                                           ("LOCUS rule (zero-shot, $\\rho$ margin)", "LOCUS-rule (zero-shot)", "0")]
    out = []
    for ds in DSS:
        out.append(f"\\begin{{table}}[p]\\centering\\caption{{All metrics on {ds} at $k={k}$ (ensemble of five seeds for learned methods; mean over seeds for threshold baselines). $^\\dagger$oracle threshold tuned on test.}}\\label{{tab:all_{ds}_k{k}}}")
        out.append("\\scriptsize\\begin{tabular}{l" + "c" * len(metrics_) + "}\\toprule")
        out.append("Method & " + " & ".join(m[1] for m in metrics_) + "\\\\\\midrule")
        present = []
        for disp, key, kk in items:
            kq = k if kk == "K" else "0"
            r = S[(S.method == key) & (S.ds == ds) & (S.k == kq)]
            if len(r):
                present.append((disp, key, r))
        elig = ["oracle" not in key for _, key, _ in present]
        R = {m: ranks([r[m].values[0] for _, _, r in present], low, elig) for m, _, low in metrics_}
        for i, (disp, key, r) in enumerate(present):
            out.append(disp + " & " + " & ".join(fmt(r[m].values[0], rank=R[m][i]) for m, _, _ in metrics_) + "\\\\")
        out.append("\\bottomrule\\end{tabular}\\end{table}\n")
    open(f"{P}/tables/all_metrics_k{k}.tex", "w").write("\n".join(out))


def table_kgrid(metric="macro_f1"):
    """Appendix: learning curves as tables, per dataset (mean +- std over seeds; ens)."""
    S = summarise()
    ks = ["5", "10", "25", "50", "100", "full"]
    items = [("Engine MSP conf.", "Engine MSP confidence"), ("ConfidNet-style", "ConfidNet-style learned conf."),
             ("Flat-LR", "Flat-LR"), ("CharEv-LR", "CharEv-LR"), ("GBDT", "GBDT (all evidence)"),
             ("DeepSets", "DeepSets-rivals"), ("LOCUS-GNN scratch", "LOCUS-GNN (scratch)"),
             ("LOCUS probe", "LOCUS-probe (un-anchored)"), ("LOCUS + CV head", "LOCUS"),
             ("Priors-only LR", "Priors-only LR"), ("\\textbf{LOCUS}", "LOCUS-GNN (pretrained, fine-tuned)")]
    out = []
    for ds in DSS:
        out.append(f"\\begin{{table}}[p]\\centering\\caption{{{ds}: {metric.replace('_', '-')} as a function of $k$ (labelled words per class). Ensemble of five seeds; subscript: std.\\ of single-seed runs.}}\\label{{tab:k_{metric}_{ds}}}")
        out.append("\\scriptsize\\begin{tabular}{l" + "c" * len(ks) + "}\\toprule")
        out.append("Method & " + " & ".join(f"$k$={k}" if k != "full" else "full" for k in ks) + "\\\\\\midrule")
        V = {}
        for disp, key in items:
            for k in ks:
                r = S[(S.method == key) & (S.ds == ds) & (S.k == k)]
                V[(key, k)] = r[metric].values[0] if len(r) else np.nan
        RK = {}
        for k in ks:
            for (disp, key), r in zip(items, ranks([V[(key, k)] for _, key in items], metric == "aurc")):
                RK[(key, k)] = r
        for disp, key in items:
            cells = []
            for k in ks:
                sd = seed_std(key, ds, k, metric)
                cells.append(fmt(V[(key, k)], rank=RK[(key, k)]) + (f"$_{{{sd:.3f}}}$".replace("0.", ".") if pd.notna(sd) else "") if pd.notna(V[(key, k)]) else "--")
            out.append(disp + " & " + " & ".join(cells) + "\\\\")
        out.append("\\bottomrule\\end{tabular}\\end{table}\n")
    open(f"{P}/tables/kgrid_{metric}.tex", "w").write("\n".join(out))


# ------------------------------------------------------------------ ablations / transfer
def table_ablation(metric="macro_f1"):
    A = rd("locus_ablate")
    if not len(A):
        return
    A["seed"] = A["seed"].astype(str)
    E = A[A.seed == "ens"]
    order = list(dict.fromkeys(E.variant))
    lines = []
    full = {ds: E[(E.variant == "LOCUS (full)") & (E.ds == ds)][metric].values for ds in DSS}
    _v = lambda v, ds: (lambda r: r[0] if len(r) else np.nan)(E[(E.variant == v) & (E.ds == ds)][metric].values)
    RK = {ds: dict(zip(order, ranks([_v(v, ds) for v in order]))) for ds in DSS}
    for v in order:
        cells, deltas = [], []
        for ds in DSS:
            r = E[(E.variant == v) & (E.ds == ds)][metric].values
            if len(r):
                cells.append(fmt(r[0], rank=RK[ds][v]))
                if len(full[ds]):
                    deltas.append(r[0] - full[ds][0])
            else:
                cells.append("--")
        d = np.mean(deltas) if deltas else np.nan
        ds_ = "" if v == "LOCUS (full)" else (f"{d:+.3f}" if pd.notna(d) else "--")
        lines.append(f"{v} & {' & '.join(cells)} & {ds_} \\\\")
        if v == "LOCUS (full)":
            lines.append("\\midrule")
    open(f"{P}/tables/ablation_{metric}.tex", "w").write("\n".join(lines) + "\n")


def table_xengine():
    X = rd("locus_xengine")
    if not len(X):
        return
    X["k"] = X["k"].astype(str)
    lines = []
    for ds in DSS:
        for tn in ("trocr", "tess"):
            r_in = X[(X.ds == ds) & (X.test == tn) & (X.setting == "in-engine")]
            r_tr = X[(X.ds == ds) & (X.test == tn) & (X.setting == "transfer")]
            if not len(r_tr):
                continue
            sn = "tess" if tn == "trocr" else "trocr"
            E = {"tess": "Tesseract", "trocr": "TrOCR-s"}
            cells = []
            for k in ("25", "full"):
                a = r_in[r_in.k == k]; b = r_tr[r_tr.k == k]
                cells += [fmt(a["macro_f1"].values[0]) if len(a) else "--", fmt(b["macro_f1"].values[0]) if len(b) else "--",
                          fmt(b["auroc"].values[0]) if len(b) else "--"]
            lines.append(f"{ds} & {E[sn]}$\\rightarrow${E[tn]} & " + " & ".join(cells) + "\\\\")
    open(f"{P}/tables/xengine.tex", "w").write("\n".join(lines) + "\n")


def table_xdataset(metric="macro_f1"):
    X = rd("locus_xdataset")
    if not len(X):
        return
    S = summarise()
    lines = []
    for src in DSS:
        cells = []
        for tgt in DSS:
            if src == tgt:
                r = S[(S.method == "LOCUS") & (S.ds == tgt) & (S.k == "full")]
                cells.append(r"\cellcolor{gray!15}" + (fmt(r[metric].values[0]) if len(r) else "--"))
            else:
                r = X[(X.train == src) & (X.test == tgt)]
                cells.append(fmt(r[metric].values[0]) if len(r) else "--")
        lines.append(f"{src} & " + " & ".join(cells) + "\\\\")
    open(f"{P}/tables/xdataset_{metric}.tex", "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ figures
CURVES = [("LOCUS-GNN (pretrained, fine-tuned)", "LOCUS", C_OURS, "o"), ("ReOCR PARSeq graded", "PARSeq re-OCR (graded)", C_B1, "s"),
          ("ReOCR TrOCR-base agree", "TrOCR-base re-OCR", "#eda100", "x"),
          ("VLM judge Qwen2.5-VL-7B", "Qwen2.5-VL-7B judge", C_B2, "^"),
          ("Engine MSP confidence", "Engine confidence", C_B3, "D"),
          ("CharEv-LR", "LR on char. evidence", C_MUTED, "v")]


def fig_learning_curves(metric="macro_f1", ylabel="Macro-F1"):
    S = summarise()
    if not len(S):
        return
    ks = ["5", "10", "25", "50", "100", "full"]
    xs = np.arange(len(ks))
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.4), sharey=False)
    for ax, ds in zip(axes, DSS):
        for key, lab, col, mk in CURVES:
            r = S[(S.method == key) & (S.ds == ds)]
            if not len(r):
                continue
            if (r.k == "0").all():  # label-free: flat reference line
                ax.axhline(r[metric].values[0], color=col, ls="--", lw=1.2, label=lab)
                continue
            y = [r[r.k == k][metric].values[0] if len(r[r.k == k]) else np.nan for k in ks]
            ax.plot(xs, y, color=col, marker=mk, ms=4, label=lab, lw=2 if key.startswith("LOCUS") else 1.4,
                    zorder=3 if key.startswith("LOCUS") else 2)
        ax.set_xticks(xs, ks)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("labelled words per class $k$")
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{P}/figs/learning_curves_{metric}.pdf", bbox_inches="tight")
    plt.close(fig)


def _test_y(ds):
    st = np.load(f"{ROOT}/results/preds_{ds}.npy", allow_pickle=True).item()
    return st


def _baseline_scores(ds, ids):
    E = MAIN_ENGINE[ds]
    ci = CASE_INSENSITIVE.get(ds, False)
    H = load_hyp(ds, E)
    out = {"Engine confidence": np.array([H[i]["conf"] for i in ids])}
    R = json.load(open(f"{ROOT}/results/reocr_{ds}.json"))
    a = np.array([float(norm(H[i]["hyp"], ci) == norm(R[i]["parseq"], ci)) for i in ids])
    c = np.array([R[i]["parseq_conf"] for i in ids])
    out["PARSeq re-OCR (graded)"] = np.where(a > 0, c, -c)
    V = json.load(open(f"{ROOT}/results/vlm_{ds}.json"))
    out["Qwen2.5-VL-7B judge"] = np.array([V[i] for i in ids])
    F = {f["id"]: f for f in torch.load(f"{ROOT}/data/{ds}/locus2_{E}.pt", weights_only=False)}
    out["TrOCR-base re-OCR"] = np.array([float(norm(H[i]["hyp"], ci) == norm(F[i]["g"], ci)) for i in ids])
    return out


def fig_risk_coverage(k="25"):
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.4))
    for ax, ds in zip(axes, DSS):
        p = f"{ROOT}/results/preds_{ds}.npy"
        if not os.path.exists(p):
            continue
        st = _test_y(ds)
        y = st["y"]
        scores = {"LOCUS": st[f"{OURS_KEY}|{k}"].mean(0)}
        scores.update(_baseline_scores(ds, st["ids"]))
        cols = {"LOCUS": C_OURS, "PARSeq re-OCR (graded)": C_B1, "TrOCR-base re-OCR": "#eda100",
                "Qwen2.5-VL-7B judge": C_B2, "Engine confidence": C_B3}
        for n, s in scores.items():
            o = np.argsort(-s)
            risk = np.cumsum(1 - y[o]) / np.arange(1, len(y) + 1)
            cov = np.arange(1, len(y) + 1) / len(y)
            ax.plot(cov, risk, color=cols[n], lw=2 if n == "LOCUS" else 1.3, label=n)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("coverage (auto-accepted)")
        ax.set_ylim(0, None)
    axes[0].set_ylabel("risk (error rate of accepted)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{P}/figs/risk_coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_rho_separation():
    """Motivation: raw likelihood vs render-normalised ratio of the claim, by class."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 3.6))
    for j, ds in enumerate(DSS):
        F = f"{ROOT}/data/{ds}/locus2_{MAIN_ENGINE[ds]}.pt"
        if not os.path.exists(F):
            continue
        Fs = [f for f in torch.load(F, weights_only=False) if f["split"] == "test"]
        ci = CASE_INSENSITIVE.get(ds, False)
        y = np.array([norm(f["claim"], ci) == norm(f["gt"], ci) for f in Fs])
        for i, (col, lab) in enumerate([(0, "$s_{V_1}(h\\mid x)$ (raw likelihood)"), (2, "$\\rho_{V_1}(h)$ (render-normalised)")]):
            v = np.array([f["x"][0, col] for f in Fs])
            v = np.clip(v, -30, 2)
            ax = axes[i, j]
            bins = np.linspace(-30, 2, 50)
            ax.hist(v[y], bins, color=C_OURS, alpha=0.6, label="Correct", density=True)
            ax.hist(v[~y], bins, color=C_B1, alpha=0.6, label="Incorrect", density=True)
            auc = metrics(v, y.astype(int))["auroc"]
            ax.text(0.03, 0.9, f"AUROC {auc:.3f}", transform=ax.transAxes, fontsize=7, color="#0b0b0b")
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
            if i == 0:
                ax.set_title(ds, fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(f"{P}/figs/rho_separation.pdf", bbox_inches="tight")
    plt.close(fig)


def table_significance(k="25", B_=2000):
    """Paired bootstrap of LOCUS vs the strongest baselines. FUNSD/CORD: cluster bootstrap over
    test DOCUMENTS (words of a page are not independent); IIIT5K: one word per image (word-level is
    exact); SROIE/IAM: the public releases carry no document ids, so word-level (may be optimistic).
    Reports Delta AUROC and Delta macro-F1 (each method at its natural operating point), two-sided p."""
    from sklearn.metrics import f1_score, roc_auc_score
    lines = []
    for ds in DSS:
        p = f"{ROOT}/results/preds_{ds}.npy"
        if not os.path.exists(p):
            continue
        st = _test_y(ds)
        y, ids = st["y"], st["ids"]
        ours = st[f"{OURS_KEY}|{k}"].mean(0)
        Bs = _baseline_scores(ds, ids)
        thr = {"PARSeq re-OCR (graded)": 0.0, "TrOCR-base re-OCR": 0.5, "Qwen2.5-VL-7B judge": 0.5}
        dp = f"{ROOT}/data/{ds}/doc_ids.json"
        if os.path.exists(dp):
            D = json.load(open(dp)); groups = np.array([D[i] for i in ids]); level = "document"
        else:
            groups = np.arange(len(y)); level = "word"
        ug = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in ug}
        rng = np.random.RandomState(0)
        for bn in ("PARSeq re-OCR (graded)", "TrOCR-base re-OCR", "Qwen2.5-VL-7B judge"):
            b = Bs[bn]
            dA, dF = [], []
            for _ in range(B_):
                ix = np.concatenate([members[g] for g in rng.choice(ug, len(ug), replace=True)])
                if len(np.unique(y[ix])) < 2:
                    continue
                dA.append(roc_auc_score(y[ix], ours[ix]) - roc_auc_score(y[ix], b[ix]))
                dF.append(f1_score(y[ix], ours[ix] >= 0.5, average="macro") - f1_score(y[ix], b[ix] >= thr[bn], average="macro"))
            cells = []
            for d in (np.array(dA), np.array(dF)):
                lo, hi = np.percentile(d, [2.5, 97.5])
                pv = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                cells += [f"{d.mean():+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", "$<$0.001" if pv < 1e-3 else f"{pv:.3f}"]
            nunits = f"{len(ug)} docs" if level == "document" else f"{len(ug)} words"
            lines.append(f"{ds} & {nunits} & {bn} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    open(f"{P}/tables/significance.tex", "w").write("\n".join(lines[:-1]) + "\n")



def table_ablation_single():
    A = rd("locus_ablate_single")
    if not len(A):
        open(f"{P}/tables/ablation_single.tex", "w").write("")
        return
    A["seed"] = A["seed"].astype(str)
    E = A[A.seed == "ens"]
    lines = []
    order = list(dict.fromkeys(E.variant))
    _v = lambda v, ds: (lambda r: r[0] if len(r) else np.nan)(E[(E.variant == v) & (E.ds == ds)]["macro_f1"].values)
    RK = {ds: dict(zip(order, ranks([_v(v, ds) for v in order]))) for ds in DSS}
    for v in order:
        cells = [fmt(_v(v, ds), rank=RK[ds][v]) if pd.notna(_v(v, ds)) else "--" for ds in DSS]
        lines.append(f"{v} & " + " & ".join(cells) + "\\\\")
    open(f"{P}/tables/ablation_single.tex", "w").write("\n".join(lines) + "\n")


def table_sensitivity():
    S = rd("locus_sensitivity")
    if not len(S):
        open(f"{P}/tables/sensitivity.tex", "w").write("")
        return
    lines = []
    for fac in dict.fromkeys(S.factor):
        vs = list(dict.fromkeys(S[S.factor == fac].value))
        _v = lambda v, ds, m: (lambda r: r[0] if len(r) else np.nan)(S[(S.factor == fac) & (S.value == v) & (S.ds == ds)][m].values)
        RK = {(ds, m): dict(zip(vs, ranks([_v(v, ds, m) for v in vs]))) for ds in ("FUNSD", "CORD", "IAM") for m in ("macro_f1", "auroc")}
        for v in vs:
            cells = []
            for ds in ("FUNSD", "CORD", "IAM"):
                r = S[(S.factor == fac) & (S.value == v) & (S.ds == ds)]
                cells += [fmt(r["macro_f1"].values[0], rank=RK[(ds, "macro_f1")][v]), fmt(r["auroc"].values[0], rank=RK[(ds, "auroc")][v])] if len(r) else ["--", "--"]
            vl = {"random": "random", "kmeans_labelfree": "label-free $k$-means"}.get(str(v), str(v))
            lines.append(f"{fac} & {vl} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    open(f"{P}/tables/sensitivity.tex", "w").write("\n".join(lines[:-1]) + "\n")


def table_mp():
    M = rd("locus_mp_analysis")
    if not len(M):
        open(f"{P}/tables/mp_analysis.tex", "w").write(""); return
    lines = []
    order = list(dict.fromkeys(M.variant))
    _v = lambda v, ds, m: (lambda r: r[0] if len(r) else np.nan)(M[(M.variant == v) & (M.ds == ds)][m].values)
    RK = {(ds, m): dict(zip(order, ranks([_v(v, ds, m) for v in order]))) for ds in DSS for m in ("macro_f1", "auroc")}
    for v in order:
        cells = []
        for ds in DSS:
            r = M[(M.variant == v) & (M.ds == ds)]
            cells.append(f"{fmt(r['macro_f1'].values[0], rank=RK[(ds, 'macro_f1')][v], color=False)} / {fmt(r['auroc'].values[0], rank=RK[(ds, 'auroc')][v], color=False)}" if len(r) else "--")
        lines.append(f"{v} & " + " & ".join(cells) + "\\\\")
    open(f"{P}/tables/mp_analysis.tex", "w").write("\n".join(lines) + "\n")


def table_fonts():
    Fz = rd("font_sensitivity")
    if not len(Fz):
        open(f"{P}/tables/font_sensitivity.tex", "w").write(""); return
    lines = []
    order = list(dict.fromkeys(Fz.variant))
    MS = ("auroc_rho_claim", "auroc_rule", "locus_macro_f1")
    _v = lambda v, ds, m: (lambda r: r[0] if len(r) else np.nan)(Fz[(Fz.variant == v) & (Fz.ds == ds)][m].values)
    RK = {(ds, m): dict(zip(order, ranks([_v(v, ds, m) for v in order]))) for ds in ("FUNSD", "CORD", "IAM") for m in MS}
    for v in order:
        cells = []
        for ds in ("FUNSD", "CORD", "IAM"):
            r = Fz[(Fz.variant == v) & (Fz.ds == ds)]
            cells += [fmt(r[m].values[0], rank=RK[(ds, m)][v]) for m in MS] if len(r) else ["--"] * 3
        lines.append(f"{v} & " + " & ".join(cells) + "\\\\")
    open(f"{P}/tables/font_sensitivity.tex", "w").write("\n".join(lines) + "\n")


def numbers():
    """LaTeX macros for every number quoted in the running text (paper/numbers.tex)."""
    S = summarise()
    g = lambda m, ds, k, met="macro_f1": S[(S.method == m) & (S.ds == ds) & (S.k == k)][met].values[0]
    L, PQ, VL = OURS_KEY, "ReOCR PARSeq graded", "VLM judge Qwen2.5-VL-7B"
    out = {}
    f3 = lambda v: f"{v:.3f}"
    sgn = lambda v: f"{v:+.3f}"
    try:
        out["VLMFUNSD"] = f3(g(VL, "FUNSD", "0")); out["VLMIAM"] = f3(g(VL, "IAM", "0"))
        out["GAINIAMFone"] = sgn(g(L, "IAM", "25") - g(PQ, "IAM", "0"))
        out["GAINIAMauc"] = sgn(g(L, "IAM", "25", "auroc") - g(PQ, "IAM", "0", "auroc"))
        out["LOCUSCORDfone"] = f3(g(L, "CORD", "25")); out["PQCORDfone"] = f3(g(PQ, "CORD", "0"))
        out["PQIIITfone"] = f3(g(PQ, "IIIT5K", "0")); out["LOCUSIIITfone"] = f3(g(L, "IIIT5K", "25"))
        out["LOCUSSROIEfone"] = f3(g(L, "SROIE", "25")); out["PQSROIEfone"] = f3(g(PQ, "SROIE", "0"))
        out["LOCUSSROIEauc"] = f3(g(L, "SROIE", "25", "auroc")); out["PQSROIEauc"] = f3(g(PQ, "SROIE", "0", "auroc"))
        out["VLMSROIEfone"] = f3(g(VL, "SROIE", "0"))
    except Exception as e:
        print("numbers incomplete:", e)
    open(f"{P}/numbers.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in out.items()) + "\n")


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    fns = {"datasets": table_datasets, "tig": table_truth_in_graph, "pre": table_pretrain_stats,
           "main": lambda: [table_main(k, m, f"main_{m}_k{k}") for k in ("10", "25", "100", "full") for m in ("macro_f1", "auroc", "aurc", "cov@95")],
           "allm": lambda: [table_all_metrics(k) for k in ("10", "25", "100")],
           "kgrid": lambda: [table_kgrid(m) for m in ("macro_f1", "auroc", "aurc")],
           "ablate": lambda: [table_ablation(m) for m in ("macro_f1", "auroc", "repair")],
           "xeng": table_xengine, "xds": lambda: [table_xdataset(m) for m in ("macro_f1", "auroc")],
           "curves": lambda: [fig_learning_curves(m, l) for m, l in (("macro_f1", "Macro-F1"), ("auroc", "AUROC"))],
           "rc": fig_risk_coverage, "rho": fig_rho_separation, "sig": table_significance,
           "single": table_ablation_single, "nums": numbers, "sens": table_sensitivity, "mp": table_mp, "fonts": table_fonts}
    for w in (fns if what == ["all"] else what):
        try:
            fns[w]()
            print("ok", w)
        except Exception as e:
            print("FAILED", w, repr(e))

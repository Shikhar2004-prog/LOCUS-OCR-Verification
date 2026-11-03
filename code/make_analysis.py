"""Deeper quantitative analyses from saved test predictions (LOCUS at k=25 vs baselines)."""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CASE_INSENSITIVE, MAIN_ENGINE, ROOT, load_hyp, metrics, norm
from features import levenshtein_ops
from make_assets import C_B1, C_B2, C_B3, C_OURS, DSS, P, _baseline_scores, fmt


def bundle(ds, k="25"):
    st = np.load(f"{ROOT}/results/preds_{ds}.npy", allow_pickle=True).item()
    man = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/manifest.jsonl"))}
    H = load_hyp(ds, MAIN_ENGINE[ds])
    S = {"LOCUS": st[f"LOCUS-GNN (pretrained, fine-tuned)|{k}"].mean(0)}
    S.update(_baseline_scores(ds, st["ids"]))
    return st, man, H, S


def thresholds(S, y):
    """Operating points: LOCUS at 0.5, PARSeq graded at 0, VLM at 0.5, engine conf at the
    macro-F1-optimal threshold on the test set (favourable to the baseline)."""
    from run import best_threshold
    return {"LOCUS": 0.5, "PARSeq re-OCR (graded)": 0.0, "Qwen2.5-VL-7B judge": 0.5, "TrOCR-base re-OCR": 0.5,
            "Engine confidence": best_threshold(S["Engine confidence"], y)}


def error_type(claim, gt, ci):
    a, b = norm(claim, ci), norm(gt, ci)
    if a.lower() == b.lower() and a != b:
        return "case only"
    ops, al, d = levenshtein_ops(a, b)
    if d > 1:
        return "multi-edit"
    if len(a) > len(b):
        return "insertion"
    if len(a) < len(b):
        return "deletion"
    return "substitution"


def table_error_types():
    """Recall of truly incorrect claims (detection rate) per error type."""
    lines = []
    types = ["substitution", "insertion", "deletion", "case only", "multi-edit"]
    for ds in DSS:
        if not os.path.exists(f"{ROOT}/results/preds_{ds}.npy"):
            continue
        st, man, H, S = bundle(ds)
        y = st["y"]; ci = CASE_INSENSITIVE.get(ds, False)
        T = thresholds(S, y)
        et = np.array([error_type(H[i]["hyp"], man[i]["gt"], ci) if yy == 0 else "" for i, yy in zip(st["ids"], y)])
        for n in ("LOCUS", "PARSeq re-OCR (graded)", "TrOCR-base re-OCR", "Qwen2.5-VL-7B judge", "Engine confidence"):
            rej = S[n] < T[n]
            cells = []
            for t in types:
                m = et == t
                cells.append(f"{100*rej[m].mean():.1f}\\,({m.sum()})" if m.sum() >= 5 else "--")
            lines.append(f"{ds if n == 'LOCUS' else ''} & {n} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    open(f"{P}/tables/error_types.tex", "w").write("\n".join(lines[:-1]) + "\n")


def table_review_cost():
    """Human-review workload: fraction of words that must be reviewed so that the
    auto-accepted words reach a target precision (lower is better)."""
    blocks = []
    names = ("LOCUS", "PARSeq re-OCR (graded)", "TrOCR-base re-OCR", "Qwen2.5-VL-7B judge", "Engine confidence")
    for ds in DSS:
        if not os.path.exists(f"{ROOT}/results/preds_{ds}.npy"):
            continue
        st, man, H, S = bundle(ds)
        y = st["y"]
        V = {}
        for n in names:
            for tgt in (0.90, 0.95, 0.98):
                o = np.argsort(-S[n])
                prec = np.cumsum(y[o]) / np.arange(1, len(y) + 1)
                ok = np.where(prec >= tgt)[0]
                cov = (ok.max() + 1) / len(y) if len(ok) else 0
                V[(n, tgt)] = float(f"{100*(1-cov):.1f}")  # share of words sent to review (same rounding as before)
        lines = []
        for n in names:
            cells = []
            for tgt in (0.90, 0.95, 0.98):
                u = sorted({V[(m, tgt)] for m in names})  # lower is better
                v = V[(n, tgt)]
                c = f"{v:.1f}"
                cells.append(f"\\best{{{c}}}" if v == u[0] else (f"\\second{{{c}}}" if len(u) > 1 and v == u[1] else c))
            lines.append(f"{ds if n == 'LOCUS' else ''} & {n} & " + " & ".join(cells) + "\\\\")
        blocks.append("\n".join(lines))
    open(f"{P}/tables/review_cost.tex", "w").write("\n\\midrule\n".join(blocks) + "\n")
    # two halves for side-by-side layouts
    open(f"{P}/tables/review_cost_a.tex", "w").write("\n\\midrule\n".join(blocks[:3]) + "\n")
    open(f"{P}/tables/review_cost_b.tex", "w").write("\n\\midrule\n".join(blocks[3:]) + "\n")


def table_repair():
    """Repair of truly incorrect claims: LOCUS's label-free proposal vs replacing the claim by
    PARSeq's / TrOCR's reading (top-1 exact match with the ground truth)."""
    lines = []
    for ds in DSS:
        p = f"{ROOT}/results/repair_{ds}_k0.npy"
        if not os.path.exists(p):
            continue
        st = np.load(f"{ROOT}/results/preds_{ds}.npy", allow_pickle=True).item()
        F = {f["id"]: f for f in torch.load(f"{ROOT}/data/{ds}/locus2_{MAIN_ENGINE[ds]}.pt", weights_only=False)}
        man = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/manifest.jsonl"))}
        R = json.load(open(f"{ROOT}/results/reocr_{ds}.json"))
        rep = np.load(p)
        ci = CASE_INSENSITIVE.get(ds, False)
        err = [i for i, yy in enumerate(st["y"]) if yy == 0]
        eq = lambda a, b: norm(a, ci) == norm(b, ci)
        tig = np.mean([any(eq(h, man[st["ids"][i]]["gt"]) for h in F[st["ids"][i]]["hyps"]) for i in err])
        ours = np.mean([rep[i] > 0 and eq(F[st["ids"][i]]["hyps"][rep[i]], man[st["ids"][i]]["gt"]) for i in err])
        pq = np.mean([eq(R[st["ids"][i]]["parseq"], man[st["ids"][i]]["gt"]) for i in err])
        tr = np.mean([eq(F[st["ids"][i]]["g"], man[st["ids"][i]]["gt"]) for i in err])
        vals = [100 * tr, 100 * pq, 100 * ours]
        u = sorted({round(v, 1) for v in vals}, reverse=True)
        cells = [f"\\best{{{v:.1f}}}" if round(v, 1) == u[0] else (f"\\second{{{v:.1f}}}" if len(u) > 1 and round(v, 1) == u[1] else f"{v:.1f}") for v in vals]
        lines.append(f"{ds} & {len(err)} & {100*tig:.1f} & " + " & ".join(cells) + " \\\\")
    open(f"{P}/tables/repair.tex", "w").write("\n".join(lines) + "\n")


def ece(p, y, bins=15):
    e = 0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def table_calibration():
    lines = []
    for ds in DSS:
        if not os.path.exists(f"{ROOT}/results/preds_{ds}.npy"):
            continue
        st, man, H, S = bundle(ds)
        y = st["y"]
        conf = np.clip(S["Engine confidence"], 0, 1)
        pq = (S["PARSeq re-OCR (graded)"] + 1) / 2  # map [-1,1] -> [0,1] for an ECE comparison
        lines.append(f"{ds} & {ece(S['LOCUS'], y):.3f} & {ece(pq, y):.3f} & {ece(S['Qwen2.5-VL-7B judge'], y):.3f} & {ece(conf, y):.3f} \\\\")
    open(f"{P}/tables/calibration.tex", "w").write("\n".join(lines) + "\n")


def fig_length():
    """AUROC by claim length bin."""
    bins = [(1, 2), (3, 4), (5, 7), (8, 12), (13, 99)]
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.2))
    cols = {"LOCUS": C_OURS, "PARSeq re-OCR (graded)": C_B1, "TrOCR-base re-OCR": "#eda100",
            "Qwen2.5-VL-7B judge": C_B2, "Engine confidence": C_B3}
    for ax, ds in zip(axes, DSS):
        if not os.path.exists(f"{ROOT}/results/preds_{ds}.npy"):
            continue
        st, man, H, S = bundle(ds)
        y = st["y"]
        L = np.array([len(H[i]["hyp"]) for i in st["ids"]])
        for n, c in cols.items():
            v = []
            for a, b in bins:
                m = (L >= a) & (L <= b)
                v.append(metrics(S[n][m], y[m])["auroc"] if m.sum() > 30 and 0 < y[m].mean() < 1 else np.nan)
            ax.plot(range(len(bins)), v, marker="o", ms=3, color=c, label=n, lw=2 if n == "LOCUS" else 1.2)
        ax.set_xticks(range(len(bins)), [f"{a}-{b}" if b < 99 else f"{a}+" for a, b in bins])
        ax.set_title(ds, fontsize=9); ax.set_xlabel("claim length")
    axes[0].set_ylabel("AUROC")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{P}/figs/by_length.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_error_type_dist():
    """Composition of real engine errors on each test set (horizontal stacked bars, one per dataset)."""
    types = ["substitution", "insertion", "deletion", "case only", "multi-edit"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]  # validated categorical slots 1-5
    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    for j, ds in enumerate(DSS):
        man = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/manifest.jsonl")]
        H = load_hyp(ds, MAIN_ENGINE[ds])
        ci = CASE_INSENSITIVE.get(ds, False)
        et = [error_type(H[r["id"]]["hyp"], r["gt"], ci) for r in man
              if r["split"] == "test" and norm(H[r["id"]]["hyp"], ci) != norm(r["gt"], ci)]
        frac = [et.count(t) / len(et) for t in types]
        left = 0
        for t, f, c in zip(types, frac, cols):
            ax.barh(j, f, left=left, color=c, edgecolor="white", linewidth=1.5, label=t if j == 0 else None)
            if f > 0.07:
                ax.text(left + f / 2, j, f"{100*f:.0f}", ha="center", va="center", fontsize=6.5, color="white")
            left += f
    ax.set_yticks(range(len(DSS)), DSS); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("share of real engine errors on the test set")
    ax.legend(ncol=5, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(f"{P}/figs/error_type_dist.pdf", bbox_inches="tight")
    plt.close(fig)


def recogniser_dependence(k="25"):
    """How much does LOCUS depend on its verifiers? (a) verifier word accuracy vs LOCUS's gain over
    the best re-recognition baseline; (b) decision accuracy stratified by which verifier reads the
    word correctly; (c) figure of the gain against the best verifier's accuracy."""
    from sklearn.metrics import f1_score, roc_auc_score
    A, Bl, pts = [], [], []
    for ds in DSS:
        if not os.path.exists(f"{ROOT}/results/preds_{ds}.npy"):
            continue
        st, man, H, S = bundle(ds, k)
        y, ids = st["y"], st["ids"]
        ci = CASE_INSENSITIVE.get(ds, False)
        F = {f["id"]: f for f in torch.load(f"{ROOT}/data/{ds}/locus2_{MAIN_ENGINE[ds]}.pt", weights_only=False)}
        R = json.load(open(f"{ROOT}/results/reocr_{ds}.json"))
        t_ok = np.array([norm(F[i]["g"], ci) == norm(man[i]["gt"], ci) for i in ids])
        p_ok = np.array([norm(R[i]["parseq"], ci) == norm(man[i]["gt"], ci) for i in ids])
        T = thresholds(S, y)
        dec = {n: (S[n] >= T[n]).astype(int) for n in ("LOCUS", "PARSeq re-OCR (graded)", "TrOCR-base re-OCR")}
        auc = {n: roc_auc_score(y, S[n]) for n in dec}
        f1 = {n: f1_score(y, dec[n], average="macro") for n in dec}
        best_b = max(("PARSeq re-OCR (graded)", "TrOCR-base re-OCR"), key=lambda n: f1[n])
        A.append(f"{ds} & {100*t_ok.mean():.1f} & {100*p_ok.mean():.1f} & {best_b.split()[0]} & "
                 f"{f1[best_b]:.3f} & {f1['LOCUS']:.3f} & {f1['LOCUS']-f1[best_b]:+.3f} & "
                 f"{auc[best_b]:.3f} & {auc['LOCUS']:.3f} & {auc['LOCUS']-auc[best_b]:+.3f} \\\\")
        pts.append((ds, max(t_ok.mean(), p_ok.mean()), auc["LOCUS"] - auc[best_b], f1["LOCUS"] - f1[best_b]))
        for name, m in (("both right", t_ok & p_ok), ("only TrOCR right", t_ok & ~p_ok),
                        ("only PARSeq right", ~t_ok & p_ok), ("neither right", ~t_ok & ~p_ok)):
            if m.sum() < 20:
                Bl.append(f"{ds if name == 'both right' else ''} & {name} & {m.sum()} & -- & -- & -- \\\\"); continue
            acc = {n: (dec[n][m] == y[m]).mean() for n in dec}
            Bl.append(f"{ds if name == 'both right' else ''} & {name} & {m.sum()} & {100*(1-y[m].mean()):.0f} & "
                      + " & ".join(f"{100*acc[n]:.1f}" for n in ("LOCUS", "PARSeq re-OCR (graded)", "TrOCR-base re-OCR")) + " \\\\")
        Bl.append("\\midrule")
    open(f"{P}/tables/recog_dependence.tex", "w").write("\n".join(A) + "\n")
    open(f"{P}/tables/recog_strata.tex", "w").write("\n".join(Bl[:-1]) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for ax, j, lab in ((axes[0], 2, "AUROC gain over best re-OCR"), (axes[1], 3, "macro-F1 gain over best re-OCR")):
        for ds, acc, ga, gf in pts:
            v = ga if j == 2 else gf
            ax.scatter(100 * acc, v, s=28, color=C_OURS, zorder=3)
            ax.annotate(ds, (100 * acc, v), textcoords="offset points", xytext=(4, 3), fontsize=7)
        ax.axhline(0, color="#52514e", lw=0.7)
        ax.set_xlabel("word accuracy of the better verifier (%)"); ax.set_ylabel(lab, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{P}/figs/recog_dependence.pdf", bbox_inches="tight")
    plt.close(fig)


def char_error_subset(k="25"):
    """Robustness: re-score every method after removing incorrect claims whose only difference from the
    ground truth is case or punctuation (annotation-convention and crop-boundary artefacts)."""
    from make_qual2 import artefact
    lines = []
    names = ("LOCUS", "PARSeq re-OCR (graded)", "TrOCR-base re-OCR", "Qwen2.5-VL-7B judge", "Engine confidence")
    for ds in DSS:
        if not os.path.exists(f"{ROOT}/results/preds_{ds}.npy"):
            continue
        st, man, H, S = bundle(ds, k)
        y, ids = st["y"], st["ids"]
        T = thresholds(S, y)
        keep = np.array([yy == 1 or artefact(H[i]["hyp"], man[i]["gt"]) is None for i, yy in zip(ids, y)])
        cells = []
        for n in names:
            full = metrics(S[n], y, thr=T[n]); sub = metrics(S[n][keep], y[keep], thr=T[n])
            cells.append(f"{full['auroc']:.3f} / {sub['auroc']:.3f}")
        lines.append(f"{ds} & {100*(1-keep.mean())/max(1e-9,(1-y.mean())):.0f}\\% & " + " & ".join(cells) + " \\\\")
    open(f"{P}/tables/char_error_subset.tex", "w").write("\n".join(lines) + "\n")


def table_runtime():
    """Measured wall-clock throughput (words/s, one RTX A6000) of each verifier."""
    import time
    R = {}
    p = f"{ROOT}/results/runtime.json"
    if os.path.exists(p):
        R = json.load(open(p))
    lines = [f"{k} & {v['params']} & " + (f"{v['wps']:.1f}" if v['wps'] == v['wps'] else "--") + " \\\\" for k, v in R.items()]
    open(f"{P}/tables/runtime.tex", "w").write("\n".join(lines) + "\n")


def _rank_marks(vals, color=True):
    """best / second best (higher is better, ties share a mark) as LaTeX wrappers."""
    u = sorted({v for v in vals if v is not None}, reverse=True)
    b, s2 = ("\\best", "\\second") if color else ("\\textbf", "\\underline")
    return [(b if v == u[0] else (s2 if len(u) > 1 and v == u[1] else "")) if v is not None else "" for v in vals]


def mark_comparison_tables():
    """Add best / second-best marks to the multi-method analysis tables."""
    # recogniser strata: last three columns are LOCUS, PARSeq agreement, TrOCR agreement
    f = f"{P}/tables/recog_strata.tex"
    out = []
    for line in open(f).read().splitlines():
        cells = [c.strip() for c in line.rstrip(" \\").split("&")]
        if len(cells) == 7:
            vals = [float(c) for c in cells[4:7]]
            mk = _rank_marks(vals)
            cells[4:7] = [f"{m}{{{c}}}" if m else c for m, c in zip(mk, cells[4:7])]
            line = " & ".join(cells) + " \\\\"
        out.append(line)
    open(f, "w").write("\n".join(out) + "\n")
    # character-error subset: columns 2.. hold "all / char-only" AUROC per method
    f = f"{P}/tables/char_error_subset.tex"
    out = []
    for line in open(f).read().splitlines():
        cells = [c.strip() for c in line.rstrip(" \\").split("&")]
        pairs = [c.split(" / ") for c in cells[2:]]
        ma = _rank_marks([float(a) for a, _ in pairs], color=False)
        mb = _rank_marks([float(b) for _, b in pairs], color=False)
        cells[2:] = [(f"{x}{{{a}}}" if x else a) + " / " + (f"{y}{{{b}}}" if y else b) for (a, b), x, y in zip(pairs, ma, mb)]
        out.append(" & ".join(cells) + " \\\\")
    open(f, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    for fn in (table_error_types, table_review_cost, table_repair, table_calibration, fig_length, table_runtime,
               fig_error_type_dist, recogniser_dependence, char_error_subset, mark_comparison_tables):
        try:
            fn(); print("ok", fn.__name__)
        except Exception as e:
            print("FAILED", fn.__name__, repr(e))

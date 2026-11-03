"""Qualitative figures from real test predictions: teaser (Fig. 1) and galleries.
Every example is selected by rule from saved predictions, never hand-edited."""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CASE_INSENSITIVE, MAIN_ENGINE, ROOT, load_hyp, norm

P = f"{ROOT}/paper/figs"
C_OURS, C_BAD, C_GOOD, C_MUTED = "#2a78d6", "#e34948", "#008300", "#8a8984"
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
DSS = ["FUNSD", "CORD", "SROIE", "IAM", "IIIT5K"]


def load(ds, k="25"):
    st = np.load(f"{ROOT}/results/preds_{ds}.npy", allow_pickle=True).item()
    F = {f["id"]: f for f in torch.load(f"{ROOT}/data/{ds}/locus2_{MAIN_ENGINE[ds]}.pt", weights_only=False)}
    man = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/manifest.jsonl"))}
    H = load_hyp(ds, MAIN_ENGINE[ds])
    R = json.load(open(f"{ROOT}/results/reocr_{ds}.json"))
    rep = np.load(f"{ROOT}/results/repair_{ds}_k0.npy")
    p = st[f"LOCUS-GNN (pretrained, fine-tuned)|{k}"].mean(0)
    rows = []
    ci = CASE_INSENSITIVE.get(ds, False)
    for i, id_ in enumerate(st["ids"]):
        f = F[id_]
        rows.append(dict(ds=ds, id=id_, y=int(st["y"][i]), p=float(p[i]), claim=f["claim"], gt=man[id_]["gt"],
                         conf=H[id_]["conf"], parseq=R[id_]["parseq"], pconf=R[id_]["parseq_conf"],
                         parseq_agree=norm(R[id_]["parseq"], ci) == norm(f["claim"], ci),
                         repair=f["hyps"][rep[i]] if rep[i] > 0 else None, path=man[id_]["path"], f=f))
    return rows


def crop(ax, path, title=None):
    im = Image.open(path).convert("RGB")
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
    if title:
        ax.set_title(title, fontsize=7, loc="left")


def rho_bars(ax, f, gt, ci, top=6):
    """Bars of the summed render-normalised ratio of the claim and its strongest rivals."""
    r = f["x"][:, 2] + f["x"][:, 13]
    order = [0] + [j for j in np.argsort(-r) if j != 0][: top - 1]
    labs = [f["hyps"][j] for j in order]
    vals = [r[j] for j in order]
    cols = [C_BAD if j == 0 and norm(f["hyps"][0], ci) != norm(gt, ci) else
            (C_GOOD if norm(f["hyps"][j], ci) == norm(gt, ci) else C_MUTED) for j in order]
    y = np.arange(len(order))[::-1]
    ax.barh(y, np.clip(vals, -40, 5), color=cols, height=0.6)
    ax.set_yticks(y, [("claim: " if j == 0 else "") + l for j, l in zip(order, labs)], fontsize=6, family="monospace")
    ax.axvline(0, color="#52514e", lw=0.6)
    ax.set_xlabel(r"$\rho_{V_1}+\rho_{V_2}$", fontsize=6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def pick(rows, cond, n, key=None):
    c = [r for r in rows if cond(r)]
    c.sort(key=key or (lambda r: r["id"]))
    return c[:n]


def gallery(name, cases, title):
    n = len(cases)
    fig = plt.figure(figsize=(7.2, 1.05 * n))
    for i, r in enumerate(cases):
        ci = CASE_INSENSITIVE.get(r["ds"], False)
        ax = fig.add_axes([0.0, 1 - (i + 1) / n + 0.02, 0.22, 1 / n - 0.04])
        crop(ax, r["path"])
        ax2 = fig.add_axes([0.24, 1 - (i + 1) / n + 0.02, 0.36, 1 / n - 0.04])
        ax2.axis("off")
        verdict = "ACCEPT" if r["p"] >= 0.5 else "REJECT"
        txt = (f"{r['ds']}   truth: \"{r['gt']}\"\n"
               f"OCR claim: \"{r['claim']}\"  (engine conf {r['conf']:.2f})\n"
               f"PARSeq: \"{r['parseq']}\" ({'agrees' if r['parseq_agree'] else 'disagrees'}, {r['pconf']:.2f})\n"
               f"LOCUS: {verdict}  P(correct)={r['p']:.2f}" + (f"\nrepair: \"{r['repair']}\"" if r["repair"] and r["p"] < 0.5 else ""))
        ax2.text(0, 0.5, txt, va="center", fontsize=6.3, family="monospace")
        ax3 = fig.add_axes([0.75, 1 - (i + 1) / n + 0.03, 0.24, 1 / n - 0.06])
        rho_bars(ax3, r["f"], r["gt"], ci, top=4)
    fig.suptitle(title, fontsize=8, y=1.01)
    fig.savefig(f"{P}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{P}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    rows = []
    for ds in DSS:
        if os.path.exists(f"{ROOT}/results/preds_{ds}.npy"):
            rows += load(ds)
    eq = lambda a, b, ds: norm(a, CASE_INSENSITIVE.get(ds, False)) == norm(b, CASE_INSENSITIVE.get(ds, False))
    per = lambda cond, n=2: [x for ds in DSS for x in pick([r for r in rows if r["ds"] == ds], cond, n,
                                                           key=lambda r: -r["conf"])]
    # (a) confident engine error, PARSeq fooled too or absent, LOCUS rejects and repairs correctly
    A = per(lambda r: r["y"] == 0 and r["p"] < 0.5 and r["repair"] and eq(r["repair"], r["gt"], r["ds"]) and r["conf"] > 0.8)
    gallery("qual_rejected_repaired", A, "Confident OCR errors rejected and repaired by LOCUS")
    # (b) correct claims that PARSeq misreads (re-OCR would reject) but LOCUS accepts
    Bc = per(lambda r: r["y"] == 1 and not r["parseq_agree"] and r["p"] >= 0.5)
    gallery("qual_parseq_wrong", Bc, "Correct claims rejected by PARSeq re-recognition but accepted by LOCUS")
    # (c) errors that both PARSeq and the engine accept but LOCUS rejects
    Cc = per(lambda r: r["y"] == 0 and r["parseq_agree"] and r["p"] < 0.5)
    gallery("qual_both_fooled", Cc, "Errors accepted by PARSeq agreement but rejected by LOCUS")
    # (d) LOCUS failures (most confident mistakes)
    D = [x for ds in DSS for x in pick([r for r in rows if r["ds"] == ds and ((r["p"] >= 0.5) != (r["y"] == 1))], lambda r: True, 2,
                                       key=lambda r: -abs(r["p"] - 0.5))]
    gallery("qual_failures", D, "Most confident LOCUS mistakes (failure analysis)")
    # counts for the text
    stats = {}
    for ds in DSS:
        R_ = [r for r in rows if r["ds"] == ds]
        if not R_:
            continue
        stats[ds] = {"parseq_misread_correct": sum(r["y"] == 1 and not r["parseq_agree"] for r in R_),
                     "of_which_locus_accepts": sum(r["y"] == 1 and not r["parseq_agree"] and r["p"] >= .5 for r in R_),
                     "parseq_fooled_errors": sum(r["y"] == 0 and r["parseq_agree"] for r in R_),
                     "of_which_locus_rejects": sum(r["y"] == 0 and r["parseq_agree"] and r["p"] < .5 for r in R_)}
    json.dump(stats, open(f"{ROOT}/results/qual_stats.json", "w"), indent=1)
    with open(f"{ROOT}/paper/tables/qual_counts.tex", "w") as fh:
        for ds, v in stats.items():
            a, b = v["parseq_misread_correct"], v["of_which_locus_accepts"]
            c, d = v["parseq_fooled_errors"], v["of_which_locus_rejects"]
            fh.write(f"{ds} & {a} & {b} ({100*b/max(1,a):.0f}\\%) & {c} & {d} ({100*d/max(1,c):.0f}\\%) \\\\\n")
    print(stats)
    teaser(rows)


def teaser(rows):
    """Fig. 1: one real FUNSD/CORD example where the engine is confident and wrong."""
    eq = lambda a, b, ds: norm(a, CASE_INSENSITIVE.get(ds, False)) == norm(b, CASE_INSENSITIVE.get(ds, False))
    # an error that BOTH single-hypothesis checks accept (confident engine, PARSeq agrees)
    # but LOCUS rejects and repairs correctly; prefer document datasets, then highest confidence
    cand = [r for r in rows if r["y"] == 0 and r["conf"] > 0.8 and r["parseq_agree"] and r["p"] < 0.3
            and r["repair"] and eq(r["repair"], r["gt"], r["ds"]) and 2 <= len(r["gt"]) <= 10]
    cand.sort(key=lambda r: (r["ds"] not in ("FUNSD", "CORD", "SROIE"), -r["conf"]))
    if not cand:
        return
    r = cand[0]
    ci = CASE_INSENSITIVE.get(r["ds"], False)
    fig = plt.figure(figsize=(7.0, 2.1))
    ax = fig.add_axes([0.0, 0.35, 0.18, 0.5]); crop(ax, r["path"], "word image $x$")
    ax.text(0, -0.35, f"truth: \"{r['gt']}\"", transform=ax.transAxes, fontsize=7)
    a2 = fig.add_axes([0.2, 0.05, 0.3, 0.9]); a2.axis("off")
    a2.text(0, 0.95, "Single-hypothesis checks", fontsize=8, weight="bold", va="top")
    a2.text(0, 0.78, f"OCR claim $h$ = \"{r['claim']}\"", fontsize=7.5, va="top", family="monospace")
    a2.text(0, 0.6, f"engine confidence: {r['conf']:.2f}  -> accept", fontsize=7, va="top", color=C_BAD)
    a2.text(0, 0.45, f"PARSeq re-OCR: \"{r['parseq']}\" -> {'accept' if r['parseq_agree'] else 'reject'}", fontsize=7,
            va="top", color=C_BAD if r["parseq_agree"] else "#0b0b0b")
    a2.text(0, 0.25, "They score $h$ in isolation: is it\nplausible? not: is it the BEST\nexplanation of $x$?", fontsize=6.5,
            va="top", color="#52514e")
    a3 = fig.add_axes([0.62, 0.18, 0.2, 0.72])
    rho_bars(a3, r["f"], r["gt"], ci, top=6)
    a3.set_title("LOCUS: compare $h$ with its counterfactual rivals", fontsize=7.5, loc="left")
    a4 = fig.add_axes([0.85, 0.1, 0.15, 0.8]); a4.axis("off")
    a4.text(0, 0.7, "REJECT", color=C_BAD, fontsize=10, weight="bold")
    a4.text(0, 0.55, f"P(correct)={r['p']:.2f}", fontsize=7)
    a4.text(0, 0.35, "repair:", fontsize=7)
    a4.text(0, 0.22, f"\"{r['repair']}\"", fontsize=9, color=C_GOOD, family="monospace")
    fig.savefig(f"{P}/teaser.pdf", bbox_inches="tight")
    fig.savefig(f"{P}/teaser.png", bbox_inches="tight", dpi=200)
    Image.open(r["path"]).convert("RGB").save(f"{P}/ex_crop.png")
    json.dump({k: v for k, v in r.items() if k != "f"}, open(f"{ROOT}/results/teaser_example.json", "w"), indent=1)
    plt.close(fig)


if __name__ == "__main__":
    main()

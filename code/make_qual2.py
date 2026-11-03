"""Qualitative analysis v2: representative, not cherry-picked.

  Q1  random stratified sample per dataset (seed 0): 3 incorrect + 3 correct claims, LOCUS right or wrong
  Q2  regime "errors PARSeq agrees with": random LOCUS successes AND failures (character errors only)
  Q3  regime "correct claims PARSeq misreads": random successes AND failures
  per-verifier evidence: delta rho_V(v) = rho_V(v) - rho_V(h) for the strongest rivals (claim = 0)
  audit sheets: random LOCUS errors per dataset for manual inspection
Every row is drawn by a seeded random choice inside a rule-defined stratum, never chosen by hand.
"""
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
OURS_KEY = "LOCUS-GNN (pretrained, fine-tuned)"
DSS = ["FUNSD", "CORD", "SROIE", "IAM", "IIIT5K"]
C_T, C_P = "#2a78d6", "#eb6834"  # validated categorical slots 1-2: TrOCR, PARSeq
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})


def artefact(claim, gt):
    """Differences that are annotation conventions rather than character errors."""
    if claim == gt:
        return None
    if claim.lower() == gt.lower():
        return "case-only"
    a = "".join(c for c in claim.lower() if c.isalnum())
    b = "".join(c for c in gt.lower() if c.isalnum())
    if a == b:
        return "punctuation-only"
    return None


def load(ds, k="25"):
    st = np.load(f"{ROOT}/results/preds_{ds}.npy", allow_pickle=True).item()
    F = {f["id"]: f for f in torch.load(f"{ROOT}/data/{ds}/locus2_{MAIN_ENGINE[ds]}.pt", weights_only=False)}
    man = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/manifest.jsonl"))}
    R = json.load(open(f"{ROOT}/results/reocr_{ds}.json"))
    rep = np.load(f"{ROOT}/results/repair_{ds}_k0.npy")
    p = st[f"{OURS_KEY}|{k}"].mean(0)
    ci = CASE_INSENSITIVE.get(ds, False)
    rows = []
    for i, id_ in enumerate(st["ids"]):
        f = F[id_]
        y = int(st["y"][i])
        rows.append(dict(ds=ds, id=id_, y=y, p=float(p[i]), claim=f["claim"], gt=man[id_]["gt"],
                         parseq=R[id_]["parseq"], pconf=R[id_]["parseq_conf"],
                         parseq_agree=norm(R[id_]["parseq"], ci) == norm(f["claim"], ci),
                         artefact=artefact(f["claim"], man[id_]["gt"]) if y == 0 else None,
                         repair=f["hyps"][rep[i]] if rep[i] > 0 else None, path=man[id_]["path"], f=f,
                         locus_right=bool((p[i] >= 0.5) == (y == 1))))
    return rows


def evidence(ax, r, top=3):
    """Per-verifier render-normalised ratio of the strongest rivals RELATIVE to the claim."""
    f, ci = r["f"], CASE_INSENSITIVE.get(r["ds"], False)
    x = f["x"]
    if len(x) < 2:
        ax.axis("off"); return
    dT = x[1:, 2] - x[0, 2]
    dP = x[1:, 13] - x[0, 13]
    order = np.argsort(-(dT + dP))[:top]
    labs = [f["hyps"][j + 1] for j in order]
    y = np.arange(len(order))[::-1]
    ax.barh(y + 0.18, np.clip(dT[order], -30, 30), height=0.34, color=C_T, label="TrOCR")
    ax.barh(y - 0.18, np.clip(dP[order], -30, 30), height=0.34, color=C_P, label="PARSeq")
    ax.set_yticks(y, [("* " if norm(l, ci) == norm(r["gt"], ci) else "") + l for l in labs],
                  fontsize=6, family="monospace")
    ax.axvline(0, color="#52514e", lw=0.7)
    ax.tick_params(axis="x", labelsize=5.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def gallery(name, cases, title, note):
    n = len(cases)
    if n == 0:
        return
    fig = plt.figure(figsize=(7.2, 0.95 * n + 0.5))
    H = 1 / (n + 0.6)
    for i, r in enumerate(cases):
        y0 = 1 - (i + 1.3) * H
        ax = fig.add_axes([0.0, y0 + 0.01, 0.2, H * 0.85])
        im = Image.open(r["path"]).convert("RGB")
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
        ax2 = fig.add_axes([0.22, y0, 0.45, H]); ax2.axis("off")
        verdict = "ACCEPT" if r["p"] >= 0.5 else "REJECT"
        ok = "correct" if r["locus_right"] else "WRONG"
        lines = [f"{r['ds']}  truth [{r['gt']}]   claim [{r['claim']}]  ({'correct' if r['y'] else 'incorrect'})",
                 f"PARSeq [{r['parseq']}] ({'agrees' if r['parseq_agree'] else 'disagrees'}, {r['pconf']:.2f})",
                 f"LOCUS {verdict} P={r['p']:.2f} -> decision {ok}"]
        if r["y"] == 0 and r["p"] < 0.5:
            good = r["repair"] is not None and norm(r["repair"], CASE_INSENSITIVE.get(r["ds"], False)) == norm(r["gt"], CASE_INSENSITIVE.get(r["ds"], False))
            lines.append(f"repair [{r['repair']}] ({'= truth' if good else 'NOT the truth'})")
        ax2.text(0, 0.5, "\n".join(lines), va="center", fontsize=6.2, family="monospace",
                 color="#0b0b0b" if r["locus_right"] else "#b3261e")
        ax3 = fig.add_axes([0.78, y0 + 0.012, 0.21, H * 0.8])
        evidence(ax3, r)
    fig.text(0.0, 1 - 0.05 * H, title, fontsize=8, weight="bold")
    fig.text(0.62, 1 - 0.42 * H, "$\\Delta\\rho=\\rho(v)-\\rho(h)$ of the 3 strongest rivals (* = truth):", fontsize=6)
    fig.text(0.905, 1 - 0.42 * H, "TrOCR", fontsize=6, color=C_T, weight="bold")
    fig.text(0.955, 1 - 0.42 * H, "PARSeq", fontsize=6, color=C_P, weight="bold")
    fig.savefig(f"{P}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{P}/{name}.png", bbox_inches="tight", dpi=170)
    plt.close(fig)
    json.dump([{k: v for k, v in r.items() if k != "f"} for r in cases], open(f"{ROOT}/results/{name}.json", "w"), indent=1)


def sample(rows, cond, n, seed):
    c = [r for r in rows if cond(r)]
    rng = np.random.RandomState(seed)
    return [c[i] for i in sorted(rng.choice(len(c), min(n, len(c)), replace=False))] if c else []


def main():
    rows = {ds: load(ds) for ds in DSS if os.path.exists(f"{ROOT}/results/preds_{ds}.npy")}
    # Q1: random stratified sample (no conditioning on LOCUS's outcome)
    Q1 = []
    for ds, R in rows.items():
        Q1 += sample(R, lambda r: r["y"] == 0, 1, 0) + sample(R, lambda r: r["y"] == 1, 1, 0)
    gallery("qual_random", Q1, "Random sample (one incorrect and one correct claim per dataset, seed 0)", "")
    Q1b = []
    for ds, R in rows.items():
        Q1b += sample(R, lambda r: r["y"] == 0, 1, 1) + sample(R, lambda r: r["y"] == 1, 1, 1)
    gallery("qual_random2", Q1b, "Random sample, second draw (seed 1)", "")
    # Q2: errors PARSeq agrees with (genuine character errors only): successes and failures
    Q2 = []
    for ds, R in rows.items():
        base = lambda r: r["y"] == 0 and r["parseq_agree"] and r["artefact"] is None
        Q2 += sample(R, lambda r: base(r) and r["p"] < 0.5, 1, 0) + sample(R, lambda r: base(r) and r["p"] >= 0.5, 1, 0)
    gallery("qual_regime_fooled", Q2, "Character errors that PARSeq agrees with: a random LOCUS success and failure per dataset", "")
    # Q3: correct claims PARSeq misreads: successes and failures
    Q3 = []
    for ds, R in rows.items():
        base = lambda r: r["y"] == 1 and not r["parseq_agree"]
        Q3 += sample(R, lambda r: base(r) and r["p"] >= 0.5, 1, 0) + sample(R, lambda r: base(r) and r["p"] < 0.5, 1, 0)
    gallery("qual_regime_misread", Q3, "Correct claims that PARSeq misreads: a random LOCUS success and failure per dataset", "")
    # regime table with artefacts separated
    lines = []
    for ds, R in rows.items():
        mis = [r for r in R if r["y"] == 1 and not r["parseq_agree"]]
        fo = [r for r in R if r["y"] == 0 and r["parseq_agree"]]
        fo_c = [r for r in fo if r["artefact"] is None]
        err = [r for r in R if r["y"] == 0]
        art = sum(r["artefact"] is not None for r in err)
        a = sum(r["p"] >= .5 for r in mis); b = sum(r["p"] < .5 for r in fo_c)
        lines.append(f"{ds} & {len(err)} & {art} ({100*art/max(1,len(err)):.0f}\\%) & {len(mis)} & {a} ({100*a/max(1,len(mis)):.0f}\\%) & "
                     f"{len(fo_c)} & {b} ({100*b/max(1,len(fo_c)):.0f}\\%) \\\\")
    open(f"{ROOT}/paper/tables/qual_counts.tex", "w").write("\n".join(lines) + "\n")
    # audit sheets (manual inspection): 20 random LOCUS errors, and 20 random errors PARSeq agrees with
    os.makedirs(f"{ROOT}/results/audit", exist_ok=True)
    for ds, R in rows.items():
        for tag, cond in (("locus_wrong", lambda r: not r["locus_right"]),
                          ("parseq_fooled", lambda r: r["y"] == 0 and r["parseq_agree"])):
            S = sample(R, cond, 20, 7)
            json.dump([{k: v for k, v in r.items() if k != "f"} for r in S],
                      open(f"{ROOT}/results/audit/{ds}_{tag}.json", "w"), indent=1)
            sheet(S, f"{ROOT}/results/audit/{ds}_{tag}.png")
    if "IAM" in rows:
        teaser(rows)
    print({ds: len(R) for ds, R in rows.items()})


TEASER = ["IAM_test_01212", "IAM_test_02644"]  # chosen from rule-based shortlists after visual label check


def teaser(rows):
    """Fig. 1: two illustrative regimes where comparing hypotheses changes the decision."""
    R = {r["id"]: r for rr in rows.values() for r in rr}
    a, b = R[TEASER[0]], R[TEASER[1]]
    H = load_hyp("IAM", MAIN_ENGINE["IAM"])
    fig = plt.figure(figsize=(7.2, 2.6))
    for col, (r, title) in enumerate(((a, "(a) correct claim misread by re-recognition"),
                                      (b, "(b) confident engine error"))):
        x0 = 0.0 if col == 0 else 0.53
        fig.text(x0, 0.97, title, fontsize=8, weight="bold", va="top")
        ax = fig.add_axes([x0, 0.55, 0.2, 0.32]); ax.imshow(Image.open(r["path"]).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        conf = H[r["id"]]["conf"]
        lines = [f"truth  [{r['gt']}]", f"claim  [{r['claim']}]", "",
                 f"engine conf. {conf:.2f}: " + ("accept" + (" (wrong)" if r["y"] == 0 else "")),
                 f"PARSeq [{r['parseq']}]: " + ("accept" if r["parseq_agree"] else "reject") + (" (wrong)" if (r["parseq_agree"] != bool(r["y"])) else ""),
                 f"LOCUS P={r['p']:.2f}: " + ("accept" if r["p"] >= .5 else "reject") + (" (right)" if r["locus_right"] else " (wrong)")]
        if r["y"] == 0:
            lines.append(f"LOCUS repair [{r['repair']}]")
        fig.text(x0 + 0.215, 0.86, "\n".join(lines), fontsize=6.2, family="monospace", va="top")
        ax2 = fig.add_axes([x0 + 0.09, 0.08, 0.37, 0.36])
        evidence(ax2, r)
        ax2.set_title("$\\Delta\\rho$ of the strongest rivals vs. claim (* = truth)", fontsize=6.5)
    fig.text(0.40, -0.04, "bars:", fontsize=6.5); fig.text(0.44, -0.04, "TrOCR", fontsize=6.5, color=C_T, weight="bold")
    fig.text(0.50, -0.04, "PARSeq", fontsize=6.5, color=C_P, weight="bold")
    fig.savefig(f"{P}/teaser.pdf", bbox_inches="tight"); fig.savefig(f"{P}/teaser.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def sheet(cases, out):
    ims = []
    for j, r in enumerate(cases):
        im = Image.open(r["path"]).convert("RGB")
        s = 48 / im.size[1]
        ims.append(im.resize((min(420, max(8, int(im.size[0] * s))), 48)))
    fig = plt.figure(figsize=(9, 0.62 * len(cases)))
    for j, (r, im) in enumerate(zip(cases, ims)):
        ax = fig.add_axes([0, 1 - (j + 1) / len(cases), 0.35, 0.9 / len(cases)]); ax.imshow(im); ax.axis("off")
        fig.text(0.37, 1 - (j + 0.5) / len(cases), f"#{j:02d} truth \"{r['gt']}\"  claim \"{r['claim']}\"  y={r['y']}  "
                 f"P={r['p']:.2f}  PARSeq \"{r['parseq']}\"", fontsize=8, family="monospace", va="center")
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()


def audit_table():
    """Counts of the manual audit labels (results/audit/annotations.json, one letter per sampled word)."""
    a = json.load(open(f"{ROOT}/results/audit/annotations.json"))
    lines = []
    for ds in DSS:
        cells = []
        for tag, cats in (("locus_wrong", "GNCAP"), ("parseq_fooled", "GNCA")):
            lab = a.get(f"{ds}_{tag}", "").split()
            cells += [str(lab.count(c)) if lab else "--" for c in cats]
        lines.append(f"{ds} & " + " & ".join(cells) + " \\\\")
    open(f"{ROOT}/paper/tables/audit.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    audit_table()

"""Dataset overview figure: real test crops from all five benchmarks with their OCR claims.

Per dataset, 2 correct and 3 incorrect claims are drawn by a seeded random choice (seed 3) from the
test split; incorrect examples are restricted to character errors (case- or punctuation-only label
differences are excluded, see make_qual2.artefact). Nothing is chosen by hand.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_qual2 import DSS, P, load, sample

plt.rcParams.update({"font.size": 6, "pdf.fonttype": 42})
INFO = {"FUNSD": ("scanned forms", "word", "Tesseract 5"), "CORD": ("camera receipts", "word", "Tesseract 5"),
        "SROIE": ("scanned receipts", "text line", "Tesseract 5"), "IAM": ("handwriting", "word", "TrOCR-small"),
        "IIIT5K": ("scene text", "word", "Tesseract 5")}
GREEN, RED, INK, MUTED = "#1e7b34", "#b3261e", "#0b0b0b", "#6b6a64"
N_OK, N_ERR, SEED = 2, 3, 3


def short(s, n=20):
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    import common
    for ds in DSS:  # use the real engine name from the shared config
        INFO[ds] = INFO[ds][:2] + ({"tess": "Tesseract 5", "trocr": "TrOCR-small"}[common.MAIN_ENGINE[ds]],)
    ncol = N_OK + N_ERR
    W, rowh, lab = 7.0, 1.02, 1.05
    fig = plt.figure(figsize=(W, rowh * len(DSS) + 0.35))
    H = fig.get_figheight()
    cw = (W - lab) / ncol
    # column headers
    fig.text((lab + cw * N_OK / 2) / W, 1 - 0.12 / H, "claim is correct", ha="center", va="center",
             color=GREEN, weight="bold", fontsize=7)
    fig.text((lab + cw * (N_OK + N_ERR / 2)) / W, 1 - 0.12 / H, "claim is incorrect (character error)",
             ha="center", va="center", color=RED, weight="bold", fontsize=7)
    for i, ds in enumerate(DSS):
        rows = load(ds)
        ok = sample(rows, lambda r: r["y"] == 1, N_OK, SEED)
        bad = sample(rows, lambda r: r["y"] == 0 and r["artefact"] is None, N_ERR, SEED)
        top = H - 0.28 - i * rowh  # inches from bottom
        # row band
        bg = fig.add_axes([0, (top - rowh + 0.04) / H, 1, (rowh - 0.06) / H]); bg.axis("off")
        bg.add_patch(FancyBboxPatch((0.003, 0.02), 0.994, 0.96, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc="#f6f6f3" if i % 2 == 0 else "#ffffff", ec="#d9d8d2", lw=0.5,
                                    transform=bg.transAxes))
        dom, unit, eng = INFO[ds]
        fig.text(0.08 / W, (top - 0.26) / H, ds, fontsize=8, weight="bold", va="center")
        fig.text(0.08 / W, (top - 0.48) / H, f"{dom}\n{unit} level\nengine: {eng}", fontsize=5.6,
                 color=MUTED, va="top", linespacing=1.25)
        for j, r in enumerate(ok + bad):
            x0 = lab + j * cw + 0.06
            ax = fig.add_axes([x0 / W, (top - 0.55) / H, (cw - 0.12) / W, 0.42 / H])
            ax.imshow(Image.open(r["path"]).convert("RGB"), aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            col = GREEN if r["y"] == 1 else RED
            for sp in ax.spines.values():
                sp.set_color(col); sp.set_linewidth(1.0)
            cx = (x0 + (cw - 0.12) / 2) / W
            fig.text(cx, (top - 0.67) / H, f"claim: {short(r['claim'])}", ha="center", va="center",
                     family="monospace", fontsize=5.6, color=col)
            if r["y"] == 0:
                fig.text(cx, (top - 0.82) / H, f"truth: {short(r['gt'])}", ha="center", va="center",
                         family="monospace", fontsize=5.6, color=INK)
            else:
                fig.text(cx, (top - 0.82) / H, "(= truth)", ha="center", va="center", fontsize=5.4, color=MUTED)
    fig.savefig(f"{P}/datasets_examples.pdf", bbox_inches="tight")
    fig.savefig(f"{P}/datasets_examples.png", bbox_inches="tight", dpi=170)


if __name__ == "__main__":
    main()

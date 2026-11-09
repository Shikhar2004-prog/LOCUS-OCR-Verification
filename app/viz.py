"""Drawing helpers shared by the demo app and the video script."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

GREEN, RED, AMBER = (22, 150, 70), (215, 38, 38), (230, 150, 0)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def annotate(page, results, show_ok=True, show_repairs=True, scale=2):
    """Green box = LOCUS accepts the OCR word; red box = flagged for review (with suggested repair)."""
    im = page.convert("RGB").resize((page.width * scale, page.height * scale), Image.LANCZOS)
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f = font(11 * scale // 2 + 4, bold=True)
    for r in results:
        x0, y0, x1, y1 = [v * scale for v in r.box]
        if r.flagged:
            d.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=RED + (255,), width=3, fill=RED + (40,))
        elif show_ok:
            d.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), outline=GREEN + (200,), width=2, fill=GREEN + (22,))
    for r in results:
        if r.flagged and show_repairs:
            x0, y0 = r.box[0] * scale, r.box[1] * scale
            lab = f"{r.claim} → {r.repair}" if r.repair != r.claim else f"{r.claim} ?"
            w = d.textlength(lab, font=f)
            ty = max(0, y0 - f.size - 8)
            d.rounded_rectangle((x0 - 2, ty, x0 + w + 8, ty + f.size + 6), 4, fill=(255, 255, 255, 235),
                                outline=RED + (255,), width=2)
            d.text((x0 + 3, ty + 2), lab, font=f, fill=RED + (255,))
    return Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")


def graph_figure(word, top=9, title=None):
    """Counterfactual string graph of one word: the claim in the centre, its rivals around it.
    Node area = P_opt (LOCUS's local-optimality distribution); colour = rho summed over both verifiers."""
    g = [n for n in word.graph]
    claim = next(n for n in g if n[1] == "claim")
    riv = [n for n in g if n[1] != "claim"][:top]
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=130)
    ang = np.linspace(0, 2 * np.pi, len(riv), endpoint=False) + np.pi / 2
    pos = [(0.0, 0.0)] + [(1.25 * np.cos(a), 0.95 * np.sin(a)) for a in ang]
    nodes = [claim] + riv
    rho = np.array([n[2] + n[3] for n in nodes])
    vmax = max(5.0, np.abs(rho).max())
    cmap = plt.get_cmap("RdYlGn")
    for (x, y), n in zip(pos[1:], riv):
        ax.plot([0, x], [0, y], color="#9aa", lw=1, zorder=1)
        ax.text(x / 2, y / 2, n[1].replace("pgreedy", "PARSeq").replace("greedy", "TrOCR"), fontsize=7,
                color="#577", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5), zorder=2)
    for (x, y), n, r in zip(pos, nodes, rho):
        s = 600 + 5200 * n[4]
        ax.scatter([x], [y], s=s, color=cmap(0.5 + 0.5 * r / vmax), edgecolor="#223" if n[1] != "claim" else "#d62626",
                   linewidth=2.5 if n[1] == "claim" else 1, zorder=3)
        ax.text(x, y + 0.02, n[0], ha="center", va="center", fontsize=10, weight="bold", zorder=4)
        ax.text(x, y - 0.2, f"P_opt {n[4]:.2f}\nρ {r:+.1f}", ha="center", va="top", fontsize=7, color="#333", zorder=4)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.35, 1.35); ax.axis("off")
    ax.set_title(title or f"Counterfactual string graph of the claim “{claim[0]}”", fontsize=11)
    buf = io.BytesIO()
    fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def side_by_side(crop, rendered_claim, rendered_repair, claim, repair):
    """Real crop vs clean renderings R(claim) and R(repair) at the same height."""
    H = 64
    rs = lambda im: im.resize((max(1, int(im.width * H / im.height)), H), Image.LANCZOS)
    items = [("word image x", rs(crop)), (f"R(“{claim}”)", rs(rendered_claim))]
    if repair != claim:
        items.append((f"R(“{repair}”)", rs(rendered_repair)))
    W = sum(i.width for _, i in items) + 30 * (len(items) + 1)
    out = Image.new("RGB", (W, H + 40), "white")
    d = ImageDraw.Draw(out)
    x = 30
    for lab, im in items:
        out.paste(im, (x, 30))
        d.rectangle((x - 1, 29, x + im.width, 30 + H), outline=(170, 170, 170))
        d.text((x, 6), lab, font=font(14), fill=(40, 40, 40))
        x += im.width + 30
    return out

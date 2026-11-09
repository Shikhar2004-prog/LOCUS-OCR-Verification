"""Build the LOCUS demo video (1920x1080, 25 fps, H.264) from real outputs.

Every number shown comes from a file or a live model call:
  - the word "640" is verified live by the exported CORD-trained LOCUS model,
  - demo-page counts come from app/examples/demo_eval.json (evaluate_demo.py),
  - benchmark review workload comes from journal/tables/review_cost*.tex (the paper),
  - the application footage is the real recording app/media/app_session.webm (record_app.py).
usage: python app/make_video.py   -> app/media/LOCUS_demo.mp4
"""
import json, os, re, subprocess, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
MEDIA = os.path.join(HERE, "media")
W, H, FPS = 1920, 1080, 25
BG, INK, MUTED = (250, 250, 247), (28, 32, 40), (95, 102, 115)
ACC, RED, GREEN, BLUE = (40, 90, 200), (205, 40, 40), (25, 140, 70), (60, 110, 210)
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_fc = {}


def F(size, kind="r"):
    k = (size, kind)
    if k not in _fc:
        _fc[k] = ImageFont.truetype({"r": FR, "b": FB, "m": FM}[kind], size)
    return _fc[k]


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def blend(c, a):
    return tuple(int(BG[i] + (c[i] - BG[i]) * a) for i in range(3))


def canvas(title=None, section=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 10), fill=ACC)
    if section:
        d.text((90, 52), section.upper(), font=F(26, "b"), fill=ACC)
    if title:
        d.text((90, 92), title, font=F(56, "b"), fill=INK)
    if section != "Benchmark results from the paper":
        d.text((W - 90, H - 50), "LOCUS • OCR verification", font=F(22), fill=MUTED, anchor="rs")
    return im, d


def text_block(d, x, y, lines, t, start=0.6, step=0.9, size=36, gap=22, width=None):
    """Bullets that fade in one after another. lines: str or (str, color, bold)."""
    for i, ln in enumerate(lines):
        a = ease((t - start - i * step) / 0.5)
        if a <= 0:
            continue
        s, col, bold = (ln, INK, False) if isinstance(ln, str) else ln
        d.text((x, y + 8 * (1 - a)), s, font=F(size, "b" if bold else "r"), fill=blend(col, a))
        y += size + gap
    return y


def paste(im, src, xy, a=1.0, box=None):
    if box:
        src = src.copy(); src.thumbnail(box, Image.LANCZOS)
    if a >= 1:
        im.paste(src, xy)
    elif a > 0:
        reg = im.crop((xy[0], xy[1], xy[0] + src.width, xy[1] + src.height))
        im.paste(Image.blend(reg, src.convert("RGB"), a), xy)
    return src.size


# ----------------------------------------------------------------- data
def load_data():
    from PIL import Image as I
    from locus_engine import LocusEngine, render_hypothesis
    from viz import annotate, graph_figure
    import locus_engine as le
    ev = json.load(open(os.path.join(HERE, "examples", "demo_eval.json")))["CORD"]
    eng = LocusEngine("CORD")
    crop = I.open(os.path.join(HERE, "examples", "words", "invoice_0.png")).convert("RGB")
    w = eng.verify([crop], ["640"])[0]
    page = I.open(os.path.join(HERE, "examples", "invoice.png")).convert("RGB")
    res = le.verify_page(eng, page)
    rev = {}
    for f in ("review_cost_a.tex", "review_cost_b.tex"):
        ds = None
        for line in open(os.path.join(ROOT, "journal", "tables", f)):
            cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
            if len(cells) < 5:
                continue
            ds = cells[0] or ds
            num = lambda c: float(re.sub(r"[^0-9.]", "", c.split("{")[-1]) or "nan")
            rev.setdefault(ds, {})[cells[1]] = num(cells[3])  # 95% precision column
    return dict(ev=ev, word=w, crop=crop, r_claim=render_hypothesis("640", 64), r_rep=render_hypothesis(w.repair, 64),
                graph=graph_figure(w, top=8, title=" "), page=page, annotated=annotate(page, res, scale=1), rev=rev,
                eng=eng)


# ----------------------------------------------------------------- slides
def s_title(t, D):
    im, d = canvas()
    a = ease(t / 0.8)
    d.text((W // 2, 360), "LOCUS", font=F(140, "b"), fill=blend(ACC, a), anchor="mm")
    d.text((W // 2, 490), "Is the OCR output right?  Verifying every word before anyone relies on it",
           font=F(44), fill=blend(INK, ease((t - 0.6) / 0.8)), anchor="mm")
    d.text((W // 2, 600), "Local Optimality over Counterfactual String graphs", font=F(34), fill=blend(MUTED, ease((t - 1.2) / 0.8)), anchor="mm")
    d.text((W // 2, 760), "Shikhar Dubey  •  Sanhita Saxena", font=F(36, "b"), fill=blend(INK, ease((t - 1.8) / 0.8)), anchor="mm")
    d.text((W // 2, 815), "Indian Institute of Technology Mandi", font=F(30), fill=blend(MUTED, ease((t - 1.8) / 0.8)), anchor="mm")
    return im


def s_problem(t, D):
    im, d = canvas("OCR always returns an answer, even when it is wrong", "The problem")
    page = D["page"].crop((540, 470, 800, 600)).resize((780, 390), Image.LANCZOS)
    paste(im, page, (90, 220), ease(t / 0.6))
    d.rectangle((90, 220, 870, 610), outline=(200, 200, 200), width=2)
    d.text((90, 625), "Part of a scanned invoice (synthetic, degraded)", font=F(24), fill=MUTED)
    y = text_block(d, 960, 230, [
        ("Tesseract reads the unit price as:", INK, False),
        ("  “640”    confidence 0.92", RED, True),
        ("The page actually says:", INK, False),
        ("  “6.40”", GREEN, True)], t, start=0.8, step=0.8, size=40)
    text_block(d, 960, y + 40, [
        "→ 5 × 6.40 = 32.00 becomes 5 × 640 = 3,200.00",
        "→ the engine is confident, so nobody looks",
        ("Checking every word by hand does not scale.", INK, True),
        ("Trusting engine confidence misses errors like this.", INK, True)], t, start=4.2, step=1.0, size=32)
    return im


def s_idea(t, D):
    im, d = canvas("LOCUS: verify each OCR word with evidence", "The idea")
    text_block(d, 90, 230, [
        ("Question: is the claimed string the best explanation of the image", INK, True),
        ("among the strings the engine could have confused it with?", INK, True)], t, start=0.4, step=0.6, size=40)
    steps = [("1", "Counterfactual string graph", "confusable edits (0/O, 5/S, 1/l, .), case flips,\ndeletions, insertions, and readings of two frozen recognisers"),
             ("2", "Render-normalised evidence", "ρ(v) = s(v | real image) − s(v | clean rendering of v)\ncancels the recogniser's bias towards common strings"),
             ("3", "Graph neural network", "GATv2 over the graph; local-optimality readout P_opt\ngives P(correct) and a repair"),
             ("4", "Label-efficient", "pretrained without labels; adapted to a new\ndocument type with only 25 labelled words per class")]
    for i, (n, h, s) in enumerate(steps):
        a = ease((t - 2.0 - i * 1.3) / 0.6)
        if a <= 0:
            continue
        x = 90 + i * 445
        d.rounded_rectangle((x, 420, x + 420, 820), 18, fill=blend((235, 241, 252), a), outline=blend(ACC, a), width=3)
        d.ellipse((x + 24, 444, x + 84, 504), fill=blend(ACC, a))
        d.text((x + 54, 474), n, font=F(34, "b"), fill=blend((255, 255, 255), a), anchor="mm")
        d.text((x + 100, 446), "\n".join(_wrap(h, 17)), font=F(29, "b"), fill=blend(INK, a), spacing=4)
        d.text((x + 24, 560), "\n".join(_wrap(s.replace("\n", " "), 31)), font=F(24), fill=blend(MUTED, a), spacing=9)
    return im


def _wrap(s, n):
    out, cur = [], ""
    for w in s.split(" "):
        if len(cur) + len(w) + 1 > n and cur:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return out + [cur]


def s_word(t, D):
    w = D["word"]
    im, d = canvas("One word, step by step: “640”", "How LOCUS decides (live model output)")
    H0 = 110
    rs = lambda x: x.resize((int(x.width * H0 / x.height), H0), Image.LANCZOS)
    paste(im, rs(D["crop"]), (90, 250), ease(t / 0.5))
    d.text((90, 215), "real word image x", font=F(26), fill=MUTED)
    paste(im, rs(D["r_claim"]), (90, 450), ease((t - 1.0) / 0.5))
    d.text((90, 415), "clean rendering R(“640”)", font=F(26), fill=blend(MUTED, ease((t - 1.0) / 0.5)))
    paste(im, rs(D["r_rep"]), (90, 650), ease((t - 2.0) / 0.5))
    d.text((90, 615), f"clean rendering R(“{w.repair}”)", font=F(26), fill=blend(MUTED, ease((t - 2.0) / 0.5)))
    g = {v: (a, b, q) for v, k, a, b, q in w.graph}
    ca, cb, _ = g[w.claim]
    ra, rb, _ = g[w.repair]
    text_block(d, 90, 800, [
        (f"ρ(“640”) = {ca + cb:+.1f}", RED, True),
        ("   image supports “640” far less than a clean “640”", RED, False),
        (f"ρ(“{w.repair}”) = {ra + rb:+.1f}", GREEN, True),
        (f"   image supports “{w.repair}” as well as a clean rendering", GREEN, False)],
        t, start=3.0, step=0.7, size=28, gap=12)
    gi = D["graph"]
    paste(im, gi, (1000, 185), ease((t - 4.5) / 0.8), box=(860, 700))
    a = ease((t - 6.5) / 0.6)
    if a > 0:
        d.rounded_rectangle((960, 900, 1860, 1010), 16, fill=blend((253, 236, 236), a), outline=blend(RED, a), width=3)
        d.text((990, 916), f"Verdict: P(correct) = {w.p_correct:.2f}  →  flag for review", font=F(30, "b"), fill=blend(RED, a))
        d.text((990, 960), f"Suggested repair: “{w.repair}”   (Tesseract confidence: 0.92)", font=F(26), fill=blend(INK, a))
    return im


def s_demo_card(t, D):
    im, d = canvas("The application", "Live demo")
    text_block(d, 90, 260, [
        ("Real screen recording of the LOCUS web app (app/app.py)", INK, True),
        "1.  Upload a scanned invoice: Tesseract reads it, LOCUS checks every word",
        "     red boxes = flagged for review, with the suggested repair; green = accepted",
        "2.  A faded thermal receipt",
        "3.  Inspect one word: the counterfactual graph and the evidence behind the verdict",
        ("Runs locally on one GPU: python app/app.py  →  http://localhost:7860", ACC, False)], t, start=0.3, step=0.7, size=36)
    return im


def s_page(t, D):
    ev = D["ev"]
    im, d = canvas("What LOCUS caught on the demo pages", "Measured, word-level truth known")
    ann = D["annotated"].crop((0, 0, 820, 700))
    paste(im, ann, (90, 200), ease(t / 0.6), box=(820, 760))
    rows = [("invoice", "Invoice"), ("receipt", "Receipt"), ("claim_form", "Claim form")]
    x0, y = 980, 230
    for i, (k, lab) in enumerate(rows):
        a = ease((t - 0.8 - i * 0.9) / 0.6)
        if a <= 0:
            continue
        e = ev[k]
        d.text((x0, y), f"{lab}: {e['words']} words, {e['errors']} OCR errors", font=F(32, "b"), fill=blend(INK, a))
        y += 50
        for name, val, col in ((f"LOCUS (reviewing {e['flagged']} words)", e["errors_caught"], GREEN),
                               (f"Tesseract confidence, same {e['flagged']} words", e["errors_caught_by_conf_same_budget"], (150, 150, 150))):
            L = int(560 * ease((t - 1.2 - i * 0.9) / 1.0) * val / max(1, e["errors"]))
            d.rectangle((x0, y + 6, x0 + 560, y + 34), fill=blend((232, 232, 232), a))
            d.rectangle((x0, y + 6, x0 + L, y + 34), fill=blend(col, a))
            d.text((x0 + 575, y + 3), f"{val}/{e['errors']} caught", font=F(26, "b"), fill=blend(INK, a))
            d.text((x0, y + 38), name, font=F(22), fill=blend(MUTED, a))
            y += 78
        y += 30
    text_block(d, 980, 930, [("Honest note: on these pages, flagging disagreement with a strong", MUTED, False),
                             ("second recogniser (PARSeq) catches about as many errors.", MUTED, False)],
               t, start=4.0, step=0.0, size=24, gap=8)
    return im


def s_workload(t, D):
    rev = D["rev"]
    im, d = canvas("Practical impact: how much must a person still check?", "Benchmark results from the paper")
    d.text((90, 180), "Share of all words a human must review so that the auto-accepted words are 95% correct (k = 25; lower is better)",
           font=F(28), fill=MUTED)
    dss = ["FUNSD", "CORD", "SROIE", "IAM", "IIIT5K"]
    desc = {"FUNSD": "forms", "CORD": "receipts", "SROIE": "receipts", "IAM": "handwriting", "IIIT5K": "scene text"}
    meth = [("Engine confidence", (170, 170, 170)), ("PARSeq re-OCR (graded)", (120, 160, 225)), ("LOCUS", GREEN)]
    base, top = 850, 330
    for i, ds in enumerate(dss):
        x = 160 + i * 340
        a = ease((t - 0.5 - i * 0.5) / 0.6)
        d.text((x + 120, base + 20), ds, font=F(32, "b"), fill=blend(INK, a), anchor="ma")
        d.text((x + 120, base + 60), desc[ds], font=F(24), fill=blend(MUTED, a), anchor="ma")
        for j, (m, col) in enumerate(meth):
            v = rev[ds][m]
            h = (base - top) * v / 100 * ease((t - 0.7 - i * 0.5) / 1.0)
            bx = x + j * 80
            d.rectangle((bx, base - h, bx + 70, base), fill=blend(col, a))
            if h > 0:
                d.text((bx + 35, base - h - 34), f"{v:.0f}%", font=F(24, "b"), fill=blend(INK, a), anchor="ma")
    d.line((120, base, W - 120, base), fill=(160, 160, 160), width=2)
    lx = 160
    for m, col in meth:
        d.rectangle((lx, 236, lx + 28, 264), fill=col)
        d.text((lx + 40, 234), m, font=F(26), fill=INK)
        lx += 70 + int(d.textlength(m, font=F(26)))
    a = ease((t - 4.5) / 0.8)
    if a > 0:
        d.text((W // 2, 1000), "Receipts (SROIE): 45% instead of 91%   \u2022   Handwriting (IAM): 51% instead of 91%",
               font=F(30, "b"), fill=blend(INK, a), anchor="ms")
        d.text((W // 2, 1042), "PARSeq re-reading ties or wins on FUNSD and IIIT5K", font=F(24), fill=blend(MUTED, a), anchor="ms")
    return im


def s_uses(t, D):
    im, d = canvas("Where this helps", "Applications")
    items = [("Accounts payable", "invoices and receipts: catch wrong amounts, dates and invoice numbers before payment"),
             ("Insurance and banking", "claim forms and KYC documents: route only doubtful fields to a person"),
             ("Healthcare and government intake", "forms where one wrong digit changes a record"),
             ("Digital archives and libraries", "handwritten and historical pages: tell editors which words to proofread"),
             ("Any OCR engine", "LOCUS sits after Tesseract (or any engine) and needs only 25 labels per class for a new document type")]
    y = 230
    for i, (h, s) in enumerate(items):
        a = ease((t - 0.4 - i * 0.9) / 0.6)
        if a > 0:
            d.ellipse((96, y + 12, 120, y + 36), fill=blend(ACC, a))
            d.text((145, y), h, font=F(38, "b"), fill=blend(INK, a))
            d.text((145, y + 52), s, font=F(30), fill=blend(MUTED, a))
        y += 150
    return im


def s_limits(t, D):
    im, d = canvas("Limitations, stated plainly", "Honest assessment")
    text_block(d, 90, 240, [
        "•  A strong second recogniser (PARSeq) is a hard baseline: it ties LOCUS on forms (FUNSD)",
        "    and is better on scene text (IIIT5K).",
        "•  Exact repairs: substituting PARSeq's reading fixes more errors on 4 of 5 datasets.",
        "•  Speed: about 18 words/s on one GPU (two frozen recognisers, real image + renderings).",
        "•  Latin script, English benchmarks; a wrong Tesseract word box can be flagged but not repaired.",
        ("The strength of LOCUS is consistency: it is near the best on all five benchmarks,", INK, True),
        ("while the best re-reading baseline changes from dataset to dataset.", INK, True)], t, start=0.4, step=0.8, size=34)
    return im


def s_end(t, D):
    im, d = canvas()
    a = ease(t / 0.8)
    d.text((W // 2, 330), "Try it", font=F(80, "b"), fill=blend(ACC, a), anchor="mm")
    d.text((W // 2, 470), "github.com/Shikhar2004-prog/LOCUS-OCR-Verification", font=F(40, "m"), fill=blend(INK, a), anchor="mm")
    d.text((W // 2, 560), "pip install -r app/requirements.txt   •   python app/app.py", font=F(34, "m"), fill=blend(MUTED, a), anchor="mm")
    d.text((W // 2, 720), "Shikhar Dubey  •  Sanhita Saxena  •  IIT Mandi", font=F(36, "b"), fill=blend(INK, ease((t - 0.8) / 0.8)), anchor="mm")
    d.text((W // 2, 790), "Demo documents are synthetic; benchmark numbers are from the paper.", font=F(26), fill=blend(MUTED, ease((t - 0.8) / 0.8)), anchor="mm")
    return im


def render(slide, seconds, D, path):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
                         stdin=subprocess.PIPE)
    n = int(seconds * FPS)
    last = None
    for i in range(n):
        t = i / FPS
        fade = min(1.0, t / 0.35, (seconds - t) / 0.35)
        im = slide(t, D)
        if fade < 1:
            im = Image.blend(Image.new("RGB", (W, H), BG), im, max(0.0, fade))
        p.stdin.write(im.tobytes())
    p.stdin.close(); p.wait()


def demo_clip(path):
    """Real app recording, scaled into the frame with a caption bar."""
    src = os.path.join(MEDIA, "app_session.webm")
    vf = (f"scale=1584:990,pad={W}:{H}:(ow-iw)/2:72:color=0xFAFAF7,"
          f"drawtext=fontfile={FB}:text='Real screen recording of the LOCUS web application':x=168:y=22:fontsize=30:fontcolor=0x1C2028")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-r", str(FPS), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-an", path], check=True)


def main():
    D = load_data()
    print("word:", D["word"].claim, D["word"].p_correct, D["word"].repair, flush=True)
    tmp = os.path.join(MEDIA, "_parts")
    os.makedirs(tmp, exist_ok=True)
    plan = [(s_title, 5), (s_problem, 10), (s_idea, 10), (s_word, 11), (s_demo_card, 6), ("clip", None),
            (s_page, 9), (s_workload, 10), (s_uses, 8), (s_limits, 10), (s_end, 6)]
    if os.environ.get("PREVIEW"):  # last frame of every slide, for layout checks
        os.makedirs(os.path.join(MEDIA, "_preview"), exist_ok=True)
        for i, (s, sec) in enumerate(plan):
            if s != "clip":
                s(sec - 0.5, D).save(os.path.join(MEDIA, "_preview", f"{i:02d}.png"))
        return
    parts = []
    for i, (s, sec) in enumerate(plan):
        out = os.path.join(tmp, f"{i:02d}.mp4")
        if s == "clip":
            demo_clip(out)
        else:
            render(s, sec, D, out)
        parts.append(out)
        print("part", i, "done", flush=True)
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    final = os.path.join(MEDIA, "LOCUS_demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", final], check=True)
    print("saved", final)


if __name__ == "__main__":
    main()

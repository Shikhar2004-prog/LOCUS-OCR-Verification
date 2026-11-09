"""Build the no-install demo in ../demo: index.html + data.js (+ the video).

Runs the exported CORD-trained LOCUS once on the three synthetic demo pages and stores every word's crop, Tesseract
claim and confidence, true text, P(correct), repair, counterfactual graph and clean renderings in demo/data.js.
The page then opens in any browser with no server, GPU or Python.
usage: python app/make_static_demo.py
"""
import base64, io, json, os, re, shutil, sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "demo")
sys.path.insert(0, HERE)
from locus_engine import KIND_LABEL, LocusEngine, crop_word, render_hypothesis, verify_page  # noqa: E402
from make_demo_docs import match_truth  # noqa: E402

DOCS = [("invoice", "Degraded invoice"), ("receipt", "Faded receipt"), ("claim_form", "Insurance claim form")]
MODEL = "CORD"
TOP = 8

BENCH = {  # paper, k = 25 (README / ABOUT.md): F1, AUROC per dataset
    "cols": ["FUNSD", "CORD", "SROIE", "IAM", "IIIT5K"],
    "rows": [
        ["Engine confidence", [.756, .861], [.777, .892], [.780, .859], [.802, .878], [.788, .938]],
        ["ConfidNet-style learned confidence", [.771, .845], [.842, .929], [.826, .912], [.801, .861], [.922, .980]],
        ["Two-engine voting", [.609, .680], [.772, .807], [.976, .976], [.759, .767], [.862, .905]],
        ["Re-OCR: TrOCR-base", [.605, .675], [.811, .841], [.979, .978], [.758, .766], [.952, .966]],
        ["Re-OCR: Florence-2", [.647, .715], [.615, .685], [.478, .578], [.369, .522], [.823, .879]],
        ["Re-OCR: PARSeq (graded)", [.946, .976], [.973, .986], [.742, .760], [.879, .914], [.993, .995]],
        ["VLM judge: Qwen2.5-VL-7B", [.760, .846], [.891, .945], [.866, .941], [.700, .788], [.888, .974]],
        ["LOCUS", [.943, .974], [.978, .993], [.973, .992], [.906, .969], [.971, .991]],
    ]}


def b64(im, fmt="PNG", q=85):
    buf = io.BytesIO()
    im.save(buf, fmt, **({"quality": q, "optimize": True} if fmt == "JPEG" else {"optimize": True}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def review_cost():
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
    return rev


def main():
    os.makedirs(OUT, exist_ok=True)
    eng = LocusEngine(MODEL)
    renders, pages = {}, []

    def rend(t):
        if t not in renders:
            renders[t] = b64(render_hypothesis(t, 40).convert("L"))
        return t

    featured = {}
    for fn, claim in json.load(open(os.path.join(HERE, "examples", "words", "index.json"))):
        featured.setdefault(fn.rsplit("_", 1)[0], []).append(claim)

    for name, title in DOCS:
        page = Image.open(os.path.join(HERE, "examples", f"{name}.png")).convert("RGB")
        gt = json.load(open(os.path.join(HERE, "examples", f"{name}.json")))["words"]
        res = verify_page(eng, page)
        words = []
        for i, r in enumerate(res):
            g = match_truth(gt, r.box)
            truth = g["text"] if g else None
            graph = [[v, k, round(rt, 3), round(rp, 3), round(q, 4)] for v, k, rt, rp, q in r.graph[:TOP]]
            for v in [r.claim, r.repair] + [x[0] for x in graph]:
                rend(v)
            words.append({"i": i, "claim": r.claim, "truth": truth, "correct": int(truth == r.claim),
                          "p": round(r.p_correct, 4), "conf": r.engine_conf, "repair": r.repair,
                          "box": [int(v) for v in r.box], "trocr": r.trocr, "parseq": r.parseq,
                          "parseq_conf": round(r.parseq_conf, 3), "graph": graph,
                          "crop": b64(crop_word(page, r.box).convert("L"))})
        feat = []
        for c in featured.get(name, []):
            hit = [w["i"] for w in words if w["claim"] == c and w["i"] not in feat]
            if hit:
                feat.append(hit[0])
        y = np.array([w["correct"] for w in words]); p = np.array([w["p"] for w in words])
        print(f"{name}: {len(words)} words, {int((y == 0).sum())} errors, flagged {int((p < .5).sum())}, "
              f"caught {int(((p < .5) & (y == 0)).sum())}", flush=True)
        pages.append({"id": name, "title": title, "w": page.width, "h": page.height,
                      "img": b64(page, "JPEG", 82), "words": words, "featured": feat})

    meta = {"model": MODEL, "model_label": "Receipts (CORD-trained), k = 25 labels per class, seed 0",
            "kind_label": KIND_LABEL, "bench": BENCH, "review": review_cost(),
            "model_scores": {ds: torch_scores(ds) for ds in ("SROIE", "CORD", "FUNSD", "IAM", "IIIT5K")}}
    with open(os.path.join(OUT, "data.js"), "w") as f:
        f.write("window.LOCUS_META = " + json.dumps(meta) + ";\n")
        f.write("window.LOCUS_PAGES = " + json.dumps(pages) + ";\n")
        f.write("window.LOCUS_RENDERS = " + json.dumps(renders) + ";\n")
    shutil.copy(os.path.join(HERE, "media", "LOCUS_demo.mp4"), os.path.join(OUT, "LOCUS_demo.mp4"))
    print("saved", os.path.join(OUT, "data.js"), f"{os.path.getsize(os.path.join(OUT, 'data.js')) / 1e6:.1f} MB")


def torch_scores(ds):
    import torch
    ck = torch.load(os.path.join(HERE, "models", f"locus_{ds}.pt"), map_location="cpu", weights_only=False)
    m = ck["test_metrics"]
    return {k: round(float(m[k]), 3) for k in m if isinstance(m[k], (int, float))} | {"n_test": ck.get("n_test")}


if __name__ == "__main__":
    main()

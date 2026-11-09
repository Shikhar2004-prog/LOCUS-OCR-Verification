"""LOCUS demo application: verify what OCR read, before anyone relies on it.

Run:  python app/app.py            (then open http://localhost:7860)
      python app/app.py --share    (temporary public link)
"""
import argparse, json, os, sys

os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gradio_tmp"))
import gradio as gr  # noqa: E402
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from locus_engine import DOMAINS, KIND_LABEL, LocusEngine, crop_word, render_hypothesis, verify_page  # noqa: E402
from viz import annotate, graph_figure, side_by_side  # noqa: E402

DEFAULT = "CORD"
LABEL2DS = {v: k for k, v in DOMAINS.items()}
_ENGINES = {}


def engine(label):
    ds = LABEL2DS.get(label, DEFAULT)
    if ds not in _ENGINES:
        _ENGINES[ds] = LocusEngine(ds)
    return _ENGINES[ds]


def run_document(img, domain, threshold, psm):
    if img is None:
        raise gr.Error("Upload a document image first.")
    page = Image.fromarray(img).convert("RGB") if not isinstance(img, Image.Image) else img.convert("RGB")
    eng = engine(domain)
    res = verify_page(eng, page, threshold=threshold, psm=int(psm))
    flagged = [r for r in res if r.flagged]
    n = len(res)
    summary = (f"### {n} words read by Tesseract → **{len(flagged)} flagged for review** "
               f"({100 * len(flagged) / max(1, n):.0f}% of the page)\n"
               f"The other {n - len(flagged)} words are accepted automatically. Red boxes show the flagged words "
               f"with LOCUS's suggested repair; green boxes are accepted words.\n\n"
               f"Model: {domain} • threshold P(correct) < {threshold:.2f} • exported model's held-out test "
               f"macro-F1 {eng.ck['test_metrics']['macro_f1']:.3f}, AUROC {eng.ck['test_metrics']['auroc']:.3f} "
               f"on {eng.ck['n_test']} words of {eng.ds}")
    rows = sorted(res, key=lambda r: r.p_correct)
    df = pd.DataFrame([{"OCR word (claim)": r.claim, "P(correct)": round(r.p_correct, 3),
                        "Decision": "REVIEW" if r.flagged else "accept",
                        "LOCUS suggestion": r.repair if r.flagged else "",
                        "Tesseract confidence": round(r.engine_conf, 2), "PARSeq reading": r.parseq,
                        "TrOCR reading": r.trocr} for r in rows])
    gallery = [(crop_word(page, r.box), f"{r.claim} → {r.repair}  (P={r.p_correct:.2f})") for r in rows if r.flagged][:40]
    def fix(r):  # apply single-word repairs; a multi-word repair means Tesseract's box was wrong -> mark only
        if not r.flagged:
            return r.claim
        return r.repair if len(r.repair.split()) <= len(r.claim.split()) else f"[{r.claim}?]"
    corrected = " ".join(fix(r) for r in res)  # Tesseract reading order
    report = os.path.join(HERE, "last_report.json")
    json.dump([{"claim": r.claim, "p_correct": r.p_correct, "flagged": r.flagged, "suggestion": r.repair,
                "box": r.box, "tesseract_conf": r.engine_conf} for r in res], open(report, "w"), indent=1)
    return annotate(page, res), summary, df, gallery, corrected, report


def run_word(img, claim, domain):
    if img is None or not claim:
        raise gr.Error("Give a word image and the text the OCR engine claimed for it.")
    crop = Image.fromarray(img).convert("RGB") if not isinstance(img, Image.Image) else img.convert("RGB")
    r = engine(domain).verify([crop], [claim])[0]
    head = "❌ Likely WRONG" if r.flagged else "✅ Likely correct"
    verdict = (f"## {head} — P(correct) = {r.p_correct:.3f}\n"
               + (f"Suggested repair: **{r.repair}**\n\n" if r.flagged else "\n")
               + f"Readings of the frozen verifiers: TrOCR “{r.trocr}”, PARSeq “{r.parseq}” "
               f"(confidence {r.parseq_conf:.2f}).")
    df = pd.DataFrame([{"hypothesis v": v, "how it was generated": KIND_LABEL[k], "ρ TrOCR": round(a, 2),
                        "ρ PARSeq": round(b, 2), "P_opt (LOCUS)": round(q, 3)} for v, k, a, b, q in r.graph[:15]])
    sbs = side_by_side(crop, render_hypothesis(r.claim), render_hypothesis(r.repair), r.claim, r.repair)
    return verdict, graph_figure(r), df, sbs


EX = os.path.join(HERE, "examples")
ABOUT = open(os.path.join(HERE, "ABOUT.md")).read() if os.path.exists(os.path.join(HERE, "ABOUT.md")) else ""


def build():
    with gr.Blocks(title="LOCUS OCR verification", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# LOCUS — is the OCR output right?\n"
                    "OCR engines always return text, even when it is wrong. LOCUS checks every word: it builds the "
                    "strings the engine could have confused it with, asks two frozen recognisers which one the image "
                    "really supports (render-normalised likelihood ratio), and returns P(correct) plus a repair.")
        with gr.Tab("1. Verify a document"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(label="Document image (scan or photo)", type="pil", elem_id="doc_img")
                    dom = gr.Dropdown(list(DOMAINS.values()), value=DOMAINS[DEFAULT], label="Domain model")
                    thr = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Flag a word when P(correct) is below")
                    psm = gr.Dropdown(["3", "4", "6", "11"], value="3", label="Tesseract page segmentation mode")
                    btn = gr.Button("Read and verify", variant="primary")
                    gr.Examples([[os.path.join(EX, f"{n}.png")] for n in ("invoice", "receipt", "claim_form")], [inp],
                                label="Synthetic degraded documents (word-level truth known)")
                with gr.Column(scale=2):
                    out_img = gr.Image(label="Red = flagged for human review (with suggested repair), green = accepted")
                    summ = gr.Markdown()
            gal = gr.Gallery(label="Flagged word crops", columns=8, height=230, object_fit="contain")
            tab = gr.Dataframe(label="Every word, least trustworthy first", wrap=True)
            txt = gr.Textbox(label="Text with LOCUS suggestions applied to flagged words", lines=5)
            rep = gr.File(label="Verification report (JSON)")
            btn.click(run_document, [inp, dom, thr, psm], [out_img, summ, tab, gal, txt, rep])
        with gr.Tab("2. Inspect one word"):
            with gr.Row():
                with gr.Column(scale=1):
                    w_img = gr.Image(label="Word image", type="pil", elem_id="word_img")
                    w_claim = gr.Textbox(label="What the OCR engine claimed")
                    w_dom = gr.Dropdown(list(DOMAINS.values()), value=DOMAINS[DEFAULT], label="Domain model")
                    w_btn = gr.Button("Verify this word", variant="primary")
                    wex = [[os.path.join(EX, "words", f), c] for f, c in json.load(open(os.path.join(EX, "words", "index.json")))] \
                        if os.path.exists(os.path.join(EX, "words", "index.json")) else []
                    if wex:
                        gr.Examples(wex, [w_img, w_claim], label="Word crops from the demo documents")
                with gr.Column(scale=2):
                    w_verdict = gr.Markdown()
                    w_sbs = gr.Image(label="The real image vs clean renderings R(v): the cue ρ compares them")
                    w_graph = gr.Image(label="Counterfactual string graph", show_label=False)
            w_tab = gr.Dataframe(label="Hypotheses in the graph (ρ = s(v|x) - s(v|R(v)); P_opt = LOCUS local-optimality)")
            w_btn.click(run_word, [w_img, w_claim, w_dom], [w_verdict, w_graph, w_tab, w_sbs])
        with gr.Tab("About & results"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build().queue().launch(server_name="0.0.0.0", server_port=a.port, share=a.share)

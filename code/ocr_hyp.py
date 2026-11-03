"""Run the OCR engines whose outputs we verify, and record their native confidences.

Engines: Tesseract 5 (LSTM, psm 7 = single text line) and TrOCR-small (printed or
handwritten, per dataset). Output: data/<DS>/hyp_<engine>.jsonl with
{id, hyp, conf, ...} where conf is the engine's own confidence (baseline signal).
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

ROOT = os.environ.get("LOCUS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMALL = {"FUNSD": "printed", "CORD": "printed", "SROIE": "printed", "IAM": "handwritten",
         "IIIT5K": "printed", "OCR10K": "printed"}
BASE = {"FUNSD": "printed", "CORD": "printed", "SROIE": "printed", "IAM": "handwritten",
        "IIIT5K": "str", "OCR10K": "printed"}


def load_manifest(ds):
    with open(f"{ROOT}/data/{ds}/manifest.jsonl") as f:
        return [json.loads(l) for l in f]


def _tess(row):
    import pytesseract
    img = Image.open(row["path"]).convert("L")
    if img.size[1] < 32:  # Tesseract degrades badly on tiny text; upscale consistently
        s = 32 / img.size[1]
        img = img.resize((max(1, int(img.size[0] * s)), 32), Image.BICUBIC)
    d = pytesseract.image_to_data(img, config="--psm 7", output_type=pytesseract.Output.DICT)
    words = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip()]
    hyp = " ".join(t for t, _ in words)
    conf = float(np.mean([c for _, c in words])) / 100 if words else 0.0
    minc = float(np.min([c for _, c in words])) / 100 if words else 0.0
    return {"id": row["id"], "hyp": hyp, "conf": conf, "min_conf": minc}


def run_tesseract(ds, rows):
    with Pool(32) as p:
        out = p.map(_tess, rows, chunksize=16)
    return out


@torch.no_grad()
def run_trocr(ds, rows, device="cuda", size="small"):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    name = f"microsoft/trocr-{size}-{SMALL[ds] if size == 'small' else BASE[ds]}"
    proc = TrOCRProcessor.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name).to(device).eval().half()
    out, B = [], 128
    for i in range(0, len(rows), B):
        batch = rows[i:i + B]
        imgs = [Image.open(r["path"]).convert("RGB") for r in batch]
        pv = proc(images=imgs, return_tensors="pt").pixel_values.to(device).half()
        g = model.generate(pv, max_new_tokens=48, num_beams=1, output_scores=True,
                           return_dict_in_generate=True)
        seqs = g.sequences
        lps = torch.stack([torch.log_softmax(s.float(), -1) for s in g.scores], 1)  # B,T,V
        tok = seqs[:, 1:1 + lps.shape[1]]
        tlp = lps.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        mask = (tok != proc.tokenizer.pad_token_id).float()
        texts = proc.batch_decode(seqs, skip_special_tokens=True)
        for r, t, lp, m in zip(batch, texts, tlp, mask):
            v = lp[m.bool()]
            out.append({"id": r["id"], "hyp": t.strip(),
                        "conf": float(v.mean().exp()) if len(v) else 0.0,
                        "min_conf": float(v.min().exp()) if len(v) else 0.0,
                        "seq_lp": float(v.sum()) if len(v) else -50.0})
    return out


if __name__ == "__main__":
    ds, engine = sys.argv[1], sys.argv[2]
    rows = load_manifest(ds)
    out = run_tesseract(ds, rows) if engine == "tess" else run_trocr(
        ds, rows, size="base" if engine == "trocrB" else "small")
    with open(f"{ROOT}/data/{ds}/hyp_{engine}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(ds, engine, len(out))

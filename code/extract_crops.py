"""Turn each public dataset into a unified pool of text crops + ground-truth strings.

Output: data/<DS>/crops/<split>_<idx>.png and data/<DS>/manifest.jsonl with
{id, split, gt, path}. Splits follow the official dataset splits (train/test);
pools are randomly capped (fixed seed) to keep feature extraction tractable.
"""
import json, os, random, sys
from datasets import load_from_disk

ROOT = os.environ.get("LOCUS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAP = {"train": 8000, "test": 4000}
rng = random.Random(0)


def save(ds_name, items):
    out = os.path.join(ROOT, "data", ds_name)
    os.makedirs(os.path.join(out, "crops"), exist_ok=True)
    rows = []
    for split in ("train", "test"):
        cand = [it for it in items if it[0] == split]
        rng.shuffle(cand)
        for i, (_, img, gt) in enumerate(cand[: CAP[split]]):
            p = os.path.join(out, "crops", f"{split}_{i:05d}.png")
            img.convert("RGB").save(p)
            rows.append({"id": f"{ds_name}_{split}_{i:05d}", "split": split, "gt": gt, "path": p})
    with open(os.path.join(out, "manifest.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(ds_name, {s: sum(r["split"] == s for r in rows) for s in CAP})


def ok(img, gt):
    return gt.strip() and img.size[0] >= 6 and img.size[1] >= 8


def crop_pad(page, x0, y0, x1, y1, pad=2):
    W, H = page.size
    box = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
    if box[2] <= box[0] or box[3] <= box[1]:
        return page.crop((0, 0, 1, 1))  # rejected by ok() size check
    return page.crop(box)


def funsd():
    ds = load_from_disk(f"{ROOT}/data_raw/nielsr_funsd/ds")
    items = []
    for split in ("train", "test"):
        for ex in ds[split]:
            page = ex["image"]
            W, H = page.size  # nielsr/funsd boxes are LayoutLM-normalised to 0..1000
            for w, b in zip(ex["words"], ex["bboxes"]):
                x0, y0, x1, y1 = b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000
                if x1 <= x0 or y1 <= y0:
                    continue
                c = crop_pad(page, x0, y0, x1, y1)
                if ok(c, w):
                    items.append((split, c, w.strip()))
    save("FUNSD", items)


def cord():
    ds = load_from_disk(f"{ROOT}/data_raw/naver-clova-ix_cord-v2/ds")
    items = []
    for hf_split, split in (("train", "train"), ("validation", "train"), ("test", "test")):
        for ex in ds[hf_split]:
            page = ex["image"]
            gt = json.loads(ex["ground_truth"])
            for line in gt.get("valid_line", []):
                for w in line["words"]:
                    q = w["quad"]
                    xs, ys = [q["x1"], q["x2"], q["x3"], q["x4"]], [q["y1"], q["y2"], q["y3"], q["y4"]]
                    c = crop_pad(page, min(xs), min(ys), max(xs), max(ys))
                    if ok(c, w["text"]):
                        items.append((split, c, w["text"].strip()))
    save("CORD", items)


def simple(name, raw, splitmap):
    ds = load_from_disk(f"{ROOT}/data_raw/{raw}/ds")
    items = []
    for hf_split, split in splitmap:
        for ex in ds[hf_split]:
            if ok(ex["image"], ex["text"]):
                items.append((split, ex["image"], ex["text"].strip()))
    save(name, items)


def sroie_stream():
    """SROIE's bulk save hangs; stream with a seeded shuffle buffer instead."""
    from datasets import load_dataset
    items = []
    for split in ("train", "test"):
        ds = load_dataset("priyank-m/SROIE_2019_text_recognition", split=split, streaming=True)
        for ex in ds.shuffle(seed=0, buffer_size=20000).take(int(CAP[split] * 1.1)):
            if ok(ex["image"], ex["text"]):
                items.append((split, ex["image"], ex["text"].strip()))
    save("SROIE", items)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "FUNSD":
        funsd()
    elif which == "CORD":
        cord()
    elif which == "SROIE":
        sroie_stream()
    elif which == "IAM":
        simple("IAM", "priyank-m_IAM_words_text_recognition", [("train", "train"), ("test", "test")])
    elif which == "IIIT5K":
        simple("IIIT5K", "MiXaiLL76_IIIT5K_OCR", [("train", "train"), ("test", "test")])

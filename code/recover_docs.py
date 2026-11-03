"""Recover the source document of every FUNSD / CORD crop by replaying extract_crops.py's seeded
selection (same item order, same random.Random(0) shuffles) without writing images. The replay is
verified against the saved manifest (ground-truth strings and crop sizes must match one to one).
Output: data/<DS>/doc_ids.json  {crop id: document id}"""
import os
import json, random, sys

import numpy as np
from datasets import load_from_disk
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_crops as X

ROOT = X.ROOT


def items_funsd():
    ds = load_from_disk(f"{ROOT}/data_raw/nielsr_funsd/ds")
    out = []
    for split in ("train", "test"):
        for ex in ds[split]:
            page = ex["image"]
            W, H = page.size
            for w, b in zip(ex["words"], ex["bboxes"]):
                x0, y0, x1, y1 = b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000
                if x1 <= x0 or y1 <= y0:
                    continue
                c = X.crop_pad(page, x0, y0, x1, y1)
                if X.ok(c, w):
                    out.append((split, c.size, w.strip(), f"{split}:{ex['id']}"))
    return out


def items_cord():
    ds = load_from_disk(f"{ROOT}/data_raw/naver-clova-ix_cord-v2/ds")
    out = []
    for hf_split, split in (("train", "train"), ("validation", "train"), ("test", "test")):
        for n, ex in enumerate(ds[hf_split]):
            page = ex["image"]
            gt = json.loads(ex["ground_truth"])
            for line in gt.get("valid_line", []):
                for w in line["words"]:
                    q = w["quad"]
                    xs, ys = [q["x1"], q["x2"], q["x3"], q["x4"]], [q["y1"], q["y2"], q["y3"], q["y4"]]
                    c = X.crop_pad(page, min(xs), min(ys), max(xs), max(ys))
                    if X.ok(c, w["text"]):
                        out.append((split, c.size, w["text"].strip(), f"{hf_split}:{n}"))
    return out


def replay(name, items):
    rng = random.Random(0)  # extract_crops.py uses a module-level Random(0), one process per dataset
    man = [json.loads(l) for l in open(f"{ROOT}/data/{name}/manifest.jsonl")]
    doc, bad = {}, 0
    j = 0
    for split in ("train", "test"):
        cand = [it for it in items if it[0] == split]
        rng.shuffle(cand)
        for i, (_, size, gt, d) in enumerate(cand[: X.CAP[split]]):
            r = man[j]; j += 1
            assert r["id"] == f"{name}_{split}_{i:05d}"
            if r["gt"] != gt or Image.open(r["path"]).size != size:
                bad += 1
            doc[r["id"]] = d
    print(name, "crops", len(doc), "mismatches", bad, "documents", len(set(doc.values())))
    assert bad == 0, "replay does not reproduce the manifest"
    json.dump(doc, open(f"{ROOT}/data/{name}/doc_ids.json", "w"))


if __name__ == "__main__":
    replay("FUNSD", items_funsd())
    replay("CORD", items_cord())

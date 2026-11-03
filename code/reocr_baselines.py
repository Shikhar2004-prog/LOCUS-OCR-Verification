"""Top-venue re-recognition baselines: PARSeq (ECCV'22) and Florence-2 (CVPR'24).

For every TEST word: an independent strong recogniser re-reads the crop; the claim is
accepted iff the readings agree. PARSeq also yields its own sequence confidence, giving a
graded score: agree -> conf, disagree -> -conf.
Output: results/reocr_<DS>.json {id: {"parseq": str, "parseq_conf": float, "florence": str}}
usage: PYTHONPATH=pydeps python reocr_baselines.py <DS> [parseq|florence]
"""
import json, os, sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MAIN_ENGINE, ROOT


def test_rows(ds, all_splits=False):
    man = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/manifest.jsonl"))}
    hyp = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/hyp_{MAIN_ENGINE[ds]}.jsonl")]
    return [man[h["id"]] for h in hyp if all_splits or man[h["id"]]["split"] == "test"]


@torch.no_grad()
def parseq(rows):
    import torchvision.transforms as T
    m = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).eval().cuda()
    tf = T.Compose([T.Resize((32, 128), T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])
    out = {}
    for s in range(0, len(rows), 256):
        b = rows[s:s + 256]
        x = torch.stack([tf(Image.open(r["path"]).convert("RGB")) for r in b]).cuda()
        probs = m(x).softmax(-1)
        preds, conf = m.tokenizer.decode(probs)
        for r, p, c in zip(b, preds, conf):
            out[r["id"]] = {"parseq": p, "parseq_conf": float(c.prod())}
    return out


@torch.no_grad()
def florence(rows):
    from transformers import AutoModelForCausalLM, AutoProcessor
    name = "microsoft/Florence-2-base"
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="eager").eval().cuda()
    out = {}
    for s in range(0, len(rows), 16):
        b = rows[s:s + 16]
        imgs = [Image.open(r["path"]).convert("RGB") for r in b]
        inp = proc(text=["<OCR>"] * len(b), images=imgs, return_tensors="pt").to("cuda", torch.float16)
        g = m.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=48,
                       num_beams=1, do_sample=False, use_cache=False)
        txt = proc.batch_decode(g, skip_special_tokens=True)
        for r, t in zip(b, txt):
            out[r["id"]] = {"florence": t.strip()}
    return out


if __name__ == "__main__":
    ds, which = sys.argv[1], sys.argv[2]
    if which == "parseq_all":  # train+test readings: label-free agreement evidence for LOCUS
        json.dump(parseq(test_rows(ds, True)), open(f"{ROOT}/data/{ds}/parseq_all.json", "w"))
        sys.exit(0)
    path = f"{ROOT}/results/reocr_{ds}.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    new = parseq(test_rows(ds)) if which == "parseq" else florence(test_rows(ds))
    for k, v in new.items():
        res.setdefault(k, {}).update(v)
    json.dump(res, open(path, "w"))
    print(ds, which, len(new))

"""Modern baseline: a 7B vision-language model as zero-shot OCR verifier.

score = P("yes") / (P("yes") + P("no")) for the first answer token to:
  'Does the image show exactly the text "<claim>" (same characters, case and punctuation)?
   Answer yes or no.'
Writes results/vlm_<DS>.json: {word id: score} for every test word.
"""
import os
import json, sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MAIN_ENGINE, ROOT

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


@torch.no_grad()
def main(ds):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=32 * 28 * 28, max_pixels=256 * 28 * 28)
    proc.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    yes = proc.tokenizer.encode("yes", add_special_tokens=False)[0]
    no = proc.tokenizer.encode("no", add_special_tokens=False)[0]
    Yes = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    No = proc.tokenizer.encode("No", add_special_tokens=False)[0]
    man = {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/manifest.jsonl"))}
    hyp = [json.loads(l) for l in open(f"{ROOT}/data/{ds}/hyp_{MAIN_ENGINE[ds]}.jsonl")]
    test = [{"id": h["id"], "claim": h["hyp"] if h["hyp"].strip() else "?"} for h in hyp
            if man[h["id"]]["split"] == "test"]
    scores, B = [], 32
    for s in range(0, len(test), B):
        batch = test[s:s + B]
        texts, imgs = [], []
        for f in batch:
            img = Image.open(man[f["id"]]["path"]).convert("RGB")
            if img.size[1] < 56:  # tiny crops: upscale so the VLM sees legible glyphs
                r = 56 / img.size[1]
                img = img.resize((max(28, int(img.size[0] * r)), 56), Image.BICUBIC)
            claim = f["claim"].replace('"', "'")
            msg = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f'Does the image show exactly the text "{claim}" (same characters, '
                                         f'case and punctuation)? Answer yes or no.'}]}]
            texts.append(proc.apply_chat_template(msg, add_generation_prompt=True))
            imgs.append(img)
        inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
        logits = model(**inp).logits[:, -1].float()
        lp = torch.log_softmax(logits, -1)
        py = torch.logsumexp(torch.stack([lp[:, yes], lp[:, Yes]]), 0)
        pn = torch.logsumexp(torch.stack([lp[:, no], lp[:, No]]), 0)
        scores += torch.sigmoid(py - pn).cpu().tolist()
        if s % (B * 25) == 0:
            print(f"  {s}/{len(test)}", flush=True)
    json.dump({f["id"]: sc for f, sc in zip(test, scores)}, open(f"{ROOT}/results/vlm_{ds}.json", "w"))
    print(ds, len(scores))


if __name__ == "__main__":
    main(sys.argv[1])

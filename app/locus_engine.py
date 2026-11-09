"""LOCUS inference engine used by the demo application.

Pipeline for a page image:
  Tesseract 5 reads the page (word boxes + claims + engine confidence)
  -> every word crop and its claim go through the paper's evidence extractor
     (counterfactual string graph, render-normalised likelihood ratios from frozen TrOCR-base and PARSeq)
  -> the label-free pretrained LOCUS GNN gives P_opt over the graph and a repair proposal
  -> the few-shot fusion head (k=25 labels per class) gives P(claim correct).
The feature extraction and graph construction are imported unchanged from code/.
"""
import os, sys
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
sys.path.insert(0, CODE)
os.environ.setdefault("LOCUS_ROOT", os.path.abspath(os.path.join(HERE, "..")))
if not os.environ.get("TESSDATA_PREFIX"):  # conda's tesseract sets this on activation; venvs do not
    for _d in (os.path.join(sys.prefix, "share", "tessdata"), os.path.join(sys.base_prefix, "share", "tessdata"),
               "/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tesseract-ocr/4.00/tessdata", "/usr/local/share/tessdata"):
        if os.path.exists(os.path.join(_d, "eng.traineddata")):
            os.environ["TESSDATA_PREFIX"] = _d
            break

import locus as Lc  # noqa: E402
from common import norm  # noqa: E402
from features import FONTS, Verifier, levenshtein_ops, render  # noqa: E402
from locus2_features import KIND, ParseqVerifier, extract  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

KIND_NAME = {v: k for k, v in KIND.items()}
KIND_LABEL = {"claim": "claim", "conf": "confusable substitution", "case": "case flip", "del": "deletion",
              "vsub": "substitution toward a verifier reading", "ins": "insertion toward a verifier reading",
              "greedy": "TrOCR reading", "pgreedy": "PARSeq reading"}
DOMAINS = {"SROIE": "Receipts (SROIE-trained)", "CORD": "Receipts (CORD-trained)",
           "FUNSD": "Forms / scanned documents (FUNSD-trained)", "IAM": "Handwriting (IAM-trained)",
           "IIIT5K": "Scene text (IIIT5K-trained)"}


class _Norm:
    def __init__(self, m, s):
        self.m, self.s = m, s

    def __call__(self, x):
        return np.clip((x - self.m) / self.s, -5, 5)


@dataclass
class WordResult:
    claim: str
    p_correct: float
    flagged: bool
    repair: str
    box: tuple = None
    engine_conf: float = None
    trocr: str = ""
    parseq: str = ""
    parseq_conf: float = 0.0
    graph: list = field(default_factory=list)  # [(hyp, kind, rho_trocr, rho_parseq, p_opt)]


class LocusEngine:
    _verifiers = {}

    def __init__(self, ds="SROIE", device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        Lc.DEV = self.dev
        ck = torch.load(os.path.join(HERE, "models", f"locus_{ds}.pt"), map_location="cpu", weights_only=False)
        self.ck, self.ds, self.ci = ck, ds, ck["case_insensitive"]
        self.nz = _Norm(ck["norm_m"], ck["norm_s"])
        self.net = Lc.LOCUS().to(self.dev)
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.coef, self.b = ck["head_coef"], ck["head_intercept"]
        self.V1 = self._get(("trocr", ck["verifier"]), lambda: Verifier(ck["verifier"], self.dev))
        self.V2 = self._get(("parseq",), lambda: ParseqVerifier(self.dev))

    @classmethod
    def _get(cls, key, make):
        if key not in cls._verifiers:
            cls._verifiers[key] = make()
        return cls._verifiers[key]

    @torch.no_grad()
    def _parseq_full(self, imgs):
        x = torch.stack([self.V2.tf(i.convert("RGB")) for i in imgs]).to(self.dev)
        preds, conf = self.V2.tok.decode(self.V2.m(x).softmax(-1))
        return preds, [float(c.prod()) for c in conf]

    def _agree(self, claim, g1, rp, c):
        """Same label-free agreement evidence as locus.agree_vec, computed on the fly."""
        h, rp, g1 = norm(claim, self.ci), norm(rp, self.ci), norm(g1, self.ci)
        a = float(h == rp)
        L = max(1, len(h))
        return np.array([a, c if a else -c, c, float(h == g1), levenshtein_ops(h, rp)[2] / L,
                         levenshtein_ops(h, g1)[2] / L], dtype=np.float32)

    @torch.no_grad()
    def verify(self, crops, claims, threshold=0.5):
        if not crops:
            return []
        crops = [c.convert("RGB") for c in crops]
        feats = extract(self.V1, self.V2, crops, claims)
        pq, pc = [], []
        for s in range(0, len(crops), 256):
            a, b = self._parseq_full(crops[s:s + 256])
            pq += a; pc += b
        for f, p, c in zip(feats, pq, pc):
            f["agree"] = self._agree(f["h"], f["g"], p, c)
            f["parseq_full"], f["parseq_conf"] = p, c
        graphs = [Lc.build(f, self.nz, y=0, rep=-1) for f in feats]
        logits, nodes = [], []
        for b in DataLoader(graphs, batch_size=256):
            b = b.to(self.dev)
            lg, lp = self.net(b)
            logits += lg.cpu().tolist()
            ptr = b.ptr.cpu().tolist()
            lp = lp.exp().cpu().numpy()
            nodes += [lp[ptr[i]:ptr[i + 1]] for i in range(b.num_graphs)]
        out = []
        for f, lg, popt in zip(feats, logits, nodes):
            prior = np.array([Lc._logit(1 / (1 + np.exp(-lg))), Lc.zero_shot_score(f) / 10,
                              f["agree"][1], f["agree"][3]], dtype=np.float64)
            p = float(1 / (1 + np.exp(-(prior @ self.coef + self.b))))
            riv = popt.copy(); riv[0] = -1
            rep = f["hyps"][int(riv.argmax())] if len(riv) > 1 else f["h"]
            # display rule: the printed TrOCR verifier often reads in upper case; when the proposed repair
            # equals PARSeq's reading up to case, show PARSeq's casing (does not change P(correct))
            if rep.lower() == f["parseq_full"].lower():
                rep = f["parseq_full"]
            graph = [(v, KIND_NAME[int(k.argmax())], float(x[2]), float(x[13]), float(q))
                     for v, k, x, q in zip(f["hyps"], f["kind"], f["x"], popt)]
            out.append(WordResult(claim=f["h"], p_correct=p, flagged=p < threshold, repair=rep,
                                  trocr=f["g"], parseq=f["parseq_full"], parseq_conf=f["parseq_conf"],
                                  graph=sorted(graph, key=lambda g: -g[4])))
        return out


def tesseract_words(page, psm=3, min_h=8):
    """Word boxes, claims and confidences of Tesseract 5 on a full page."""
    import pytesseract
    d = pytesseract.image_to_data(page.convert("L"), config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    words = []
    for t, c, x, y, w, h in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"]):
        if t.strip() and float(c) >= 0 and h >= min_h:
            words.append({"claim": t.strip(), "conf": float(c) / 100, "box": (x, y, x + w, y + h)})
    return words


def crop_word(page, box, pad=0.15):
    x0, y0, x1, y1 = box
    p = int(round(pad * (y1 - y0))) + 2
    W, H = page.size
    return page.crop((max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p))).convert("RGB")


def verify_page(engine, page, threshold=0.5, psm=3):
    words = tesseract_words(page, psm)
    res = engine.verify([crop_word(page, w["box"]) for w in words], [w["claim"] for w in words], threshold)
    for r, w in zip(res, words):
        r.box, r.engine_conf = w["box"], w["conf"]
    return res


def render_hypothesis(text, h=48):
    return render(text, h, FONTS[0])

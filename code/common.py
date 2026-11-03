"""Shared data loading, graph construction, metrics."""
import os
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.data import Data

ROOT = os.environ.get("LOCUS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# label-normalisation mode per dataset (False = exact string match):
#   IIIT5K: standard scene-text protocol, case-insensitive alphanumeric (ground truth omits punctuation)
#   SROIE:  case-insensitive (ground truth is entirely upper-case, so it carries no case information)
CASE_INSENSITIVE = {"IIIT5K": "alnum", "SROIE": "case"}
MAIN_ENGINE = {"FUNSD": "tess", "CORD": "tess", "SROIE": "tess", "IIIT5K": "tess",
               "IAM": "trocr", "OCR10K": "given"}
REL_CHAIN, REL_SKIP, REL_GLOBAL = 0, 1, 2

# column groups of the node (18) / global (15) evidence, used by feature ablations
NODE_GROUPS = {"lik": [0, 1], "render": [2, 3], "cf": [4, 5, 6, 7, 8, 9], "align": [10, 11, 12],
               "cls": [13, 14, 15, 16, 17]}
GLOB_GROUPS = {"lik": [0, 9, 10, 13], "render": [1, 2, 11, 14], "cf": [12],
               "align": [3, 4, 5, 6, 8], "len": [7]}


def norm(s, ci):
    """Label normalisation. Documents/handwriting: exact string (whitespace-normalised).
    Scene text (ci=True): the standard STR protocol -- case-insensitive, alphanumeric only."""
    s = " ".join(s.split())
    if ci == "case":
        return s.lower()
    return "".join(c for c in s.lower() if c.isalnum()) if ci else s


def label_of(f, ds):
    """1 = Correct (claim matches ground truth), 0 = Incorrect."""
    if "label" in f:
        return int(f["label"])
    if "pseudo_label" in f:
        return int(f["pseudo_label"])
    ci = CASE_INSENSITIVE.get(ds, False)
    return int(norm(f["claim"], ci) == norm(f["gt"], ci))


def load_feats(ds, engine):
    feats = torch.load(f"{ROOT}/data/{ds}/feat_{engine}.pt", weights_only=False)
    for f in feats:
        f["y"] = label_of(f, ds)
    return feats


def load_hyp(ds, engine):
    return {r["id"]: r for r in map(json.loads, open(f"{ROOT}/data/{ds}/hyp_{engine}.jsonl"))}


class Normaliser:
    """z-score from an UNLABELED pool (no labels used), clipped."""

    def __init__(self, feats):
        N = np.concatenate([f["node"] for f in feats])
        G = np.stack([f["glob"] for f in feats])
        self.nm, self.ns = N.mean(0), N.std(0) + 1e-6
        self.gm, self.gs = G.mean(0), G.std(0) + 1e-6
        # do not standardise one-hot columns
        self.nm[10:], self.ns[10:] = 0, 1

    def node(self, x):
        return np.clip((x - self.nm) / self.ns, -5, 5)

    def glob(self, g):
        return np.clip((g - self.gm) / self.gs, -5, 5)


def build_graph(f, nz, drop_node=(), drop_glob=(), edges=("chain", "skip", "global")):
    x = nz.node(f["node"]).astype(np.float32)
    g = nz.glob(f["glob"]).astype(np.float32)
    if drop_node:
        x[:, list(drop_node)] = 0
    if drop_glob:
        g[list(drop_glob)] = 0
    L = len(x)
    pos = np.stack([np.arange(L) / max(1, L - 1), np.full(L, min(L, 40) / 40)], 1).astype(np.float32)
    x = np.concatenate([x, pos], 1)
    src, dst, rel = [], [], []

    def add(a, b, r):
        src.extend([a, b]); dst.extend([b, a]); rel.extend([r, r])

    if "chain" in edges:
        for t in range(L - 1):
            add(t, t + 1, REL_CHAIN)
    if "skip" in edges:
        for t in range(L - 2):
            add(t, t + 2, REL_SKIP)
    if "global" in edges:  # global hypothesis node has index L
        for t in range(L):
            add(t, L, REL_GLOBAL)
    for t in range(L + 1):  # self loops (relation-less)
        src.append(t); dst.append(t); rel.append(3)
    ei = torch.tensor([src, dst], dtype=torch.long)
    ea = torch.nn.functional.one_hot(torch.tensor(rel), 4).float()
    is_char = torch.cat([torch.ones(L, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)])
    xx = torch.zeros(L + 1, x.shape[1])
    xx[:L] = torch.from_numpy(x)
    return Data(x=xx, edge_index=ei, edge_attr=ea, is_char=is_char,
                gfeat=torch.from_numpy(g).unsqueeze(0), y=torch.tensor([f["y"]]))


def pooled_vector(f, nz):
    """Flat word vector (graph-free baselines): global evidence + mean/min/max of nodes."""
    x = nz.node(f["node"])
    return np.concatenate([nz.glob(f["glob"]), x.mean(0), x.min(0), x.max(0)]).astype(np.float32)


# ---------------------------------------------------------------- metrics
def best_threshold(scores, y):
    """Threshold maximising macro-F1 on the (labelled support) set."""
    cand = np.unique(scores)
    if len(cand) > 200:
        cand = np.quantile(scores, np.linspace(0, 1, 201))
    best, bt = -1, 0.5
    for t in cand:
        m = f1_score(y, (scores >= t).astype(int), average="macro", zero_division=0)
        if m > best:
            best, bt = m, t
    return bt


def coverage_at_precision(p_correct, y, target=0.99):
    """Fraction of words that can be auto-accepted (highest P(correct) first) while
    the accepted set keeps precision >= target. The rest go to human review."""
    order = np.argsort(-p_correct)
    ys = y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.where(prec >= target)[0]
    return float((ok.max() + 1) / len(y)) if len(ok) else 0.0


def aurc(p_correct, y):
    """Area under the risk-coverage curve (lower is better)."""
    order = np.argsort(-p_correct)
    err = 1 - y[order]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean())


def metrics(p_correct, y, thr=0.5):
    y = np.asarray(y)
    p = np.asarray(p_correct, dtype=np.float64)
    pred = (p >= thr).astype(int)
    out = {"macro_f1": f1_score(y, pred, average="macro", zero_division=0),
           "f1_incorrect": f1_score(1 - y, 1 - pred, zero_division=0),
           "acc": float((pred == y).mean())}
    if len(np.unique(y)) > 1 and len(np.unique(p)) > 1:
        out["auroc"] = roc_auc_score(y, p)
        out["auprc_err"] = average_precision_score(1 - y, -p)
    else:
        out["auroc"], out["auprc_err"] = 0.5, float((1 - y).mean())
    out["cov@99"] = coverage_at_precision(p, y, 0.99)
    out["cov@95"] = coverage_at_precision(p, y, 0.95)
    out["aurc"] = aurc(p, y)
    return {k: float(v) for k, v in out.items()}

"""LOCUS: Local Optimality over Counterfactual String graphs.

usage: python locus.py <experiment> <dataset>
  main     LOCUS vs graph-free baselines on the SAME edit-graph evidence, k grid, 5 seeds
  ablate   component ablations at k=25
  xengine  train on Tesseract claims, test on TrOCR claims (and reverse)
  xdataset fully supervised on one dataset, applied to the others
Results: results/locus_<experiment>.jsonl
"""
import copy, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, SAGEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax as seg_softmax

sys.path.insert(0, os.path.dirname(__file__))
from common import CASE_INSENSITIVE, MAIN_ENGINE, ROOT, load_hyp, metrics, norm
from run import best_threshold, sample_support

DEV = "cuda"
SEEDS = [0, 1, 2, 3, 4]
KS = [5, 10, 25, 50, 100]
FEAT = "locus2"  # dual-verifier edit graphs (locus2_features.py)
NKIND = 8         # claim, conf, case, del, vsub, ins, TrOCR reading, PARSeq reading
EDGE_DIM = NKIND + 3  # edit kind of the neighbour, + sibling / reading-path / self flags
# column groups of the 22-d hypothesis evidence: [TrOCR 0..10 | PARSeq 11..21], each
# s(v|x), s(v|R(v)), rho, mean lp, min lp, min rho, mean rho, latent mean, latent max, len, ntok
_g = lambda cols: cols + [c + 11 for c in cols]
COL_LIK = _g([0, 3, 4])
COL_REN = _g([1, 2, 5, 6])
COL_LAT = _g([7, 8])
COL_TROCR = list(range(0, 11))
COL_PARSEQ = list(range(11, 22))
DELTA_COLS = [0, 2, 3, 9, 11, 13, 14]
DELTA_SCALE = np.array([5, 5, 1, 3, 5, 5, 1])


def log_row(exp, row):
    with open(f"{ROOT}/results/locus_{exp}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def label_of(f, ds):
    if "pseudo_label" in f:
        return int(f["pseudo_label"])
    ci = CASE_INSENSITIVE.get(ds, False)
    return int(norm(f["claim"], ci) == norm(f["gt"], ci))


def repair_target(f, ds):
    """Index of the node equal to the true string (-1 if not in the graph).
    Pretraining corruptions: the pre-corruption string is pseudo-truth (label-free)."""
    ci = CASE_INSENSITIVE.get(ds, False)
    if "pseudo_label" in f:
        tgt = f.get("orig")
        if f["pseudo_label"] == 1:
            return 0
        if tgt is None:
            return -1
    else:
        tgt = f["gt"]
    for i, h in enumerate(f["hyps"]):
        if norm(h, ci) == norm(tgt, ci):
            return i
    return -1


class Norm:
    def __init__(self, feats):
        X = np.concatenate([f["x"] for f in feats])
        self.m, self.s = X.mean(0), X.std(0) + 1e-6

    def __call__(self, x):
        return np.clip((x - self.m) / self.s, -5, 5)


_PQ = {}


def agree_vec(f, ds):
    """Label-free agreement evidence of the claim: with PARSeq's full (refined) reading and with
    TrOCR's reading -- the signals used by re-recognition baselines, given to LOCUS as inputs."""
    from features import levenshtein_ops
    if ds not in _PQ:
        _PQ[ds] = json.load(open(f"{ROOT}/data/{ds}/parseq_all.json"))
    ci = CASE_INSENSITIVE.get(ds, False)
    r = _PQ[ds].get(f["id"], {"parseq": "", "parseq_conf": 0.0})
    h, rp, g1 = norm(f["claim"], ci), norm(r["parseq"], ci), norm(f.get("g", ""), ci)
    a, c = float(h == rp), float(r["parseq_conf"])
    L = max(1, len(h))
    return np.array([a, c if a else -c, c, float(h == g1), levenshtein_ops(h, rp)[2] / L,
                     levenshtein_ops(h, g1)[2] / L], dtype=np.float32)


def build(f, nz, drop_cols=(), edges=("star", "sib", "path"), y=None, rep=-1):
    x = f["x"].copy()
    d = x - x[0:1]  # deltas to the claim (raw scale, then squashed)
    delta = d[:, DELTA_COLS] / DELTA_SCALE
    xn = nz(x)
    if drop_cols:
        xn[:, list(drop_cols)] = 0
        delta[:, [i for i, c in enumerate(DELTA_COLS) if c in set(drop_cols)]] = 0
    n = len(x)
    if getattr(build, "no_delta", False):  # MP analysis: remove precomputed claim-relative features
        delta = np.zeros_like(delta)
    A = np.zeros((n, N_AGREE), dtype=np.float32)
    if "agree" in f and not getattr(build, "no_agree", False):
        A[0] = f["agree"]  # claim node only
    feat = np.concatenate([xn, np.tanh(delta), f["kind"], f["pos"], A], 1).astype(np.float32)
    src, dst, ea = [], [], []

    def add(a, b, kind_vec, flags):
        e = np.concatenate([kind_vec, flags])
        src.extend([a, b]); dst.extend([b, a]); ea.extend([e, e])

    zero = np.zeros(NKIND, np.float32)
    if "star" in edges:
        for j in range(1, n):
            add(0, j, f["kind"][j], np.array([0, 0, 0], np.float32))
    if "sib" in edges:  # competing explanations of the same glyph position
        pos = f["edit_pos"]
        for a in range(1, n):
            for b in range(a + 1, n):
                if pos[a] >= 0 and pos[a] == pos[b]:
                    add(a, b, zero, np.array([1, 0, 0], np.float32))
    if "path" in edges:  # verifier's reading <-> the single edits that lead toward it
        for gi in [j for j in range(n) if f["kind"][j, 6] == 1 or f["kind"][j, 7] == 1]:
            for j in range(1, n):
                if j != gi and (f["kind"][j, 4] == 1 or f["kind"][j, 5] == 1):
                    add(gi, j, zero, np.array([0, 1, 0], np.float32))
    for j in range(n):
        src.append(j); dst.append(j); ea.append(np.r_[zero, [0, 0, 1]].astype(np.float32))
    is_claim = torch.zeros(n, dtype=torch.bool)
    is_claim[0] = True
    return Data(x=torch.from_numpy(feat), edge_index=torch.tensor([src, dst], dtype=torch.long),
                edge_attr=torch.tensor(np.stack(ea), dtype=torch.float), is_claim=is_claim,
                y=torch.tensor([y if y is not None else f["y"]]), rep=torch.tensor([rep]))


N_AGREE = 6
IN_DIM = 22 + len(DELTA_COLS) + NKIND + 1 + N_AGREE


class LOCUS(nn.Module):
    """Relation-aware GNN over the counterfactual edit graph.

    verdict logit = MLP([h_claim, attn-pool(rivals), max-pool(rivals)])
                    + beta * log P_opt(claim)
    where P_opt(claim) = softmax over nodes of a learned plausibility q_v, i.e. the
    probability that the claim is the local optimum of its edit neighbourhood.
    The same q_v (softmax) is the repair head: argmax over rivals = proposed correction.
    """

    def __init__(self, conv="gatv2", d=64, layers=3, dropout=0.2, use_opt=True):
        super().__init__()
        self.conv_kind, self.use_opt = conv, use_opt
        self.inp = nn.Sequential(nn.Linear(IN_DIM, d), nn.ReLU())
        mk = {"gatv2": lambda: GATv2Conv(d, d // 4, heads=4, edge_dim=EDGE_DIM, add_self_loops=False),
              "gcn": lambda: GCNConv(d, d, add_self_loops=False), "sage": lambda: SAGEConv(d, d),
              "gin": lambda: GINConv(nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))),
              "none": None}[conv]
        self.convs = nn.ModuleList([mk() for _ in range(layers)]) if mk else nn.ModuleList()
        self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(len(self.convs))])
        self.drop = dropout
        self.att = nn.Linear(d, 1)
        self.q = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, b):
        h = self.inp(b.x)
        for conv, ln in zip(self.convs, self.lns):
            m = conv(h, b.edge_index, b.edge_attr) if self.conv_kind == "gatv2" else conv(h, b.edge_index)
            h = ln(h + F.dropout(F.relu(m), self.drop, self.training))
        q = self.q(h).squeeze(-1)
        logp = torch.log(seg_softmax(q, b.batch) + 1e-9)
        hc = h[b.is_claim]
        riv = ~b.is_claim
        br = b.batch[riv]
        a = seg_softmax(self.att(h[riv]).squeeze(-1), br)
        za = torch.zeros_like(hc).index_add_(0, br, h[riv] * a.unsqueeze(-1))
        zm = torch.full_like(hc, -1e4).scatter_reduce(0, br.unsqueeze(-1).expand(-1, hc.shape[1]), h[riv], "amax")
        zm = torch.where(zm < -1e3, torch.zeros_like(zm), zm)  # claims with no rivals
        z = torch.cat([hc, za, zm], -1)
        logit = self.head(z).squeeze(-1)
        if self.use_opt:
            logit = logit + self.beta * logp[b.is_claim]
        if getattr(self, "_embed", False):  # representation for the few-shot linear probe
            return torch.cat([z, logp[b.is_claim].unsqueeze(-1), logit.unsqueeze(-1)], -1), logp
        return logit, logp


class DeepSetsRivals(nn.Module):
    """Graph-free ablation over the same nodes: claim + permutation-invariant rival pooling."""

    def __init__(self, d=64, dropout=0.2):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(IN_DIM, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, b):
        h = self.phi(b.x)
        riv = ~b.is_claim
        hc = h[b.is_claim]
        zr = torch.zeros_like(hc).index_add_(0, b.batch[riv], h[riv])
        cnt = torch.bincount(b.batch[riv], minlength=hc.shape[0]).clamp(min=1).unsqueeze(-1)
        zm = torch.zeros_like(hc).scatter_reduce(0, b.batch[riv].unsqueeze(-1).expand(-1, hc.shape[1]), h[riv], "amax")
        logit = self.head(torch.cat([hc, zr / cnt, zm], -1)).squeeze(-1)
        return logit, torch.zeros(b.x.shape[0], device=b.x.device)


def train(make, graphs, seed, steps=300, lr=1e-3, init=None, epochs=None, val=None, lam=0.5):
    torch.manual_seed(seed); np.random.seed(seed)
    m = make().to(DEV)
    if init is not None:
        m.load_state_dict(init)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    ys = np.array([int(g.y) for g in graphs])
    pw = torch.tensor(max(1, (ys == 0).sum()) / max(1, (ys == 1).sum()), device=DEV, dtype=torch.float)
    bs = int(min(64, max(4, len(graphs) // 2)))
    ld = DataLoader(graphs, batch_size=bs, shuffle=True, drop_last=len(graphs) > bs)
    step, best, best_state, bad = 0, -1, None, 0
    for ep in range(epochs or 10 ** 9):
        m.train()
        for b in ld:
            b = b.to(DEV)
            logit, logp = m(b)
            loss = F.binary_cross_entropy_with_logits(logit, b.y.float(), pos_weight=pw)
            # repair / local-optimality supervision where the true node is known
            ok = b.rep >= 0
            if lam > 0 and ok.any():
                ptr = b.ptr[:-1]
                idx = (ptr + b.rep.clamp(min=0))[ok]
                loss = loss - lam * logp[idx].mean()
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if not epochs and step >= steps:
                return m
        if val is not None:
            mf = metrics(predict(m, val)[0], np.array([int(g.y) for g in val]))["macro_f1"]
            if mf > best:
                best, best_state, bad = mf, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
                if bad >= 8:
                    break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m


@torch.no_grad()
def predict(m, graphs):
    """Returns P(claim correct) and, per graph, the index of the best rival (repair)."""
    m.eval()
    P, R = [], []
    for b in DataLoader(graphs, batch_size=512):
        b = b.to(DEV)
        logit, logp = m(b)
        P.append(torch.sigmoid(logit).cpu())
        lp = logp.clone()
        lp[b.is_claim] = -1e9
        for gi in range(b.num_graphs):
            s, e = int(b.ptr[gi]), int(b.ptr[gi + 1])
            R.append(int(lp[s:e].argmax()) if e - s > 1 else 0)
    return torch.cat(P).numpy(), np.array(R)


@torch.no_grad()
def embed(m, graphs):
    m.eval()
    m._embed = True
    out = [m(b.to(DEV))[0].cpu() for b in DataLoader(graphs, batch_size=512)]
    m._embed = False
    return torch.cat(out).numpy()


def probe_fit_predict(Ztr, Zte, sup, ys, C=1.0):
    """Few-shot head: standardise with UNLABELED pool statistics, L2 logistic regression."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    lr = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    lr.fit((Ztr[sup] - mu) / sd, ys)
    return lr.predict_proba((Zte - mu) / sd)[:, 1]


_EMB = {}


def locus_embeddings(B, s):
    """Label-free representation of every train/test word from the pretrained LOCUS encoder."""
    key = (B.ds, B.engine, s)
    if key not in _EMB:
        m = L()().to(DEV)
        m.load_state_dict(pretrained(B, s, L()))
        _EMB[key] = (embed(m, B.gtr), embed(m, B.gte))
    return _EMB[key]


def full_features(B, s, use_gnn=True, use_char=True):
    Etr, Ete = locus_embeddings(B, s) if use_gnn else (np.zeros((len(B.train), 0)), np.zeros((len(B.test), 0)))
    parts_tr, parts_te = [Etr], [Ete]
    if use_char and B.Ctr is not None:
        parts_tr.append(B.Ctr); parts_te.append(B.Cte)
    return np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1)


def flat_vector(f, nz):
    """Strongest graph-free summary of the same edit-graph evidence."""
    x = nz(f["x"])
    raw = f["x"]
    c = x[0]
    if len(x) > 1:
        riv = x[1:]
        ex = []
        for rc, sc in ((2, 0), (13, 11)):  # per verifier: rho and likelihood margins vs rivals
            dr, ds_ = raw[1:, rc] - raw[0, rc], raw[1:, sc] - raw[0, sc]
            ex += [dr.max(), ds_.max(), (dr > 0).mean(), (ds_ > 0).mean()]
        s = np.r_[riv.max(0), riv.mean(0), ex, len(riv) / 28]
    else:
        s = np.r_[np.zeros(44), [-10, -10, 0, 0] * 2, 0]
    return np.r_[c, s].astype(np.float32)


def zero_shot_score(f):
    """Training-free local-optimality test: margin of the claim's render-normalised
    likelihood ratio over its best rival (rho(h) - max_v rho(v))."""
    r = f["x"][:, 2] + f["x"][:, 13]  # summed render-normalised ratio of both verifiers
    return float(r[0] - (r[1:].max() if len(r) > 1 else r[0] - 10))


class Bundle:
    def __init__(self, ds, engine=None):
        self.ds, self.engine = ds, engine or MAIN_ENGINE[ds]
        Fs = torch.load(f"{ROOT}/data/{ds}/{FEAT}_{self.engine}.pt", weights_only=False)
        for f in Fs:
            f["y"] = label_of(f, ds)
            f["agree"] = agree_vec(f, ds)
        self.train = [f for f in Fs if f["split"] == "train"]
        self.test = [f for f in Fs if f["split"] == "test"]
        self.nz = Norm(self.train)
        self.ytr = np.array([f["y"] for f in self.train])
        self.yte = np.array([f["y"] for f in self.test])
        # repair targets: only the binary label is used in training (claim node when
        # correct); the true string of incorrect words is used only for evaluation.
        self.gtr = [build(f, self.nz, rep=0 if f["y"] == 1 else -1) for f in self.train]
        self.gte = [build(f, self.nz, rep=repair_target(f, ds)) for f in self.test]
        self.Xtr = np.stack([np.r_[flat_vector(f, self.nz), f["agree"]] for f in self.train])
        self.Xte = np.stack([np.r_[flat_vector(f, self.nz), f["agree"]] for f in self.test])
        self.pre = None
        p = f"{ROOT}/data/{ds}/{FEAT}_pretrain.pt"
        if os.path.exists(p):
            P = torch.load(p, weights_only=False)
            for f in P:
                f["agree"] = agree_vec(f, ds)
            self.pre_raw = P
            self.pre = [build(f, self.nz, y=f["pseudo_label"], rep=pre_rep(f, ds)) for f in P]
        # character-level counterfactual evidence (same words), pooled to a word vector
        self.Ctr = self.Cte = None
        cp = f"{ROOT}/data/{ds}/feat_{self.engine}.pt"
        if os.path.exists(cp):
            from common import Normaliser, pooled_vector
            C = {f["id"]: f for f in torch.load(cp, weights_only=False)}
            if all(f["id"] in C for f in self.train + self.test):
                cn = Normaliser([C[f["id"]] for f in self.train])
                self.Ctr = np.stack([np.r_[pooled_vector(C[f["id"]], cn), f["agree"]] for f in self.train])
                self.Cte = np.stack([np.r_[pooled_vector(C[f["id"]], cn), f["agree"]] for f in self.test])
        self.conf_tr = self.conf_te = None
        if self.engine != "given":
            H = load_hyp(ds, self.engine)
            self.conf_tr = np.array([H[f["id"]]["conf"] for f in self.train])
            self.conf_te = np.array([H[f["id"]]["conf"] for f in self.test])
        print(f"[LOCUS {ds}/{self.engine}] train {len(self.train)} test {len(self.test)} "
              f"err {1 - self.yte.mean():.3f} pretrain {0 if self.pre is None else len(self.pre)} "
              f"truth-in-graph (test errors) {np.mean([g.rep.item() > 0 for g, y in zip(self.gte, self.yte) if y == 0]):.3f}",
              flush=True)


def pre_rep(f, ds):
    """Pretraining pairs come in (positive, corrupted) order with the same id; the
    positive's claim is the pseudo-truth for the corrupted one."""
    if f["pseudo_label"] == 1:
        return 0
    tgt = _PRE_POS.get(f["id"])
    if tgt is None:
        return -1
    for i, h in enumerate(f["hyps"]):
        if h == tgt:
            return i
    return -1


_PRE_POS = {}


def index_pretrain_truth(ds):
    p = f"{ROOT}/data/{ds}/{FEAT}_pretrain.pt"
    if os.path.exists(p):
        for f in torch.load(p, weights_only=False):
            if f["pseudo_label"] == 1:
                _PRE_POS[f["id"]] = f["claim"]


_PT = {}


def pretrained(B, seed, make, key="full", graphs=None):
    k = (B.ds, B.engine, seed, key)
    if k not in _PT:
        _PT[k] = copy.deepcopy(train(make, graphs or B.pre, seed, epochs=15).state_dict())
    return _PT[k]


def L(**kw):
    return lambda: LOCUS(**kw)


def repair_acc(R, B, P=None):
    """Top-1 repair accuracy on test words that are truly incorrect."""
    err = np.where(B.yte == 0)[0]
    hits = [R[i] == B.gte[i].rep.item() and B.gte[i].rep.item() > 0 for i in err]
    return float(np.mean(hits)) if len(err) else 0.0


def cv_probe(Ztr, Zte, sup, ys):
    """L2 logistic regression whose C is chosen by stratified CV on the labelled support only;
    standardisation uses UNLABELED pool statistics. Used for ours and for every LR baseline."""
    from sklearn.linear_model import LogisticRegressionCV
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    nmin = int(min((ys == 0).sum(), (ys == 1).sum()))
    X = (Ztr[sup] - mu) / sd
    if nmin >= 3:
        lr = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1.0], cv=min(5, nmin), max_iter=5000,
                                  class_weight="balanced", scoring="roc_auc").fit(X, ys)
    else:
        lr = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit(X, ys)
    return lr.predict_proba((Zte - mu) / sd)[:, 1]


def split_for(B, k, s):
    if k == "full":
        perm = np.random.RandomState(s).permutation(len(B.train))
        nv = int(0.15 * len(perm))
        return perm[nv:], dict(epochs=60, val=[B.gtr[i] for i in perm[:nv]])
    return sample_support(B.ytr, k, s), dict(steps=300)


_ZS = {}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def zero_shot_logits(B, s, gtr=None, gte=None, state=None, key="full", make=None):
    """Label-free prior: logit of the pretrained (never label-tuned) LOCUS network."""
    k = (B.ds, B.engine, s, key)
    if k not in _ZS:
        m = (make or L())().to(DEV)
        m.load_state_dict(state if state is not None else pretrained(B, s, L()))
        _ZS[k] = (_logit(predict(m, gtr or B.gtr)[0]), _logit(predict(m, gte or B.gte)[0]))
    return _ZS[k]


def anchored_head(prior_tr, prior_te, F_tr, F_te, sup, ys, anchor_scale=10.0):
    """Prior-anchored residual head. Label-free priors enter (scaled so the l2 penalty is
    negligible on them); the remaining evidence learns a shrunken residual correction. The
    penalty C is chosen by stratified CV on the labelled support only."""
    from sklearn.linear_model import LogisticRegressionCV
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6  # unlabeled-pool statistics
    Xtr = np.c_[prior_tr * anchor_scale, (F_tr - mu) / sd]
    Xte = np.c_[prior_te * anchor_scale, (F_te - mu) / sd]
    nmin = int(min((ys == 0).sum(), (ys == 1).sum()))
    if nmin >= 3:
        lr = LogisticRegressionCV(Cs=[0.003, 0.01, 0.03, 0.1, 0.3], cv=min(5, nmin), max_iter=5000,
                                  class_weight="balanced", scoring="roc_auc").fit(Xtr[sup], ys)
    else:
        lr = LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced").fit(Xtr[sup], ys)
    return lr.predict_proba(Xte)[:, 1]


def priors(B, s, zs=None, use_agree=True):
    """Label-free detectors fused by the head: pretrained LOCUS logit, the parameter-free
    rho-margin local-optimality rule, and (optionally) PARSeq agreement."""
    ztr, zte = zs if zs is not None else zero_shot_logits(B, s)
    rtr = np.array([zero_shot_score(f) for f in B.train]) / 10
    rte = np.array([zero_shot_score(f) for f in B.test]) / 10
    if use_agree:  # agreement with BOTH frozen verifiers' readings (PARSeq graded, TrOCR binary)
        Atr = np.stack([f["agree"][[1, 3]] for f in B.train]); Ate = np.stack([f["agree"][[1, 3]] for f in B.test])
        return np.c_[ztr, rtr, Atr], np.c_[zte, rte, Ate]
    return np.c_[ztr, rtr], np.c_[zte, rte]


def select_head(prior_tr, prior_te, F_tr, F_te, sup, ys, seed=0):
    """Choose, by repeated stratified CV on the labelled support only, between the low-capacity
    fusion of label-free priors and the prior-anchored residual head."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    nmin = int(min((ys == 0).sum(), (ys == 1).sum()))
    fuse = lambda tr_idx, te_idx: LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(
        prior_tr[tr_idx], ys_all[tr_idx]).predict_proba(prior_tr[te_idx])[:, 1]
    ys_all = np.zeros(len(prior_tr)); ys_all[sup] = ys
    if nmin < 5:
        lr = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(prior_tr[sup], ys)
        return lr.predict_proba(prior_te)[:, 1], "fusion"
    sc = {"fusion": [], "anchored": []}
    for tr, te in RepeatedStratifiedKFold(n_splits=min(5, nmin), n_repeats=3, random_state=seed).split(sup, ys):
        a, b = sup[tr], sup[te]
        if len(np.unique(ys[te])) < 2:
            continue
        sc["fusion"].append(roc_auc_score(ys[te], fuse(a, b)))
        sc["anchored"].append(roc_auc_score(ys[te], anchored_head(prior_tr, prior_tr[b], F_tr, F_tr[b], a, ys[tr])))
    choice = "anchored" if np.mean(sc["anchored"]) > np.mean(sc["fusion"]) else "fusion"
    if choice == "fusion":
        lr = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(prior_tr[sup], ys)
        return lr.predict_proba(prior_te)[:, 1], choice
    return anchored_head(prior_tr, prior_te, F_tr, F_te, sup, ys), choice


def locus_predict(B, sup, s, kw, gtr=None, gte=None, make=None, pre_state=None, emb=None, use_char=True,
                  use_agree=True, key="full"):
    """Full LOCUS few-shot predictor. Returns (final, fine-tuned GNN, un-anchored probe, repair idx).
    final = prior-anchored residual head: priors = [label-free LOCUS logit, PARSeq agreement score];
    residual evidence = [pretrained LOCUS embedding, character evidence]."""
    gtr, gte, make = gtr or B.gtr, gte or B.gte, make or L()
    init = pre_state if pre_state is not None else pretrained(B, s, L())
    m = train(make, [gtr[i] for i in sup], s, lr=3e-4, init=init, **kw)
    p_ft, rep = predict(m, gte)
    Etr, Ete = emb if emb is not None else locus_embeddings(B, s)
    if use_char and B.Ctr is not None:
        Etr, Ete = np.concatenate([Etr, B.Ctr], 1), np.concatenate([Ete, B.Cte], 1)
    ys = B.ytr[sup]
    p_pr = cv_probe(Etr, Ete, sup, ys)
    zs = zero_shot_logits(B, s, gtr, gte, init, key, make)
    Ptr, Pte = priors(B, s, zs, use_agree)
    p, choice = select_head(Ptr, Pte, Etr, Ete, np.asarray(sup), ys, s)
    locus_predict.last_choice = choice
    # repair proposals come from the label-free pretrained network (most reliable distribution)
    mz = L()().to(DEV) if make is None else make().to(DEV)
    mz.load_state_dict(init)
    rep = predict(mz, gte)[1]
    return p, p_ft, p_pr, rep


def run_main(ds):
    index_pretrain_truth(ds)
    B = Bundle(ds)
    zs = np.array([zero_shot_score(f) for f in B.test])
    ztr = np.array([zero_shot_score(f) for f in B.train])
    oracle = np.mean([g.rep.item() > 0 for g, y in zip(B.gte, B.yte) if y == 0])
    log_row("main", {"ds": ds, "method": "truth-in-graph", "k": 0, "seed": -1, "value": float(oracle)})
    log_row("main", {"ds": ds, "method": "LOCUS-rule (zero-shot)", "k": 0, "seed": -1, **metrics(zs, B.yte, thr=0.0)})
    store = {"ids": [f["id"] for f in B.test], "y": B.yte}
    P, RA = [], []
    for s in SEEDS:  # zero-shot: label-free pretraining only
        m = L()().to(DEV)
        m.load_state_dict(pretrained(B, s, L()))
        p, r = predict(m, B.gte)
        P.append(p); RA.append(repair_acc(r, B))
        log_row("main", {"ds": ds, "method": "LOCUS-ZeroShot", "k": 0, "seed": s, "repair": RA[-1], **metrics(p, B.yte)})
    log_row("main", {"ds": ds, "method": "LOCUS-ZeroShot", "k": 0, "seed": "ens", "repair": float(np.mean(RA)),
                     **metrics(np.mean(P, 0), B.yte)})
    store["zs_LOCUS"] = np.mean(P, 0)
    np.save(f"{ROOT}/results/repair_{ds}_k0.npy", r)
    for k in KS + ["full"]:
        preds = {}
        for s in SEEDS:
            sup, kw = split_for(B, k, s)
            ys = B.ytr[sup]
            row = {"ds": ds, "k": k, "seed": s}
            t = best_threshold(ztr[sup], ys)
            log_row("main", {**row, "method": "LOCUS-rule (thr)", **metrics(zs, B.yte, thr=t)})
            if B.conf_tr is not None:
                t = best_threshold(B.conf_tr[sup], ys)
                log_row("main", {**row, "method": "EngineConf", **metrics(B.conf_te, B.yte, thr=t)})
            def add(name, p, extra=None):
                preds.setdefault(name, []).append(p)
                log_row("main", {**row, "method": name, **(extra or {}), **metrics(p, B.yte)})
            add("Flat-LR", cv_probe(B.Xtr, B.Xte, sup, ys))
            if B.Ctr is not None:
                add("CharEv-LR", cv_probe(B.Ctr, B.Cte, sup, ys))
                add("Flat+CharEv-LR", cv_probe(np.c_[B.Xtr, B.Ctr], np.c_[B.Xte, B.Cte], sup, ys))
            gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced",
                                                min_samples_leaf=max(2, min(20, len(sup) // 10)), random_state=s)
            Xg_tr = np.c_[B.Xtr, B.Ctr] if B.Ctr is not None else B.Xtr
            Xg_te = np.c_[B.Xte, B.Cte] if B.Ctr is not None else B.Xte
            add("GBDT (all evidence)", gb.fit(Xg_tr[sup], ys).predict_proba(Xg_te)[:, 1])
            gs = [B.gtr[i] for i in sup]
            add("DeepSets-rivals", predict(train(lambda: DeepSetsRivals(), gs, s, lam=0, **kw), B.gte)[0])
            m = train(L(), gs, s, **kw)
            p, r = predict(m, B.gte)
            add("LOCUS-GNN (scratch)", p, {"repair": repair_acc(r, B)})
            p, p_ft, p_pr, r = locus_predict(B, sup, s, kw)
            add("LOCUS-GNN (pretrained, fine-tuned)", p_ft, {"repair": repair_acc(r, B)})
            add("LOCUS-probe (un-anchored)", p_pr)
            add("LOCUS", p, {"repair": repair_acc(r, B), "head": locus_predict.last_choice})
            Ptr, Pte = priors(B, s)
            add("Priors-only LR", cv_probe(Ptr, Pte, sup, ys))
        for name, Pk in preds.items():
            log_row("main", {"ds": ds, "k": k, "seed": "ens", "method": name, **metrics(np.mean(Pk, 0), B.yte)})
            store[f"{name}|{k}"] = np.stack(Pk)
        np.save(f"{ROOT}/results/repair_{ds}_k{k}.npy", r)
        print(f"  k={k} done", flush=True)
    np.save(f"{ROOT}/results/preds_{ds}.npy", store, allow_pickle=True)


def run_ablate(ds, k=25):
    """Component ablations of LOCUS at k=25 (5 seeds + ensemble). Each GNN variant is
    pretrained (label-free) with the same change, then fused exactly like the full model."""
    index_pretrain_truth(ds)
    B = Bundle(ds)
    V = {
        "LOCUS (full)": dict(),
        "+ CV-selected prior/anchored head": dict(only="head"),
        "+ linear probe on embedding": dict(only="probe"),
        "w/o verifier-agreement node features": dict(no_agree=True),
        "w/o edit-graph GNN (char evidence LR)": dict(only="char"),
        "w/o label-free pretraining": dict(nopre=True),
        "w/o local-optimality term": dict(make=L(use_opt=False)),
        "w/o repair loss": dict(lam=0),
        "w/o render-normalisation": dict(drop=COL_REN),
        "w/o latent render distance": dict(drop=COL_LAT),
        "likelihood only (no render)": dict(drop=COL_REN + COL_LAT),
        "TrOCR verifier only": dict(drop=COL_PARSEQ),
        "PARSeq verifier only": dict(drop=COL_TROCR),
        "star edges only": dict(edges=("star",)),
        "no message passing": dict(make=L(conv="none")),
        "GCN backbone": dict(make=L(conv="gcn")),
        "GraphSAGE backbone": dict(make=L(conv="sage")),
        "GIN backbone": dict(make=L(conv="gin")),
    }
    done = set()
    if os.path.exists(f"{ROOT}/results/locus_ablate.jsonl"):  # resume: skip finished variants
        for l in open(f"{ROOT}/results/locus_ablate.jsonl"):
            r = json.loads(l)
            if r["ds"] == ds and str(r["seed"]) == "ens":
                done.add(r["variant"])
    for name, v in V.items():
        if name in done:
            continue
        drop, edges, lam = v.get("drop", ()), v.get("edges", ("star", "sib", "path")), v.get("lam", 0.5)
        make = v.get("make", L())
        custom = bool(drop) or "edges" in v or v.get("no_agree", False)
        build.no_agree = v.get("no_agree", False)
        gtr = [build(f, B.nz, drop, edges, rep=0 if f["y"] == 1 else -1) for f in B.train] if custom else B.gtr
        gte = [build(f, B.nz, drop, edges, rep=repair_target(f, ds)) for f in B.test] if custom else B.gte
        pre = [build(f, B.nz, drop, edges, y=f["pseudo_label"], rep=pre_rep(f, ds)) for f in B.pre_raw] if custom else B.pre
        build.no_agree = False
        P, RA = [], []
        for s in SEEDS:
            sup, kw = split_for(B, k, s)
            ys = B.ytr[sup]
            if v.get("only") == "char":
                p, r = cv_probe(B.Ctr, B.Cte, sup, ys), None
            else:
                if v.get("nopre"):
                    st = copy.deepcopy(make().state_dict())  # random init "encoder"
                    torch.manual_seed(s)
                elif not (custom or "make" in v or "lam" in v):
                    st = pretrained(B, s, L())
                else:
                    st = copy.deepcopy(train(make, pre, s, epochs=15, lam=lam).state_dict())
                mm = make().to(DEV); mm.load_state_dict(st)
                emb = (embed(mm, gtr), embed(mm, gte))
                if v.get("nopre"):
                    m = train(make, [gtr[i] for i in sup], s, lr=1e-3, lam=lam, **kw)
                    p_ft, r = predict(m, gte)
                    Etr, Ete = emb
                    if B.Ctr is not None:
                        Etr, Ete = np.c_[Etr, B.Ctr], np.c_[Ete, B.Cte]
                    p_pr = cv_probe(Etr, Ete, sup, ys)
                    p = p_ft
                else:
                    p, p_ft, p_pr, r = locus_predict(B, sup, s, kw, gtr, gte, make, st, emb, v.get("use_char", True),
                                                     v.get("use_agree", True), key=name)
                p = {"head": p, "probe": p_pr}.get(v.get("only"), p_ft)  # LOCUS = fine-tuned GNN
            P.append(p)
            RA.append(repair_acc(r, B) if r is not None else float("nan"))
            log_row("ablate", {"ds": ds, "k": k, "seed": s, "variant": name, "repair": RA[-1], **metrics(p, B.yte)})
        log_row("ablate", {"ds": ds, "k": k, "seed": "ens", "variant": name, "repair": float(np.nanmean(RA)) if not all(np.isnan(RA)) else None,
                           **metrics(np.mean(P, 0), B.yte)})
        print(f"  {name} done", flush=True)


def transfer_predict(S, sup, s, kw, tgt_graphs, *unused):
    """LOCUS (pretrained, fine-tuned GNN) trained on the source support, applied to target graphs
    (target node features are z-scored with the target's own UNLABELED statistics)."""
    m = train(L(), [S.gtr[i] for i in sup], s, lr=3e-4, init=pretrained(S, s, L()), **kw)
    return predict(m, tgt_graphs)[0]


def run_xengine(ds):
    """Engine transfer: train on claims of one engine, test on another engine's claims."""
    index_pretrain_truth(ds)
    A, T = Bundle(ds, "tess"), Bundle(ds, "trocr")
    for src, tgt, sn, tn in ((A, T, "tess", "trocr"), (T, A, "trocr", "tess")):
        gte = [build(f, src.nz, rep=repair_target(f, ds)) for f in tgt.test]
        for k in (25, "full"):
            P, Pin = [], []
            for s in SEEDS:
                sup, kw = split_for(src, k, s)
                P.append(transfer_predict(src, sup, s, kw, gte, tgt.Cte, np.c_[np.array([zero_shot_score(f) for f in tgt.test]) / 10, np.stack([f['agree'][[1, 3]] for f in tgt.test])]))
                tsup, tkw = split_for(tgt, k, s)  # in-engine reference
                Pin.append(locus_predict(tgt, tsup, s, tkw)[1])
            log_row("xengine", {"ds": ds, "train": sn, "test": tn, "k": k, "seed": "ens", "setting": "transfer",
                                **metrics(np.mean(P, 0), tgt.yte)})
            log_row("xengine", {"ds": ds, "train": tn, "test": tn, "k": k, "seed": "ens", "setting": "in-engine",
                                **metrics(np.mean(Pin, 0), tgt.yte)})


def run_xdataset(src):
    """Dataset transfer: full-supervision LOCUS on src applied to every other dataset."""
    index_pretrain_truth(src)
    S = Bundle(src)
    sups = [split_for(S, "full", s) for s in SEEDS]
    targets = os.environ.get("XDS_TARGETS", "FUNSD,CORD,SROIE,IAM,IIIT5K").split(",")
    for tgt in targets:
        if tgt == src or not os.path.exists(f"{ROOT}/data/{tgt}/{FEAT}_{MAIN_ENGINE[tgt]}.pt"):
            continue
        T = Bundle(tgt)
        gte = [build(f, T.nz, rep=repair_target(f, tgt)) for f in T.test]  # target unlabeled stats
        P = [transfer_predict(S, sup, s, kw, gte, T.Cte, np.c_[np.array([zero_shot_score(f) for f in T.test]) / 10, np.stack([f['agree'][[1, 3]] for f in T.test])]) for s, (sup, kw) in zip(SEEDS, sups)]
        log_row("xdataset", {"train": src, "test": tgt, "seed": "ens", **metrics(np.mean(P, 0), T.yte)})


def run_linegraph(ds, ks=(10, 25, 50, 100)):
    """Prior work (NCCI'26 70-d line-graph GCN) vs LOCUS on the SAME words. The prior method
    gets its published advantages: per-class k-means supports (needs pool labels) and a
    labelled validation split for early stopping. LOCUS gets random supports, no validation."""
    from models import LineGraphGCN
    from run import predict as predict_ce, train_neural
    index_pretrain_truth(ds)
    B = Bundle(ds)
    LG = {f["id"]: f for f in torch.load(f"{ROOT}/data/{ds}/linegraph_{B.engine}.pt", weights_only=False)}
    tr = [i for i, f in enumerate(B.train) if f["id"] in LG]
    te = [i for i, f in enumerate(B.test) if f["id"] in LG]
    X = np.concatenate([LG[B.train[i]["id"]]["x"] for i in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-6

    def g(f, y):
        x = torch.tensor(np.clip((f["x"] - mu) / sd, -5, 5), dtype=torch.float)
        n = len(x)
        e = [[i, i + 1] for i in range(n - 1)]
        e = torch.tensor(e + [[b, a] for a, b in e] + [[i, i] for i in range(n)], dtype=torch.long).t()
        return Data(x=x, edge_index=e, y=torch.tensor([int(y)]))

    lg_tr = [g(LG[B.train[i]["id"]], B.ytr[i]) for i in tr]
    lg_te = [g(LG[B.test[i]["id"]], B.yte[i]) for i in te]
    ytr, yte = B.ytr[tr], B.yte[te]
    Xpool = np.stack([x.x.mean(0).numpy() for x in lg_tr])
    log_row("linegraph", {"ds": ds, "method": "coverage", "coverage_test": len(te) / len(B.test),
                          "n_test": len(te), "n_train_pool": len(tr)})
    lo_te = [B.gte[i] for i in te]
    for k in list(ks) + ["full"]:
        P = {"LineGraph-GCN (prior)": [], "LOCUS": []}
        for s in SEEDS:
            rng = np.random.RandomState(s)
            if k == "full":
                perm = rng.permutation(len(lg_tr))
                nv = int(0.15 * len(perm))
                val, sup, csup = perm[:nv], perm[nv:], perm[nv:]
                kw = dict(epochs=60, val=[B.gtr[tr[i]] for i in val])
            else:
                sup = sample_support(ytr, k, s, "kmeans_oracle", Xpool)
                rest = np.setdiff1d(np.arange(len(lg_tr)), sup)
                val = rng.choice(rest, min(len(rest), 300), replace=False)
                csup, kw = sample_support(ytr, k, s), dict(steps=300)
            m = train_neural(LineGraphGCN, [lg_tr[i] for i in sup], s, lr=1e-3, epochs=50,
                             val=[lg_tr[i] for i in val], ce2=True)
            P["LineGraph-GCN (prior)"].append(predict_ce(m, lg_te, ce2=True))
            init = pretrained(B, s, L()) if B.pre is not None else None
            m = train(L(), [B.gtr[tr[i]] for i in csup], s, lr=3e-4 if init else 1e-3, init=init, **kw)
            P["LOCUS"].append(predict(m, lo_te)[0])
            for n_ in P:
                log_row("linegraph", {"ds": ds, "k": k, "seed": s, "method": n_, **metrics(P[n_][-1], yte)})
        for n_ in P:
            log_row("linegraph", {"ds": ds, "k": k, "seed": "ens", "method": n_, **metrics(np.mean(P[n_], 0), yte)})
        print(f"  linegraph k={k} done", flush=True)


def run_ablate_single(ds, k=25):
    """Is render-normalisation needed when only ONE (possibly biased) verifier is available?"""
    index_pretrain_truth(ds)
    B = Bundle(ds)
    V = {"TrOCR only": COL_PARSEQ, "TrOCR only, w/o render-norm.": COL_PARSEQ + _g([1, 2, 5, 6])[:4] + COL_LAT[:2],
         "PARSeq only": COL_TROCR, "PARSeq only, w/o render-norm.": COL_TROCR + [12, 13, 16, 17, 18, 19]}
    for name, drop in V.items():
        build.no_agree = True  # single-verifier setting: no agreement shortcut either
        gtr = [build(f, B.nz, drop, rep=0 if f["y"] == 1 else -1) for f in B.train]
        gte = [build(f, B.nz, drop, rep=repair_target(f, ds)) for f in B.test]
        pre = [build(f, B.nz, drop, y=f["pseudo_label"], rep=pre_rep(f, ds)) for f in B.pre_raw]
        build.no_agree = False
        P = []
        for s in SEEDS:
            sup, kw = split_for(B, k, s)
            st = copy.deepcopy(train(L(), pre, s, epochs=15).state_dict())
            m = train(L(), [gtr[i] for i in sup], s, lr=3e-4, init=st, **kw)
            P.append(predict(m, gte)[0])
            log_row("ablate_single", {"ds": ds, "k": k, "seed": s, "variant": name, **metrics(P[-1], B.yte)})
        log_row("ablate_single", {"ds": ds, "k": k, "seed": "ens", "variant": name, **metrics(np.mean(P, 0), B.yte)})
        print(f"  {name} done", flush=True)


def run_sensitivity(ds, k=25):
    """(a) size of the label-free pretraining set, (b) fine-tuning steps, (c) support selection."""
    index_pretrain_truth(ds)
    B = Bundle(ds)
    for frac in (0.1, 0.25, 0.5, 1.0):
        P = []
        for s in SEEDS:
            rng = np.random.RandomState(100 + s)
            sub = [B.pre[i] for i in rng.permutation(len(B.pre))[: max(32, int(frac * len(B.pre)))]]
            st = pretrained(B, s, L()) if frac == 1.0 else copy.deepcopy(train(L(), sub, s, epochs=15).state_dict())
            sup, kw = split_for(B, k, s)
            m = train(L(), [B.gtr[i] for i in sup], s, lr=3e-4, init=st, **kw)
            P.append(predict(m, B.gte)[0])
        log_row("sensitivity", {"ds": ds, "k": k, "factor": "pretrain fraction", "value": frac, **metrics(np.mean(P, 0), B.yte)})
    for steps in (100, 300, 1000):
        P = []
        for s in SEEDS:
            sup, _ = split_for(B, k, s)
            m = train(L(), [B.gtr[i] for i in sup], s, lr=3e-4, init=pretrained(B, s, L()), steps=steps)
            P.append(predict(m, B.gte)[0])
        log_row("sensitivity", {"ds": ds, "k": k, "factor": "fine-tuning steps", "value": steps, **metrics(np.mean(P, 0), B.yte)})
    Eall = np.concatenate([locus_embeddings(B, 0)[0]], 1)  # label-free features for clustering
    for mode in ("random", "kmeans_labelfree"):
        P = []
        for s in SEEDS:
            sup = sample_support(B.ytr, k, s, mode, Eall)
            m = train(L(), [B.gtr[i] for i in sup], s, lr=3e-4, init=pretrained(B, s, L()), steps=300)
            P.append(predict(m, B.gte)[0])
        log_row("sensitivity", {"ds": ds, "k": k, "factor": "support selection", "value": mode, **metrics(np.mean(P, 0), B.yte)})
    print("sensitivity", ds, "done", flush=True)


def run_mp_analysis(ds, k=25):
    """Why does message passing add little? Controlled variants (each pretrained label-free with the
    same change): full / no MP / no explicit claim-relative features (deltas + agreement) / both /
    claim node only (no counterfactual rivals)."""
    index_pretrain_truth(ds)
    B = Bundle(ds)

    def claim_only(f):
        g = dict(f)
        for key in ("x", "kind", "pos"):
            g[key] = f[key][:1]
        g["hyps"], g["edit_pos"] = f["hyps"][:1], f["edit_pos"][:1]
        return g

    V = {"A full": dict(), "B no message passing": dict(make=L(conv="none")),
         "C no claim-relative features": dict(nodelta=True), "D no claim-relative features, no MP": dict(nodelta=True, make=L(conv="none")),
         "E claim node only (no rivals)": dict(claim_only=True)}
    for name, v in V.items():
        make = v.get("make", L())
        build.no_delta = build.no_agree = v.get("nodelta", False)
        tf = claim_only if v.get("claim_only") else (lambda f: f)
        gtr = [build(tf(f), B.nz, rep=0 if f["y"] == 1 else -1) for f in B.train]
        gte = [build(tf(f), B.nz, rep=0 if f["y"] == 1 else -1) for f in B.test]
        pre = [build(tf(f), B.nz, y=f["pseudo_label"], rep=pre_rep(f, ds) if not v.get("claim_only") else (0 if f["pseudo_label"] == 1 else -1)) for f in B.pre_raw]
        build.no_delta = build.no_agree = False
        P = []
        for s in SEEDS:
            st = pretrained(B, s, L()) if name == "A full" else copy.deepcopy(train(make, pre, s, epochs=15).state_dict())
            sup, kw = split_for(B, k, s)
            m = train(make, [gtr[i] for i in sup], s, lr=3e-4, init=st, **kw)
            P.append(predict(m, gte)[0])
        log_row("mp_analysis", {"ds": ds, "k": k, "seed": "ens", "variant": name, **metrics(np.mean(P, 0), B.yte)})
        print(f"  {name} done", flush=True)


if __name__ == "__main__":
    exp, ds = sys.argv[1], sys.argv[2]
    t = time.time()
    {"main": run_main, "ablate": run_ablate, "xengine": run_xengine, "xdataset": run_xdataset,
     "linegraph": run_linegraph, "ablate_single": run_ablate_single, "sensitivity": run_sensitivity,
     "mp_analysis": run_mp_analysis}[exp](ds)
    print(f"{exp} {ds} finished in {time.time() - t:.0f}s")

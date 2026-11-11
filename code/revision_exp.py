"""Experiments added in the major revision (reviewer comments R1.x / R2.x).

usage: python code/revision_exp.py <experiment> <dataset>
  noise    run-to-run variability of the full model and of "no message passing" (3 independent re-runs each,
           identical configuration; differences come only from non-deterministic GPU kernels)       [R1.2, R1.1, R2.1]
  gsize    graph capacity / pruning sensitivity: truncate the priority-ordered neighbourhood to M nodes,
           readings only, edits only (no recogniser readings)                                         [R2.7, R1.5]
  pseudo   noise-robust label-free pretraining: confidence-thresholded, confidence-weighted and
           three-engine-agreement pseudo-labels                                                       [R2.10, R2.11]
  xcal     engine transfer with recalibration (label-free prior shift EM, k-shot target threshold)   [R2.12]
Results: results/rev_<experiment>.jsonl (+ rev_<experiment>_<ds>.npz predictions where useful)
"""
import copy, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import locus as LC
from common import CASE_INSENSITIVE, ROOT, load_hyp, metrics, norm
from run import best_threshold

DEV = LC.DEV
SEEDS = LC.SEEDS


def log(exp, row):
    with open(f"{ROOT}/results/rev_{exp}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def train_w(make, graphs, seed, steps=300, lr=1e-3, init=None, epochs=None, lam=0.5):
    """LC.train with an optional per-graph weight g.w (pseudo-label confidence)."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = make().to(DEV)
    if init is not None:
        m.load_state_dict(init)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    ys = np.array([int(g.y) for g in graphs])
    pw = torch.tensor(max(1, (ys == 0).sum()) / max(1, (ys == 1).sum()), device=DEV, dtype=torch.float)
    bs = int(min(64, max(4, len(graphs) // 2)))
    ld = DataLoader(graphs, batch_size=bs, shuffle=True, drop_last=len(graphs) > bs)
    step = 0
    for ep in range(epochs or 10 ** 9):
        m.train()
        for b in ld:
            b = b.to(DEV)
            logit, logp = m(b)
            w = b.w if hasattr(b, "w") and b.w is not None else torch.ones_like(logit)
            loss = (F.binary_cross_entropy_with_logits(logit, b.y.float(), pos_weight=pw, reduction="none") * w).mean()
            ok = b.rep >= 0
            if lam > 0 and ok.any():
                idx = (b.ptr[:-1] + b.rep.clamp(min=0))[ok]
                loss = loss - lam * (logp[idx] * w[ok]).sum() / w[ok].sum().clamp(min=1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if not epochs and step >= steps:
                return m
    return m


def fit_eval(B, gtr, gte, pre, make=None, tag=""):
    """Label-free pretraining on `pre` + k=25 fine-tuning, 5 seeds; returns per-seed and ensemble P."""
    make = make or LC.L()
    P = []
    for s in SEEDS:
        st = copy.deepcopy(train_w(make, pre, s, epochs=15).state_dict())
        sup, kw = LC.split_for(B, 25, s)
        m = LC.train(make, [gtr[i] for i in sup], s, lr=3e-4, init=st, **kw)
        P.append(LC.predict(m, gte)[0])
    return np.array(P)


# ----------------------------------------------------------------------------------------------- noise
def run_noise(ds, R=3):
    LC.index_pretrain_truth(ds)
    B = LC.Bundle(ds)
    out = {}
    for name, make in (("full", LC.L()), ("no message passing", LC.L(conv="none"))):
        for r in range(R):
            P = fit_eval(B, B.gtr, B.gte, B.pre, make)
            out[f"{name}|{r}"] = P
            log("noise", {"ds": ds, "variant": name, "rerun": r, **metrics(P.mean(0), B.yte)})
            print(f"  {name} rerun {r}", flush=True)
    np.savez_compressed(f"{ROOT}/results/rev_noise_{ds}.npz", y=B.yte, **out)


# ----------------------------------------------------------------------------------------------- gsize
KIND_READ = (6, 7)          # TrOCR / PARSeq greedy readings
KIND_ALIGN = (4, 5)         # reading-aligned substitutions / insertions


def subgraph(f, keep):
    g = dict(f)
    keep = [0] + [j for j in keep if j != 0]
    g["x"], g["kind"], g["pos"] = f["x"][keep], f["kind"][keep], f["pos"][keep]
    g["hyps"] = [f["hyps"][j] for j in keep]
    g["edit_pos"] = [f["edit_pos"][j] for j in keep]
    return g


def sel_trunc(M):
    return lambda f: list(range(min(len(f["hyps"]), M + 1)))


def sel_readings(f):
    return [j for j in range(len(f["hyps"])) if f["kind"][j, list(KIND_READ)].any()]


def sel_no_readings(f):
    return [j for j in range(len(f["hyps"])) if not f["kind"][j, list(KIND_READ + KIND_ALIGN)].any()]


def rebuild(B, ds, sel):
    tf = lambda f: subgraph(f, sel(f))
    gtr = [LC.build(tf(f), B.nz, rep=0 if f["y"] == 1 else -1) for f in B.train]
    tes = [tf(f) for f in B.test]
    gte = [LC.build(t, B.nz, rep=LC.repair_target(t, ds)) for t in tes]
    pres = [tf(f) for f in B.pre_raw]
    pre = [LC.build(t, B.nz, y=t["pseudo_label"], rep=LC.pre_rep(t, ds)) for t in pres]
    nodes = float(np.mean([len(t["hyps"]) for t in tes]))
    cov = float(np.mean([g.rep.item() > 0 for g, y in zip(gte, B.yte) if y == 0]))
    return gtr, gte, pre, nodes, cov


def run_gsize(ds):
    LC.index_pretrain_truth(ds)
    B = LC.Bundle(ds)
    capped = float(np.mean([len(f["hyps"]) >= 29 for f in B.test]))
    log("gsize", {"ds": ds, "variant": "share of test claims at the M=28 cap", "value": capped,
                  "mean_len": float(np.mean([len(f["claim"]) for f in B.test]))})
    V = [("M=2", sel_trunc(2)), ("M=4", sel_trunc(4)), ("M=8", sel_trunc(8)), ("M=16", sel_trunc(16)),
         ("M=28 (default)", sel_trunc(28)), ("readings only", sel_readings),
         ("edits only (no recogniser readings)", sel_no_readings)]
    store = {"y": B.yte}
    for name, sel in V:
        gtr, gte, pre, nodes, cov = rebuild(B, ds, sel)
        P = fit_eval(B, gtr, gte, pre)
        # repair from a pretrained-only model (as in the main protocol) is skipped: coverage is reported instead
        store[name] = P
        log("gsize", {"ds": ds, "variant": name, "nodes": nodes, "truth_in_graph": cov, **metrics(P.mean(0), B.yte)})
        print(f"  {name}: nodes {nodes:.1f} cov {cov:.3f}", flush=True)
    np.savez_compressed(f"{ROOT}/results/rev_gsize_{ds}.npz", **{k.replace("/", "_"): v for k, v in store.items()})


# ----------------------------------------------------------------------------------------------- pseudo
def run_pseudo(ds):
    LC.index_pretrain_truth(ds)
    B = LC.Bundle(ds)
    e1, e2 = ("trocr", "trocrB") if ds == "IAM" else ("tess", "trocr")
    H1, H2 = load_hyp(ds, e1), load_hyp(ds, e2)
    man = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/data/{ds}/manifest.jsonl")}
    ci = CASE_INSENSITIVE.get(ds, False)
    pos = {f["id"]: f for f in B.pre_raw if f["pseudo_label"] == 1}
    ids = sorted(pos)
    conf = {i: float(H1[i]["conf"]) * float(H2[i]["conf"]) for i in ids}
    # pseudo-label precision is measured with the training ground truth (analysis only, never used in training)
    correct = {i: int(norm(pos[i]["claim"], ci) == norm(man[i]["gt"], ci)) for i in ids}
    triple = {i: int(norm(pos[i]["claim"], ci) == norm(pos[i]["g2"], ci)) for i in ids}
    c = np.array([conf[i] for i in ids])
    med = float(np.quantile(c, 0.3))  # drop the 30% least confident agreements (label-free rule)
    rank = {i: r for r, i in enumerate(np.array(ids)[np.argsort(c)])}
    wsoft = {i: 0.25 + 1.5 * rank[i] / max(1, len(ids) - 1) for i in ids}  # rank-based weight in [0.25, 1.75]
    V = {"agreement (default)": (lambda i: True, None),
         "confidence-thresholded (drop lowest 30%)": (lambda i: conf[i] >= med, None),
         "confidence-weighted (rank weights)": (lambda i: True, wsoft),
         "three-engine agreement (+PARSeq)": (lambda i: triple[i] == 1, None)}
    for name, (keep, wmap) in V.items():
        kept = [i for i in ids if keep(i)]
        prec = float(np.mean([correct[i] for i in kept])) if kept else float("nan")
        if wmap is not None:
            prec = float(np.average([correct[i] for i in kept], weights=[wmap[i] for i in kept]))
        ks = set(kept)
        pre = []
        for f, g in zip(B.pre_raw, B.pre):
            if f["id"] in ks:
                g = g.clone()
                g.w = torch.tensor([wmap[f["id"]] if wmap else 1.0])
                pre.append(g)
        P = fit_eval(B, B.gtr, B.gte, pre)
        log("pseudo", {"ds": ds, "variant": name, "n_pos": len(kept), "precision": prec,
                       "precision_all": float(np.mean(list(correct.values()))), **metrics(P.mean(0), B.yte)})
        print(f"  {name}: kept {len(kept)} prec {prec:.3f}", flush=True)


# ----------------------------------------------------------------------------------------------- xcal
def ece(p, y, nb=15):
    p = np.asarray(p); y = np.asarray(y)
    bins = np.minimum((p * nb).astype(int), nb - 1)
    return float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(nb) if (bins == b).any()))


def em_prior(p, pi_src, it=100):
    """Saerens-style EM: re-estimate the target class prior from unlabelled posteriors; return adjusted posteriors."""
    pi = pi_src
    for _ in range(it):
        a = p * pi / pi_src
        b = (1 - p) * (1 - pi) / (1 - pi_src)
        q = a / (a + b)
        pi_new = q.mean()
        if abs(pi_new - pi) < 1e-6:
            break
        pi = pi_new
    return q, pi


def run_xcal(ds):
    LC.index_pretrain_truth(ds)
    Bs = {"tess": LC.Bundle(ds, "tess"), "trocr": LC.Bundle(ds, "trocr")}
    for sn, tn in (("tess", "trocr"), ("trocr", "tess")):
        S, T = Bs[sn], Bs[tn]
        Ptr, Pte = [], []
        for s in SEEDS:
            sup, kw = LC.split_for(S, 25, s)
            m = LC.train(LC.L(), [S.gtr[i] for i in sup], s, lr=3e-4, init=LC.pretrained(S, s, LC.L()), **kw)
            Pte.append(LC.predict(m, T.gte)[0]); Ptr.append(LC.predict(m, T.gtr)[0])
        pte, ptr = np.mean(Pte, 0), np.mean(Ptr, 0)
        # balanced fine-tuning (pos_weight) makes the source-model prior 0.5
        row = {"ds": ds, "train": sn, "test": tn, "target_correct_rate": float(T.yte.mean())}
        log("xcal", {**row, "method": "transfer, threshold 0.5", "ece": ece(pte, T.yte), **metrics(pte, T.yte)})
        q_tr, pi = em_prior(ptr, 0.5)                       # prior estimated on the UNLABELLED target pool
        q_te = pte * pi / 0.5 / (pte * pi / 0.5 + (1 - pte) * (1 - pi) / 0.5)
        log("xcal", {**row, "method": "label-free prior-shift EM", "pi_hat": float(pi), "ece": ece(q_te, T.yte),
                     **metrics(q_te, T.yte)})
        for kt in (5, 25):
            M, E = [], []
            for s in SEEDS:
                sup = LC.sample_support(T.ytr, kt, s)
                t = best_threshold(ptr[sup], T.ytr[sup])
                M.append(metrics(pte, T.yte, thr=t))
            log("xcal", {**row, "method": f"target threshold from {kt} labels/class",
                         **{k: float(np.mean([m[k] for m in M])) for k in M[0]}})
        log("xcal", {**row, "method": "in-engine reference (k=25)", **metrics(
            np.mean([LC.predict(LC.train(LC.L(), [T.gtr[i] for i in LC.split_for(T, 25, s)[0]], s, lr=3e-4,
                                         init=LC.pretrained(T, s, LC.L()), steps=300), T.gte)[0] for s in SEEDS], 0), T.yte)})
        print(f"  {sn}->{tn} done", flush=True)


if __name__ == "__main__":
    exp, ds = sys.argv[1], sys.argv[2]
    lock = f"{ROOT}/logs/rev/lock_{exp}_{ds}"
    if os.path.exists(lock):
        print("already taken", exp, ds); sys.exit(0)
    open(lock, "w").write(str(os.getpid()))
    t = time.time()
    {"noise": run_noise, "gsize": run_gsize, "pseudo": run_pseudo, "xcal": run_xcal}[exp](ds)
    print(f"{exp} {ds} finished in {time.time() - t:.0f}s", flush=True)

"""Experiment runner. Every number in the paper is produced by this file.

usage: python run.py <experiment> <dataset> [extra]
  main      few-shot learning curves for all methods (k grid x 5 seeds + ensembles)
  ablate    feature / structure / readout / backbone / pretraining ablations at k=25
  sampling  support-selection ablation (random / per-class k-means oracle / label-free k-means)
  xengine   train on Tesseract claims, test on TrOCR claims (and vice-versa)
Results are appended as JSON rows to results/<experiment>.jsonl
"""
import copy, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from common import (GLOB_GROUPS, MAIN_ENGINE, NODE_GROUPS, ROOT, Normaliser, best_threshold,
                    build_graph, load_feats, load_hyp, metrics, pooled_vector)
from models import CEGNet, LineGraphGCN, SetNet, TransformerNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2, 3, 4]
KS = [5, 10, 25, 50, 100]


def log_row(exp, row):
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(f"{ROOT}/results/{exp}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ supports
def sample_support(y, k, seed, mode="random", X=None):
    rng = np.random.RandomState(seed)
    if mode == "random":
        idx = []
        for c in (0, 1):
            pool = np.where(y == c)[0]
            idx += list(rng.choice(pool, min(k, len(pool)), replace=False))
        return np.array(idx)
    if mode == "kmeans_oracle":  # prior work: per-class k-means (needs labels of the pool)
        idx = []
        for c in (0, 1):
            pool = np.where(y == c)[0]
            km = KMeans(min(k, len(pool)), n_init=3, random_state=seed).fit(X[pool])
            for j in range(km.n_clusters):
                d = ((X[pool] - km.cluster_centers_[j]) ** 2).sum(1)
                idx.append(pool[d.argmin()])
        return np.array(sorted(set(idx)))
    if mode == "kmeans_labelfree":  # cluster the UNLABELED pool into 2k, label the medoids
        km = KMeans(2 * k, n_init=3, random_state=seed).fit(X)
        idx = [int(((X - c) ** 2).sum(1).argmin()) for c in km.cluster_centers_]
        return np.array(sorted(set(idx)))
    raise ValueError(mode)


# ------------------------------------------------------------------ neural training
def train_neural(make, graphs, seed, steps=300, lr=1e-3, init=None, val=None, epochs=None,
                 ce2=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make().to(DEV)
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ys = np.array([int(g.y) for g in graphs])
    pos_w = torch.tensor(max(1e-3, (ys == 0).sum()) / max(1e-3, (ys == 1).sum()), device=DEV)
    bs = int(min(64, max(4, len(graphs) // 2)))
    loader = DataLoader(graphs, batch_size=bs, shuffle=True, drop_last=len(graphs) > bs)
    best, best_state, bad, step = -1, None, 0, 0
    n_epochs = epochs if epochs else 10 ** 9
    for ep in range(n_epochs):
        model.train()
        for b in loader:
            b = b.to(DEV)
            out = model(b)
            if ce2:  # prior-work model: 2-way softmax cross-entropy
                loss = F.cross_entropy(out, b.y)
            else:
                loss = F.binary_cross_entropy_with_logits(out, b.y.float(), pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if not epochs and step >= steps:
                return model
        if val is not None:  # full-supervision only: early stopping on a val split
            p = predict(model, val, ce2)
            m = metrics(p, np.array([int(g.y) for g in val]))["macro_f1"]
            if m > best:
                best, best_state, bad = m, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= 8:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, graphs, ce2=False):
    model.eval()
    out = []
    for b in DataLoader(graphs, batch_size=512):
        o = model(b.to(DEV))
        out.append(torch.softmax(o, -1)[:, 1] if ce2 else torch.sigmoid(o))
    return torch.cat(out).cpu().numpy()


# ------------------------------------------------------------------ dataset bundle
class Bundle:
    def __init__(self, ds, engine=None):
        self.ds, self.engine = ds, engine or MAIN_ENGINE[ds]
        F_ = load_feats(ds, self.engine)
        self.train = [f for f in F_ if f["split"] == "train"]
        self.test = [f for f in F_ if f["split"] == "test"]
        self.nz = Normaliser(self.train)  # unlabeled statistics
        self.ytr = np.array([f["y"] for f in self.train])
        self.yte = np.array([f["y"] for f in self.test])
        self.Xtr = np.stack([pooled_vector(f, self.nz) for f in self.train])
        self.Xte = np.stack([pooled_vector(f, self.nz) for f in self.test])
        self.gtr = [build_graph(f, self.nz) for f in self.train]
        self.gte = [build_graph(f, self.nz) for f in self.test]
        pt = f"{ROOT}/data/{ds}/feat_pretrain.pt"
        self.pre = None
        if os.path.exists(pt) and ds != "OCR10K":
            P = torch.load(pt, weights_only=False)
            for f in P:
                f["y"] = int(f["pseudo_label"])
            self.pre = [build_graph(f, self.nz) for f in P]
        # engine-native confidences (baseline); missing for OCR10K (claims are given)
        self.conf_te = self.conf_tr = None
        if self.engine != "given":
            H = load_hyp(ds, self.engine)
            self.conf_tr = np.array([H[f["id"]]["conf"] for f in self.train])
            self.conf_te = np.array([H[f["id"]]["conf"] for f in self.test])
        # two-engine agreement (classic voting baseline)
        self.agree_te = None
        other = {"tess": "trocr", "trocr": "trocrB" if ds == "IAM" else "tess"}.get(self.engine)
        if other and os.path.exists(f"{ROOT}/data/{ds}/hyp_{other}.jsonl"):
            H2 = load_hyp(ds, other)
            self.agree_te = np.array([float(H2[f["id"]]["hyp"].strip() == f["claim"].strip())
                                      for f in self.test])
        print(f"[{ds}/{self.engine}] train {len(self.train)} (err {1 - self.ytr.mean():.3f}) "
              f"test {len(self.test)} (err {1 - self.yte.mean():.3f}) pretrain "
              f"{0 if self.pre is None else len(self.pre)}", flush=True)


_PT_CACHE = {}


def pretrained_state(B, seed, make):
    key = (B.ds, B.engine, seed, make.__name__ if hasattr(make, "__name__") else str(make))
    if key not in _PT_CACHE:
        m = train_neural(make, B.pre, seed, epochs=15, lr=1e-3)
        _PT_CACHE[key] = copy.deepcopy(m.state_dict())
    return _PT_CACHE[key]


def ceg(**kw):
    def make():
        return CEGNet(**kw)
    make.__name__ = "ceg_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    return make


# ------------------------------------------------------------------ methods
def score_baselines(B, sup):
    """Threshold-only baselines: threshold fitted on the same labelled support set."""
    out = {}
    ys = B.ytr[sup]
    raw = {"VerLL": lambda X: X[:, 0], "RenderNormLL": lambda X: X[:, 2]}
    gtr = np.stack([f["glob"] for f in B.train])
    gte = np.stack([f["glob"] for f in B.test])
    for name, fn in raw.items():
        t = best_threshold(fn(gtr)[sup], ys)
        s = fn(gte)
        out[name] = (s, t)
    if B.conf_tr is not None:
        t = best_threshold(B.conf_tr[sup], ys)
        out["EngineConf"] = (B.conf_te, t)
    return out


def run_main(ds):
    B = Bundle(ds)
    exp = "main"
    # label-free baselines (no training)
    reocr = np.stack([f["glob"][4] for f in B.test])  # claim == verifier's own reading
    log_row(exp, {"ds": ds, "method": "ReOCR-Agree", "k": 0, "seed": -1, **metrics(reocr, B.yte)})
    if B.agree_te is not None:
        log_row(exp, {"ds": ds, "method": "TwoEngine-Agree", "k": 0, "seed": -1,
                      **metrics(B.agree_te, B.yte)})
    if B.pre is not None:  # zero-shot CEGNet (label-free pretraining only)
        P = []
        for s in SEEDS:
            mdl = ceg()().to(DEV)
            mdl.load_state_dict(pretrained_state(B, s, ceg()))
            P.append(predict(mdl, B.gte))
            log_row(exp, {"ds": ds, "method": "CEGNet-ZeroShot", "k": 0, "seed": s, **metrics(P[-1], B.yte)})
        log_row(exp, {"ds": ds, "method": "CEGNet-ZeroShot", "k": 0, "seed": "ens",
                      **metrics(np.mean(P, 0), B.yte)})
    for k in KS + ["full"]:
        preds = {}
        for s in SEEDS:
            if k == "full":
                rng = np.random.RandomState(s)
                perm = rng.permutation(len(B.train))
                nv = int(0.15 * len(perm))
                val, sup = perm[:nv], perm[nv:]
            else:
                sup, val = sample_support(B.ytr, k, s), None
            ysup = B.ytr[sup]
            row = {"ds": ds, "k": k, "seed": s}
            for name, (sc, t) in score_baselines(B, sup).items():
                m = metrics(sc, B.yte, thr=t)
                log_row(exp, {**row, "method": name, **m})
            # graph-free learned baselines on the same evidence
            lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(B.Xtr[sup], ysup)
            pl = lr.predict_proba(B.Xte)[:, 1]
            preds.setdefault("LogReg", []).append(pl)
            log_row(exp, {**row, "method": "LogReg", **metrics(pl, B.yte)})
            gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                min_samples_leaf=max(2, min(20, len(sup) // 10)),
                                                class_weight="balanced", random_state=s)
            pg = gb.fit(B.Xtr[sup], ysup).predict_proba(B.Xte)[:, 1]
            preds.setdefault("GBDT", []).append(pg)
            log_row(exp, {**row, "method": "GBDT", **metrics(pg, B.yte)})
            gs = [B.gtr[i] for i in sup]
            gv = [B.gtr[i] for i in val] if val is not None else None
            kw = dict(epochs=60, val=gv) if k == "full" else dict(steps=300)
            # CEGNet without pretraining
            m = train_neural(ceg(), gs, s, lr=1e-3, **kw)
            p = predict(m, B.gte)
            preds.setdefault("CEGNet-scratch", []).append(p)
            log_row(exp, {**row, "method": "CEGNet-scratch", **metrics(p, B.yte)})
            # CEGNet with label-free pretraining (full method)
            if B.pre is not None:
                m = train_neural(ceg(), gs, s, lr=3e-4, init=pretrained_state(B, s, ceg()), **kw)
                p = predict(m, B.gte)
                preds.setdefault("CEGNet", []).append(p)
                log_row(exp, {**row, "method": "CEGNet", **metrics(p, B.yte)})
        for name, P in preds.items():
            log_row(exp, {"ds": ds, "k": k, "seed": "ens", "method": name, **metrics(np.mean(P, 0), B.yte)})
        print(f"  k={k} done", flush=True)


def run_ablate(ds, k=25):
    """All variants at fixed k with 5 seeds (+ensemble)."""
    B = Bundle(ds)
    exp = "ablate"
    variants = {
        "full": dict(make=ceg(), drop_node=(), drop_glob=(), pre=True),
        "no-pretrain": dict(make=ceg(), drop_node=(), drop_glob=(), pre=False),
        "w/o counterfactual": dict(make=ceg(), drop_node=NODE_GROUPS["cf"], drop_glob=GLOB_GROUPS["cf"], pre=True),
        "w/o render-norm": dict(make=ceg(), drop_node=NODE_GROUPS["render"], drop_glob=GLOB_GROUPS["render"], pre=True),
        "w/o verifier-align": dict(make=ceg(), drop_node=NODE_GROUPS["align"], drop_glob=GLOB_GROUPS["align"], pre=True),
        "likelihood only": dict(make=ceg(), drop_node=NODE_GROUPS["render"] + NODE_GROUPS["cf"] + NODE_GROUPS["align"],
                                drop_glob=GLOB_GROUPS["render"] + GLOB_GROUPS["cf"] + GLOB_GROUPS["align"], pre=True),
        "mean readout": dict(make=ceg(readout="mean"), drop_node=(), drop_glob=(), pre=True),
        "chain only": dict(make=ceg(), edges=("chain",), pre=True),
        "no global node": dict(make=ceg(), edges=("chain", "skip"), pre=True),
        "backbone GCN": dict(make=ceg(conv="gcn"), pre=True),
        "backbone SAGE": dict(make=ceg(conv="sage"), pre=True),
        "backbone GIN": dict(make=ceg(conv="gin"), pre=True),
        "backbone GAT": dict(make=ceg(conv="gat"), pre=True),
        "DeepSets (no edges)": dict(make=SetNet, pre=True),
        "Transformer (dense)": dict(make=TransformerNet, pre=True),
    }
    for name, v in variants.items():
        dn, dg = v.get("drop_node", ()), v.get("drop_glob", ())
        edges = v.get("edges", ("chain", "skip", "global"))
        custom = bool(dn or dg or "edges" in v)
        gtr = [build_graph(f, B.nz, dn, dg, edges) for f in B.train] if custom else B.gtr
        gte = [build_graph(f, B.nz, dn, dg, edges) for f in B.test] if custom else B.gte
        pre = [build_graph(f, B.nz, dn, dg, edges) for f in _raw_pre(ds)] if (custom and v["pre"] and B.pre) else B.pre
        P = []
        for s in SEEDS:
            sup = sample_support(B.ytr, k, s)
            init = None
            if v["pre"] and pre is not None:
                init = copy.deepcopy(train_neural(v["make"], pre, s, epochs=15).state_dict())
            m = train_neural(v["make"], [gtr[i] for i in sup], s, steps=300, lr=3e-4 if init else 1e-3, init=init)
            P.append(predict(m, gte))
            log_row(exp, {"ds": ds, "k": k, "seed": s, "variant": name, **metrics(P[-1], B.yte)})
        log_row(exp, {"ds": ds, "k": k, "seed": "ens", "variant": name, **metrics(np.mean(P, 0), B.yte)})
        print(f"  {name} done", flush=True)


def run_linegraph(ds, ks=(10, 25, 50, 100)):
    """Prior work (NCCI'26 line-graph GCN, 70-d features) vs CEGNet on the SAME words:
    the prior pipeline drops words whose proportional cuts are empty, so both methods are
    trained/tested on the intersection. The prior method is given the advantage it had in
    its paper: per-class k-means support selection and a labelled 15% validation split
    for early stopping (CEGNet uses neither)."""
    from torch_geometric.data import Data
    B = Bundle(ds)
    exp = "linegraph"
    LG = torch.load(f"{ROOT}/data/{ds}/linegraph_{B.engine}.pt", weights_only=False)
    lg = {f["id"]: f for f in LG}
    tr_ids = [i for i, f in enumerate(B.train) if f["id"] in lg]
    te_ids = [i for i, f in enumerate(B.test) if f["id"] in lg]
    Xall = np.concatenate([lg[B.train[i]["id"]]["x"] for i in tr_ids])
    mu, sd = Xall.mean(0), Xall.std(0) + 1e-6

    def lgraph(f, y):
        x = torch.tensor(np.clip((f["x"] - mu) / sd, -5, 5), dtype=torch.float)
        n = len(x)  # chain: RS0 - RR01 - RS1 - RR12 - ...
        e = [[i, i + 1] for i in range(n - 1)]
        e = torch.tensor(e + [[b, a] for a, b in e] + [[i, i] for i in range(n)], dtype=torch.long).t()
        return Data(x=x, edge_index=e, y=torch.tensor([y]))

    gtr = [lgraph(lg[B.train[i]["id"]], B.ytr[i]) for i in tr_ids]
    gte = [lgraph(lg[B.test[i]["id"]], B.yte[i]) for i in te_ids]
    ytr, yte = B.ytr[tr_ids], B.yte[te_ids]
    Xpool = np.stack([g.x.mean(0).numpy() for g in gtr])
    cov = {"ds": ds, "coverage_test": len(te_ids) / max(1, sum(1 for f in B.test if f["claim"].strip())),
           "n_test_covered": len(te_ids), "n_train_covered": len(tr_ids)}
    log_row(exp, {**cov, "method": "coverage"})
    cgte = [B.gte[i] for i in te_ids]
    for k in list(ks) + ["full"]:
        P = {"LineGraph-GCN (prior)": [], "CEGNet": []}
        for s in SEEDS:
            rng = np.random.RandomState(s)
            if k == "full":
                perm = rng.permutation(len(gtr))
                nv = int(0.15 * len(perm))
                val, sup = perm[:nv], perm[nv:]
            else:
                sup = sample_support(ytr, k, s, "kmeans_oracle", Xpool)
                rest = np.setdiff1d(np.arange(len(gtr)), sup)
                val = rng.choice(rest, min(len(rest), 300), replace=False)  # labelled val (prior work)
            m = train_neural(LineGraphGCN, [gtr[i] for i in sup], s, lr=1e-3, epochs=50,
                             val=[gtr[i] for i in val], ce2=True)
            P["LineGraph-GCN (prior)"].append(predict(m, gte, ce2=True))
            # CEGNet: random support of the same size, no validation labels
            if k == "full":
                csup, kw = sup, dict(epochs=60, val=[B.gtr[tr_ids[i]] for i in val])
            else:
                csup, kw = sample_support(ytr, k, s), dict(steps=300)
            init = pretrained_state(B, s, ceg()) if B.pre is not None else None
            m = train_neural(ceg(), [B.gtr[tr_ids[i]] for i in csup], s, lr=3e-4 if init else 1e-3,
                             init=init, **kw)
            P["CEGNet"].append(predict(m, cgte))
            for name in P:
                log_row(exp, {"ds": ds, "k": k, "seed": s, "method": name, **metrics(P[name][-1], yte)})
        for name in P:
            log_row(exp, {"ds": ds, "k": k, "seed": "ens", "method": name, **metrics(np.mean(P[name], 0), yte)})
        print(f"  linegraph k={k} done", flush=True)


def run_sampling(ds, ks=(10, 25)):
    B = Bundle(ds)
    for k in ks:
        for mode in ("random", "kmeans_oracle", "kmeans_labelfree"):
            P = []
            for s in SEEDS:
                sup = sample_support(B.ytr, k, s, mode, B.Xtr)
                init = pretrained_state(B, s, ceg()) if B.pre is not None else None
                m = train_neural(ceg(), [B.gtr[i] for i in sup], s, steps=300,
                                 lr=3e-4 if init else 1e-3, init=init)
                P.append(predict(m, B.gte))
                log_row("sampling", {"ds": ds, "k": k, "seed": s, "mode": mode, "n_labels": len(sup),
                                     "frac_incorrect": float(1 - B.ytr[sup].mean()), **metrics(P[-1], B.yte)})
            log_row("sampling", {"ds": ds, "k": k, "seed": "ens", "mode": mode, **metrics(np.mean(P, 0), B.yte)})


def run_xengine(ds):
    """Engine transfer: train on claims of one engine, test on claims of another engine."""
    A, Bt = Bundle(ds, "tess"), Bundle(ds, "trocr")
    for src, tgt, sname, tname in ((A, Bt, "tess", "trocr"), (Bt, A, "trocr", "tess")):
        # graphs of the target engine, normalised with the SOURCE statistics (no target labels)
        gte = [build_graph(f, src.nz) for f in tgt.test]
        for k in (25, "full"):
            P, Pin = [], []
            for s in SEEDS:
                if k == "full":
                    perm = np.random.RandomState(s).permutation(len(src.train))
                    nv = int(0.15 * len(perm))
                    sup, kw = perm[nv:], dict(epochs=60, val=[src.gtr[i] for i in perm[:nv]])
                else:
                    sup, kw = sample_support(src.ytr, k, s), dict(steps=300)
                init = pretrained_state(src, s, ceg()) if src.pre is not None else None
                m = train_neural(ceg(), [src.gtr[i] for i in sup], s, lr=3e-4 if init else 1e-3, init=init, **kw)
                P.append(predict(m, gte))
                log_row("xengine", {"ds": ds, "train": sname, "test": tname, "k": k, "seed": s,
                                    **metrics(P[-1], tgt.yte)})
            log_row("xengine", {"ds": ds, "train": sname, "test": tname, "k": k, "seed": "ens",
                                **metrics(np.mean(P, 0), tgt.yte)})


def run_xdataset(src_ds):
    """Dataset transfer: fully supervised CEGNet on src, applied to every other dataset.
    Target features are z-scored with the TARGET's unlabeled statistics (label-free)."""
    S = Bundle(src_ds)
    models_ = []
    for s in SEEDS:
        perm = np.random.RandomState(s).permutation(len(S.train))
        nv = int(0.15 * len(perm))
        init = pretrained_state(S, s, ceg()) if S.pre is not None else None
        models_.append(train_neural(ceg(), [S.gtr[i] for i in perm[nv:]], s, lr=3e-4 if init else 1e-3,
                                    init=init, epochs=60, val=[S.gtr[i] for i in perm[:nv]]))
    for tgt in ["FUNSD", "CORD", "SROIE", "IAM", "IIIT5K", "OCR10K"]:
        if not os.path.exists(f"{ROOT}/data/{tgt}/feat_{MAIN_ENGINE[tgt]}.pt"):
            continue
        T = Bundle(tgt)
        P = [predict(m, T.gte) for m in models_]
        log_row("xdataset", {"train": src_ds, "test": tgt, "seed": "ens", **metrics(np.mean(P, 0), T.yte)})


def _raw_pre(ds):
    P = torch.load(f"{ROOT}/data/{ds}/feat_pretrain.pt", weights_only=False)
    for f in P:
        f["y"] = int(f["pseudo_label"])
    return P


if __name__ == "__main__":
    exp, ds = sys.argv[1], sys.argv[2]
    t = time.time()
    {"main": run_main, "ablate": run_ablate, "linegraph": run_linegraph, "sampling": run_sampling,
     "xengine": run_xengine, "xdataset": run_xdataset}[exp](ds)
    print(f"{exp} {ds} finished in {time.time() - t:.0f}s")

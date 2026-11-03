"""CEGNet and graph/graph-free comparison models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GATConv, GATv2Conv, GCNConv, GINConv, SAGEConv, global_max_pool,
                                global_mean_pool)
from torch_geometric.utils import softmax as seg_softmax

NODE_IN, GLOB_IN = 20, 15


def make_conv(kind, d):
    if kind == "gatv2":  # relation-aware attention (edge type as edge feature)
        return GATv2Conv(d, d // 4, heads=4, edge_dim=4, add_self_loops=False)
    if kind == "gat":
        return GATConv(d, d // 4, heads=4, edge_dim=4, add_self_loops=False)
    if kind == "gcn":
        return GCNConv(d, d, add_self_loops=False)
    if kind == "sage":
        return SAGEConv(d, d)
    if kind == "gin":
        return GINConv(nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)))
    raise ValueError(kind)


class CEGNet(nn.Module):
    """Counterfactual Evidence Graph network.

    char nodes  <- per-character counterfactual / render-normalised evidence
    global node <- word-level evidence (its input is the projected global features)
    readout     <- MLP([global, attention-pool(chars)]) - tau*logsumexp(suspicion_t / tau)
                   ("weakest link": one suspicious character suffices to reject the word)
    """

    def __init__(self, conv="gatv2", d=64, layers=3, dropout=0.2, readout="weakest", tau=0.5):
        super().__init__()
        self.conv_kind, self.readout, self.tau = conv, readout, tau
        self.inp_c = nn.Sequential(nn.Linear(NODE_IN, d), nn.ReLU())
        self.inp_g = nn.Sequential(nn.Linear(GLOB_IN, d), nn.ReLU())
        self.convs = nn.ModuleList([make_conv(conv, d) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.drop = dropout
        self.att = nn.Linear(d, 1)
        self.susp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def node_states(self, data):
        is_c = data.is_char
        h = torch.zeros(data.x.shape[0], self.inp_c[0].out_features, device=data.x.device)
        h[is_c] = self.inp_c(data.x[is_c])
        h[~is_c] = self.inp_g(data.gfeat)  # one global node per graph, in batch order
        for conv, ln in zip(self.convs, self.norms):
            if self.conv_kind in ("gatv2", "gat"):
                m = conv(h, data.edge_index, data.edge_attr)
            else:
                m = conv(h, data.edge_index)
            h = ln(h + F.dropout(F.relu(m), self.drop, self.training))
        return h

    def forward(self, data, return_susp=False):
        h = self.node_states(data)
        is_c = data.is_char
        hc, bc = h[is_c], data.batch[is_c]
        hg = h[~is_c]
        a = seg_softmax(self.att(hc).squeeze(-1), bc)
        zc = global_mean_pool(hc * a.unsqueeze(-1), bc) * torch.bincount(bc).unsqueeze(-1).float()
        logit = self.head(torch.cat([hg, zc], -1)).squeeze(-1)
        s = self.susp(hc).squeeze(-1)
        if self.readout == "weakest":
            # soft maximum of per-character suspicion, per graph
            smax = global_max_pool(s.unsqueeze(-1), bc).squeeze(-1)
            lse = smax + self.tau * torch.log(
                global_mean_pool(torch.exp((s - smax[bc]) / self.tau).unsqueeze(-1), bc).squeeze(-1)
                * torch.bincount(bc).float())
            logit = logit - lse
        return (logit, s) if return_susp else logit


class SetNet(nn.Module):
    """Graph-free ablation: same inputs, nodes encoded independently (DeepSets), no edges."""

    def __init__(self, d=64, dropout=0.2):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(NODE_IN, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.g = nn.Sequential(nn.Linear(GLOB_IN, d), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, data):
        is_c = data.is_char
        hc, bc = self.phi(data.x[is_c]), data.batch[is_c]
        z = torch.cat([self.g(data.gfeat), global_mean_pool(hc, bc), global_max_pool(hc, bc)], -1)
        return self.head(z).squeeze(-1)


class TransformerNet(nn.Module):
    """Fully-connected attention over the same nodes (no explicit relational structure)."""

    def __init__(self, d=64, layers=2, dropout=0.2):
        super().__init__()
        self.inp_c = nn.Linear(NODE_IN, d)
        self.inp_g = nn.Linear(GLOB_IN, d)
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, data):
        from torch_geometric.utils import to_dense_batch
        is_c = data.is_char
        xc, mask = to_dense_batch(self.inp_c(data.x[is_c]), data.batch[is_c])
        g = self.inp_g(data.gfeat).unsqueeze(1)
        x = torch.cat([g, xc], 1)
        m = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        h = self.enc(x, src_key_padding_mask=~m)
        return self.head(h[:, 0]).squeeze(-1)


class LineGraphGCN(nn.Module):
    """Prior work (NCCI'26): 3-layer GCN 70->128->128->128, BN, ReLU, dropout 0.5 on the
    first two blocks, mean pooling, linear head."""

    def __init__(self, inp=70, d=128):
        super().__init__()
        self.c = nn.ModuleList([GCNConv(inp, d), GCNConv(d, d), GCNConv(d, d)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(d) for _ in range(3)])
        self.out = nn.Linear(d, 2)

    def forward(self, data):
        h = data.x
        for i, (c, b) in enumerate(zip(self.c, self.bn)):
            h = F.relu(b(c(h, data.edge_index)))
            if i < 2:
                h = F.dropout(h, 0.5, self.training)
        return self.out(global_mean_pool(h, data.batch))

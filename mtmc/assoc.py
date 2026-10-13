"""Cross-camera tracklet association: graph construction, LSTM tracklet encoder,
(transition-aware) graph attention, edge classification, constrained identity clustering.

Pipeline for one scene
  tracklets -> candidate pairs (different cameras; optional same-camera re-entry)
            -> spatio-temporal feasibility mask (time-gap window learned from TRAIN positives)
            -> appearance threshold tau on fixed input embeddings (=> edge count is monotone in tau)
            -> node encoder  : 'mean' | 'lstm' | 'gru' over the time-ordered frame embeddings
            -> relational    : 'none' | 'gcn' | 'gat' | 'tgat' (GATv2 + edge features: time gap,
                                overlap, cosine, learned camera-pair embedding; no calibration)
            -> edge classifier p_ij -> constrained agglomerative clustering (cannot-link: same camera
               and overlapping in time) -> global identities
"""
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import GATv2Conv, GCNConv

NUM_EDGE = 6          # numeric edge features (see edge_features)
CAMPAIR_DIM = 8


# ============================================================================ graph
def time_gap(a, b):
    """>0: seconds between the two tracklets; <0: they overlap in time by that many seconds."""
    return max(b["t0"] - a["t1"], a["t0"] - b["t1"])


def campair_key(ca, cb):
    return "|".join(sorted([ca, cb]))


def learn_gap_window(train_tracklets_by_scene, q=(0.005, 0.995), margin=1.25):
    """Data-driven feasibility window [lo, hi] (seconds) from true cross-camera transitions in TRAIN scenes."""
    gaps = []
    for trs in train_tracklets_by_scene.values():
        by = defaultdict(list)
        for t in trs:
            by[t["gt"]].append(t)
        for lst in by.values():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if lst[i]["cam"] != lst[j]["cam"]:
                        gaps.append(time_gap(lst[i], lst[j]))
    if not gaps:
        return -1e9, 1e9
    lo, hi = np.quantile(gaps, q)
    return float(lo * margin if lo < 0 else lo / margin), float(hi * margin)


def candidate_pairs(trs, window=None, same_cam=False, same_cam_max_gap=5.0):
    """All pairs that are allowed BEFORE appearance thresholding. Returns arrays i, j, gap, same."""
    n = len(trs)
    if n < 2:
        z = np.zeros(0)
        return z.astype(np.int64), z.astype(np.int64), z.astype(np.float32), z.astype(np.float32)
    I, J = np.triu_indices(n, k=1)
    t0 = np.array([t["t0"] for t in trs])
    t1 = np.array([t["t1"] for t in trs])
    cams = np.array([t["cam"] for t in trs])
    G = np.maximum(t0[J] - t1[I], t0[I] - t1[J])
    S = cams[I] == cams[J]
    ok_same = S & same_cam & (G >= 0) & (G <= same_cam_max_gap)
    ok_cross = ~S
    if window is not None:
        ok_cross &= (G >= window[0]) & (G <= window[1])
    k = ok_same | ok_cross
    return I[k].astype(np.int64), J[k].astype(np.int64), G[k].astype(np.float32), S[k].astype(np.float32)


def topk_mask(n, I, J, score, k):
    """True for edges that are among the k highest-scoring edges of at least one endpoint (vectorised)."""
    if k is None or len(I) == 0:
        return np.ones(len(I), bool)
    node = np.concatenate([I, J])
    eid = np.concatenate([np.arange(len(I)), np.arange(len(I))])
    sc = np.concatenate([score, score])
    order = np.lexsort((-sc, node))                  # by node, then score descending
    node_s = node[order]
    start = np.searchsorted(node_s, node_s, side="left")
    rank = np.arange(len(order)) - start
    ok = np.zeros(len(I), bool)
    ok[eid[order][rank < k]] = True
    return ok


def build_graph(trs, tau, window=None, same_cam=False, campair_vocab=None, knn=20):
    """Graph over feasible pairs: each tracklet keeps its knn most similar feasible candidates (fixed set,
    independent of tau), then edges with input cosine <= tau are removed -> edge count is monotone in tau
    and memory stays bounded (at most knn * n_tracklets edges)."""
    M = np.stack([t["mean"] for t in trs]).astype(np.float32) if trs else np.zeros((0, 512), np.float32)
    I, J, G, S = candidate_pairs(trs, window, same_cam)
    n_candidates_all = len(trs) * (len(trs) - 1) // 2
    cos = (M @ M.T)[I, J].astype(np.float32) if len(I) else np.zeros(0, np.float32)
    keep = (cos > tau) & topk_mask(len(trs), I, J, cos, knn)
    I, J, G, S, cos = I[keep], J[keep], G[keep], S[keep], cos[keep]
    feat = edge_features(cos, G, S, [trs[i]["n"] for i in I], [trs[j]["n"] for j in J])
    vocab = campair_vocab or {}
    cp = np.array([vocab.get(campair_key(trs[i]["cam"], trs[j]["cam"]), 0) for i, j in zip(I, J)],
                  dtype=np.int64)
    y = np.array([float(trs[i]["gt"] == trs[j]["gt"]) for i, j in zip(I, J)], dtype=np.float32)
    return {"I": I, "J": J, "gap": G, "cos": cos, "feat": feat, "campair": cp, "y": y,
            "n_nodes": len(trs), "n_candidates_all": n_candidates_all, "n_feasible": int(keep.size)}


def edge_features(cos, gap, same, ni, nj):
    gap = np.asarray(gap, np.float32)
    return np.stack([
        np.asarray(cos, np.float32),
        np.clip(gap / 60.0, -2, 5),
        np.clip(np.abs(gap) / 60.0, 0, 5),
        (gap < 0).astype(np.float32),
        np.asarray(same, np.float32),
        np.log1p(np.minimum(ni, nj)).astype(np.float32) / 5.0,
    ], 1).astype(np.float32) if len(cos) else np.zeros((0, NUM_EDGE), np.float32)


def build_campair_vocab(train_tracklets_by_scene):
    keys = set()
    for trs in train_tracklets_by_scene.values():
        cams = sorted({t["cam"] for t in trs})
        for a in range(len(cams)):
            for b in range(a, len(cams)):
                keys.add(campair_key(cams[a], cams[b]))
    return {k: i + 1 for i, k in enumerate(sorted(keys))}          # 0 = unseen camera pair


def node_tensors(trs, device):
    T = max(len(t["seq"]) for t in trs)
    N, D = len(trs), trs[0]["seq"].shape[1]
    seq = torch.zeros(N, T, D)
    st = torch.zeros(N, T, 1)
    lens = torch.zeros(N, dtype=torch.long)
    for k, t in enumerate(trs):
        L = len(t["seq"])
        seq[k, :L] = torch.from_numpy(t["seq"])
        st[k, :L, 0] = torch.from_numpy(t["seq_t"]) / 10.0
        lens[k] = L
    mean = torch.from_numpy(np.stack([t["mean"] for t in trs]))
    return seq.to(device), st.to(device), lens, mean.to(device)


# ============================================================================ model
class SeqEncoder(nn.Module):
    """Tracklet encoder over the TIME-ORDERED frame embeddings of a tracklet (<=16 uniformly sampled frames).
    'lstm'/'gru': recurrent model, input = 512-D appearance + time since tracklet start.
    'attn'      : temporal self-attention over the frames (+ time embedding), masked mean pooling.
    'mean'      : plain average (no temporal model)."""

    def __init__(self, kind="lstm", d=512, h=256, dropout=0.1):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = rnn(d + 1, h, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * h, d)
        elif kind == "attn":
            # temporal self-attention fusion (the module used in the V15 notebook), with a time embedding
            self.t_emb = nn.Linear(1, d)
            self.attn = nn.MultiheadAttention(d, num_heads=4, dropout=dropout, batch_first=True)
            self.attn_norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, d)
        elif kind == "mean":
            self.out = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        else:
            raise ValueError(kind)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq, st, lens, mean):
        if self.kind == "mean":
            delta = self.out(mean)
        elif self.kind == "attn":
            T = seq.shape[1]
            pad = torch.arange(T, device=seq.device)[None, :] >= lens.to(seq.device)[:, None]   # True = padding
            x = seq + self.t_emb(st)
            a, _ = self.attn(x, x, x, key_padding_mask=pad)
            a = self.attn_norm(x + a).masked_fill(pad[..., None], 0.0)
            delta = self.out(a.sum(1) / lens.to(seq.device)[:, None].clamp(min=1))
        else:
            packed = pack_padded_sequence(torch.cat([seq, st], -1), lens.cpu(), batch_first=True,
                                          enforce_sorted=False)
            _, hN = self.rnn(packed)
            hN = hN[0] if self.kind == "lstm" else hN
            delta = self.out(torch.cat([hN[-2], hN[-1]], -1))
        return self.norm(mean + self.drop(delta))


class Relational(nn.Module):
    def __init__(self, kind="tgat", d=512, heads=4, layers=2, dropout=0.1, n_campairs=1):
        super().__init__()
        self.kind = kind
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        if kind == "tgat":
            self.cp_emb = nn.Embedding(n_campairs + 1, CAMPAIR_DIM)
        for _ in range(layers if kind != "none" else 0):
            if kind == "gcn":
                self.layers.append(GCNConv(d, d))
            elif kind == "gat":
                self.layers.append(GATv2Conv(d, d // heads, heads=heads, dropout=dropout))
            elif kind == "tgat":
                self.layers.append(GATv2Conv(d, d // heads, heads=heads, dropout=dropout,
                                             edge_dim=NUM_EDGE + CAMPAIR_DIM))
            else:
                raise ValueError(kind)
            self.norms.append(nn.LayerNorm(d))
        self.drop = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, x, ei, efeat, ecp):
        for k, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            if self.kind == "gcn":
                h = conv(x, ei)
            elif self.kind == "gat":
                h, att = conv(x, ei, return_attention_weights=True)
                self.last_attention = att
            else:
                ea = torch.cat([efeat, self.cp_emb(ecp)], -1)
                h, att = conv(x, ei, edge_attr=ea, return_attention_weights=True)
                self.last_attention = att
            x = norm(x + self.drop(F.elu(h)))
        return x


class AssocNet(nn.Module):
    def __init__(self, enc="lstm", gnn="tgat", d=512, n_campairs=1, dropout=0.1):
        super().__init__()
        self.enc = SeqEncoder(enc, d, dropout=dropout)
        self.gnn = Relational(gnn, d, dropout=dropout, n_campairs=n_campairs)
        self.head = nn.Sequential(nn.Linear(2 * d + NUM_EDGE, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 1))
        # Residual over appearance similarity: logit = s * (cos - b) + correction(context).
        # The correction starts at exactly zero, so an untrained model ranks pairs like the cosine baseline.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.cos_scale = nn.Parameter(torch.tensor(10.0))
        self.cos_bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, nodes, g, device, edge_drop=0.0):
        seq, st, lens, mean = nodes
        x = self.enc(seq, st, lens, mean)
        I = torch.as_tensor(g["I"], device=device)
        J = torch.as_tensor(g["J"], device=device)
        feat = torch.as_tensor(g["feat"], device=device)
        cp = torch.as_tensor(g["campair"], device=device)
        mp = torch.ones(len(I), dtype=torch.bool, device=device)
        if edge_drop > 0 and len(I):
            mp = torch.rand(len(I), device=device) > edge_drop
        ei = torch.stack([torch.cat([I[mp], J[mp]]), torch.cat([J[mp], I[mp]])])     # both directions
        z = self.gnn(x, ei, torch.cat([feat[mp], feat[mp]]), torch.cat([cp[mp], cp[mp]]))
        zi, zj = z[I], z[J]
        prior = self.cos_scale * (feat[:, 0] - self.cos_bias)
        return prior + self.head(torch.cat([zi * zj, (zi - zj).abs(), feat], -1)).squeeze(-1)


# ============================================================================ train / infer
def train_assoc(train_data, cfg, device="cuda", log=print, val_fn=None, eval_every=10):
    """train_data: list of (trs, graph). cfg keys: enc, gnn, epochs, lr, wd, edge_drop, n_campairs, seed.
    val_fn(net) -> score on the TUNING scene (S01). If given, the best checkpoint (epoch 0 included, i.e. the
    cosine-equivalent model) is kept: early stopping on validation data, never on test scenes."""
    import copy
    from .common import set_seed
    set_seed(cfg.get("seed", 0))
    net = AssocNet(cfg["enc"], cfg["gnn"], n_campairs=cfg.get("n_campairs", 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.get("epochs", 60))
    cached = [(node_tensors(trs, device), g) for trs, g in train_data if len(g["I"])]
    best = (val_fn(net), 0, copy.deepcopy(net.state_dict())) if val_fn else None
    for ep in range(cfg.get("epochs", 60)):
        net.train()
        tot = 0.0
        for nodes, g in cached:
            y = torch.as_tensor(g["y"], device=device)
            pw = torch.clamp((1 - y).sum() / y.sum().clamp(min=1), max=20.0)
            seq, st, lens, mean = nodes
            noisy = (seq + 0.02 * torch.randn_like(seq), st, lens, F.normalize(mean + 0.02 * torch.randn_like(mean), dim=1))
            logit = net(noisy, g, device, edge_drop=cfg.get("edge_drop", 0.1))
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        sched.step()
        if (ep + 1) % eval_every == 0:
            msg = f"  [assoc {cfg['enc']}/{cfg['gnn']} seed{cfg.get('seed', 0)}] epoch {ep + 1} loss {tot / max(1, len(cached)):.4f}"
            if val_fn:
                v = val_fn(net)
                msg += f" val {v:.4f}"
                if v > best[0]:
                    best = (v, ep + 1, copy.deepcopy(net.state_dict()))
            log(msg)
    if val_fn:
        net.load_state_dict(best[2])
        net.best_epoch, net.best_val = best[1], best[0]
        log(f"  [assoc {cfg['enc']}/{cfg['gnn']} seed{cfg.get('seed', 0)}] kept epoch {best[1]} (S01 IDF1 {best[0]:.4f})")
    return net


@torch.no_grad()
def predict_edges(net, trs, g, device="cuda"):
    if len(g["I"]) == 0:
        return np.zeros(0, np.float32)
    if net is None:                       # appearance-only baseline: cosine as the score
        return g["cos"].astype(np.float32)
    net.eval()
    return torch.sigmoid(net(node_tensors(trs, device), g, device)).cpu().numpy()


def _conflict(trs, A, B):
    for a in A:
        ta = trs[a]
        for b in B:
            tb = trs[b]
            if ta["cam"] == tb["cam"] and ta["t0"] <= tb["t1"] and tb["t0"] <= ta["t1"]:
                return True
    return False


def cluster(trs, g, p, theta, constrained=True, linkage="average"):
    """Agglomerative identity clustering over the graph edges.
    linkage="single" : merge along any edge with p >= theta (highest first). Prone to chaining: one wrong edge
                       joins two whole identities, which produces giant clusters in large scenes.
    linkage="average": merge the two clusters whose AVERAGE association probability over all their cross-camera
                       tracklet pairs is highest, while it is >= theta (pairs without an edge count as 0).
                       Robust to single wrong edges.
    Cannot-link (constrained=True): two tracklets of the same camera that overlap in time never share an identity."""
    if linkage == "single":
        return _cluster_single(trs, g, p, theta, constrained)
    import heapq
    from collections import Counter
    n = len(trs)
    members = {i: [i] for i in range(n)}
    camcount = {i: Counter([trs[i]["cam"]]) for i in range(n)}
    S = {i: {} for i in range(n)}                 # S[a][b] = summed p over edges between clusters a and b
    for e in range(len(p)):
        a, b = int(g["I"][e]), int(g["J"][e])
        if a == b:
            continue
        S[a][b] = S[a].get(b, 0.0) + float(p[e])
        S[b][a] = S[b].get(a, 0.0) + float(p[e])

    def cross(a, b):
        na, nb = len(members[a]), len(members[b])
        same = sum(c * camcount[b].get(cam, 0) for cam, c in camcount[a].items())
        return max(na * nb - same, 1)

    version = {i: 0 for i in range(n)}
    heap = []
    for a in range(n):
        for b, sv in S[a].items():
            if a < b:
                sc = sv / cross(a, b)
                if sc >= theta:
                    heap.append((-sc, a, b, 0, 0))
    heapq.heapify(heap)
    accepted = 0
    while heap:
        neg, a, b, va, vb = heapq.heappop(heap)
        if a not in members or b not in members or version[a] != va or version[b] != vb:
            continue
        if constrained and _conflict(trs, members[a], members[b]):
            continue
        if len(members[a]) < len(members[b]):
            a, b = b, a
        members[a].extend(members.pop(b))
        camcount[a].update(camcount.pop(b))
        for c, sv in S.pop(b).items():
            if c == a:
                continue
            S[a][c] = S[a].get(c, 0.0) + sv
            S[c][a] = S[c].get(a, 0.0) + sv
            S[c].pop(b, None)
        S[a].pop(b, None)
        version[a] += 1
        accepted += 1
        for c, sv in S[a].items():
            if c in members:
                sc = sv / cross(a, c)
                if sc >= theta:
                    x, y = (a, c) if a < c else (c, a)
                    heapq.heappush(heap, (-sc, x, y, version[x], version[y]))
    gid = np.empty(n, dtype=np.int64)
    for k, (root, mem) in enumerate(members.items()):
        gid[mem] = k
    return gid, accepted


def _cluster_single(trs, g, p, theta, constrained=True):
    n = len(trs)
    parent = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = np.argsort(-p)
    accepted = 0
    for e in order:
        if p[e] < theta:
            break
        ra, rb = find(int(g["I"][e])), find(int(g["J"][e]))
        if ra == rb:
            continue
        if constrained and _conflict(trs, members[ra], members[rb]):
            continue
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra].extend(members.pop(rb))
        accepted += 1
    roots = {}
    gid = np.array([roots.setdefault(find(i), len(roots)) for i in range(n)], dtype=np.int64)
    return gid, accepted

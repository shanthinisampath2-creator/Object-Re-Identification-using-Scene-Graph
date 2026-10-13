"""Controlled experiments for the revision (CityFlow).

Setting B (GT boxes, isolates cross-camera association)
  appearance backbone : trained on TRAIN-scene crops (S01, S03, S04)
  association model   : trained on S03 + S04 (GT tracklets)
  tau, theta          : selected on S01 (max IDF1)          -> no test-set tuning
  report              : S02, S05, OVERALL (exact MTMC metrics, equal to motmetrics in this setting)
Setting C (detections, end-to-end MTMCT)
  provided detections -> ROI/score filter -> Kalman + Hungarian tracklets -> SAME trained association models
  -> constrained clustering -> keep identities seen in >= 2 cameras -> official-style motmetrics
  tau, theta selected on S01 detections; report S02, S05, OVERALL incl. FN/FP/IDSW/GT.
Every learned variant uses the same epochs, optimiser, edge head and seeds -> one factor changes per row.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .assoc import (AssocNet, build_campair_vocab, build_graph, cluster, learn_gap_window, predict_edges,
                    train_assoc)
from .evaluate import (fast_gtbox_metrics, fast_idf1, fast_idf1_det, filter_multicam, graph_stats,
                       gt_box_count, mtmc_motmetrics, per_camera_motmetrics, predictions_from_tracklets,
                       tracklet_level, write_aic)
from .extract import load_scene_cache
from .tracklets import build_det_tracklets, build_tracklets

VARIANTS = {
    # name               temporal encoder  relational  feasibility-mask  constrained-clustering
    "cosine":            dict(enc=None,   gnn=None,   mask=True,  constrained=True),
    "pair_mlp":          dict(enc="mean", gnn="none", mask=True,  constrained=True),
    "lstm_only":         dict(enc="lstm", gnn="none", mask=True,  constrained=True),
    "attn_only":         dict(enc="attn", gnn="none", mask=True,  constrained=True),
    "gcn":               dict(enc="mean", gnn="gcn",  mask=True,  constrained=True),
    "gat":               dict(enc="mean", gnn="gat",  mask=True,  constrained=True),
    "gat_lstm":          dict(enc="lstm", gnn="gat",  mask=True,  constrained=True),
    "gat_attn":          dict(enc="attn", gnn="gat",  mask=True,  constrained=True),   # closest to V15
    "tgat":              dict(enc="mean", gnn="tgat", mask=True,  constrained=True),
    "tgat_gru":          dict(enc="gru",  gnn="tgat", mask=True,  constrained=True),
    "tgat_lstm":         dict(enc="lstm", gnn="tgat", mask=True,  constrained=True),   # proposed, LSTM version
    "tgat_attn":         dict(enc="attn", gnn="tgat", mask=True,  constrained=True),   # proposed, attention version
    "tgat_lstm_nomask":  dict(enc="lstm", gnn="tgat", mask=False, constrained=True),
    "tgat_lstm_cc":      dict(enc="lstm", gnn="tgat", mask=True,  constrained=False),
    # previous clustering (single linkage, prone to chaining) for the ablation
    "cosine_single":     dict(enc=None,   gnn=None,   mask=True,  constrained=True, linkage="single"),
    "tgat_lstm_single":  dict(enc="lstm", gnn="tgat", mask=True,  constrained=True, linkage="single"),
    # the same three ablations for the attention-pooling version
    "tgat_attn_nomask":  dict(enc="attn", gnn="tgat", mask=False, constrained=True),
    "tgat_attn_cc":      dict(enc="attn", gnn="tgat", mask=True,  constrained=False),
    "tgat_attn_single":  dict(enc="attn", gnn="tgat", mask=True,  constrained=True, linkage="single"),
}
# Identical decision grids for every method (fair comparison): the cosine baseline thresholds cosine on COS_GRID;
# learned models threshold the association probability at the SAME points mapped through the cosine prior
# sigmoid(10 (cos - 0.5)). An untrained learned model therefore reproduces the cosine baseline exactly.
COS_GRID = tuple(float(c) for c in np.round(np.arange(0.20, 0.991, 0.02), 2))
THETAS = tuple(float(1 / (1 + np.exp(-10 * (c - 0.5)))) for c in COS_GRID)
CORE = ["cosine", "cosine_single", "gat", "gat_lstm", "gat_attn", "tgat", "tgat_lstm", "tgat_attn",
        "tgat_lstm_nomask", "tgat_lstm_cc", "tgat_lstm_single"]
EXTRA = ["pair_mlp", "lstm_only", "attn_only", "gcn", "tgat_gru"]


class Data:
    """Loads cached embeddings once and builds tracklets for every scene."""

    def __init__(self, cache_dir, scenes, tracklet_mode="gt", seq_len=16, det_kw=None):
        kind = "det" if tracklet_mode == "det" else "gt"
        self.cache = {s: load_scene_cache(cache_dir, s, kind) for s in scenes}
        if tracklet_mode == "det":
            self.trs = {s: build_det_tracklets(s, self.cache[s], seq_len=seq_len, **(det_kw or {})) for s in scenes}
        else:
            self.trs = {s: build_tracklets(s, self.cache[s], mode=tracklet_mode, seq_len=seq_len) for s in scenes}


def context(gt_data, train_scenes=("S03", "S04")):
    """Feasibility window and camera-pair vocabulary, learned from TRAIN-scene GT tracklets only."""
    sub = {s: gt_data.trs[s] for s in train_scenes}
    return learn_gap_window(sub), build_campair_vocab(sub)


def _graph(trs, tau, v, window, vocab, same_cam):
    return build_graph(trs, tau, window=window if v["mask"] else None, same_cam=same_cam, campair_vocab=vocab)


def _load_net(name, seed, model_dir, vocab, device):
    v = VARIANTS[name]
    if v["enc"] is None:
        return None
    p = Path(model_dir) / f"assoc_{name}_seed{seed}.pt"
    net = AssocNet(v["enc"], v["gnn"], n_campairs=len(vocab)).to(device)
    net.load_state_dict(torch.load(p, map_location=device))
    net.eval()
    return net


# ============================================================================ Setting B
CV_FOLDS = {"S02": ("S05",), "S05": ("S02",)}    # test scene -> scene(s) used for epoch / tau / theta selection


def _fold_tag(tune):
    return "tune" + "".join(tune)


def _select(net, trs_d, cache_d, tune, taus, thetas, v, window, vocab, same_cam, device):
    """tau/theta maximising IDF1 on the tuning scene(s)."""
    best = (-1, None, None)
    for tau in taus:
        gs = {s: _graph(trs_d[s], tau, v, window, vocab, same_cam) for s in tune}
        ps = {s: predict_edges(net, trs_d[s], gs[s], device) for s in tune}
        for th in thetas:
            f = np.mean([fast_idf1(trs_d[s], cluster(trs_d[s], gs[s], ps[s], th, v["constrained"], v.get("linkage", "average"))[0],
                                   cache_d[s])["idf1"] for s in tune])
            if f > best[0]:
                best = (f, tau, th)
    return best


def run_variant(data, name, seeds=(0, 1, 2), train_scenes=("S03", "S04"), tune_scenes=("S01",),
                test_scenes=("S02", "S05"), taus=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
                thetas=THETAS, tau_train=None, epochs=60, folds=None,
                same_cam=False, device="cuda", out_dir="results", gt_data=None, log=print):
    """folds: {test_scene: tuning scenes}. Default = CV_FOLDS (2-fold cross-validation over the unseen scenes:
    early-stopping epoch, tau and theta for S02 are chosen on S05 and vice versa). Never tuned on the test scene."""
    v = VARIANTS[name]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds = folds or {s: tuple(tune_scenes) for s in test_scenes}
    window, vocab = context(gt_data or data, train_scenes)
    tau_train = min(taus) if tau_train is None else tau_train
    rows = []
    for seed in (seeds if v["enc"] is not None else (0,)):
        t_start = time.time()
        rec = {"variant": name, "seed": seed}
        tot = {"idtp": 0.0, "n": 0, "sw": 0}
        tune_scores = []
        nets = {}
        for s in test_scenes:
            tune = folds[s]
            tag = _fold_tag(tune)
            net = nets.get(tag)
            if v["enc"] is not None and net is None:
                train_data = [(data.trs[t], _graph(data.trs[t], tau_train, v, window, vocab, same_cam)) for t in train_scenes]
                vg = {t: _graph(data.trs[t], tau_train, v, window, vocab, same_cam) for t in tune}

                def val_fn(model, tune=tune, vg=vg):   # early stopping on the tuning scene(s) only
                    return max(np.mean([fast_idf1(data.trs[t], cluster(data.trs[t], vg[t], predict_edges(model, data.trs[t], vg[t], device),
                                                                       th, v["constrained"], v.get("linkage", "average"))[0], data.cache[t])["idf1"]
                                        for t in tune]) for th in thetas)

                net = train_assoc(train_data, dict(enc=v["enc"], gnn=v["gnn"], epochs=epochs, seed=seed,
                                                   n_campairs=len(vocab)), device=device, log=log, val_fn=val_fn)
                torch.save(net.state_dict(), out / f"assoc_{name}_seed{seed}_{tag}.pt")
                nets[tag] = net
            f_t, tau, th = _select(net, data.trs, data.cache, tune, taus, thetas if net is not None else COS_GRID,
                                   v, window, vocab, same_cam, device)
            tune_scores.append(f_t)
            rec.update({f"{s}_tau": tau, f"{s}_theta": th, f"{s}_tune_idf1": f_t, f"{s}_tuned_on": "+".join(tune),
                        f"{s}_best_epoch": getattr(net, "best_epoch", None)})
            g = _graph(data.trs[s], tau, v, window, vocab, same_cam)
            gid, _ = cluster(data.trs[s], g, predict_edges(net, data.trs[s], g, device), th, v["constrained"], v.get("linkage", "average"))
            write_aic(predictions_from_tracklets(data.trs[s], gid, data.cache[s]), out / f"pred_{name}_seed{seed}_{s}.txt")
            m = fast_gtbox_metrics(data.trs[s], gid, data.cache[s])
            for k in ("idf1", "idp", "idr", "mota", "num_switches", "num_misses", "num_false_positives", "num_objects"):
                rec[f"{s}_{k}"] = m[k]
            tot["idtp"] += m["idf1"] * m["n_boxes"]
            tot["n"] += m["n_boxes"]
            tot["sw"] += m["num_switches"]
            tl = tracklet_level(data.trs[s], g, gid)
            rec.update({f"{s}_{k}": tl[k] for k in ("FR", "MER", "pair_precision", "pair_recall")})
            gsx = graph_stats(data.trs[s], g, tau)
            rec.update({f"{s}_{k}": gsx[k] for k in ("edges_kept", "components", "max_comp_size")})
        rec["tune_idf1"] = float(np.mean(tune_scores))
        eps = [rec.get(f"{s}_best_epoch") for s in test_scenes]
        rec["best_epoch"] = "/".join(str(e) for e in eps) if v["enc"] is not None else None
        rec["OVERALL_idf1"] = tot["idtp"] / max(tot["n"], 1)
        rec["OVERALL_idp"] = rec["OVERALL_idr"] = rec["OVERALL_idf1"]
        rec["OVERALL_num_switches"] = tot["sw"]
        rec["OVERALL_num_objects"] = tot["n"]
        rec["OVERALL_mota"] = 1 - tot["sw"] / max(tot["n"], 1)
        rec["minutes"] = (time.time() - t_start) / 60
        log(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items()}))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"runs_{name}.csv", index=False)
    return df


def _load_fold_net(name, seed, model_dir, vocab, device, tune):
    v = VARIANTS[name]
    if v["enc"] is None:
        return None
    p = Path(model_dir) / f"assoc_{name}_seed{seed}_{_fold_tag(tune)}.pt"
    net = AssocNet(v["enc"], v["gnn"], n_campairs=len(vocab)).to(device)
    net.load_state_dict(torch.load(p, map_location=device))
    net.eval()
    return net


def tau_sweep(data, name, seed, runs_row, model_dir, train_scenes=("S03", "S04"), test_scenes=("S02", "S05"),
              taus=tuple(np.round(np.arange(0.1, 0.91, 0.05), 2)), folds=None, same_cam=False, device="cuda",
              out_csv=None, log=print):
    """Sensitivity analysis: the trained Setting-B model of each fold, FIXED embeddings, theta fixed to the value
    selected for that fold; only tau changes -> edges kept decrease monotonically by construction."""
    v = VARIANTS[name]
    folds = folds or CV_FOLDS
    window, vocab = context(data, train_scenes)
    rows = []
    for s in test_scenes:
        net = _load_fold_net(name, seed, model_dir, vocab, device, folds[s])
        theta = float(runs_row[f"{s}_theta"])
        for tau in taus:
            g = _graph(data.trs[s], tau, v, window, vocab, same_cam)
            gid, _ = cluster(data.trs[s], g, predict_edges(net, data.trs[s], g, device), theta, v["constrained"], v.get("linkage", "average"))
            rows.append({"variant": name, "scene": s, "theta": theta, **graph_stats(data.trs[s], g, tau),
                         **fast_idf1(data.trs[s], gid, data.cache[s]), **tracklet_level(data.trs[s], g, gid)})
    df = pd.DataFrame(rows).sort_values(["scene", "tau"])
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


# ============================================================================ Setting C
def run_setting_c(det_data, gt_data, root, names, seeds, model_dir, out_dir, train_scenes=("S03", "S04"),
                  tune_scene=None, test_scenes=("S02", "S05"), taus=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                  thetas=THETAS, folds=None, device="cuda", log=print):
    """Same 2-fold protocol as Setting B: tau/theta for S02 chosen on S05 detections and vice versa, using the
    Setting-B association model of the same fold."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds = folds or CV_FOLDS
    window, vocab = context(gt_data, train_scenes)
    n_gt = {s: gt_box_count(root, s) for s in set(test_scenes) | {t for f in folds.values() for t in f}}
    rows = []
    for name in names:
        v = VARIANTS[name]
        for seed in (seeds if v["enc"] is not None else (0,)):
            t0 = time.time()
            preds = {}
            rec = {"variant": name, "seed": seed}
            for s in test_scenes:
                tune = folds[s]
                net = _load_fold_net(name, seed, model_dir, vocab, device, tune)
                best = (-1, None, None)
                for tau in taus:
                    gts = {t: _graph(det_data.trs[t], tau, v, window, vocab, False) for t in tune}
                    pts = {t: predict_edges(net, det_data.trs[t], gts[t], device) for t in tune}
                    for th in (thetas if net is not None else COS_GRID):
                        f = np.mean([fast_idf1_det(det_data.trs[t], filter_multicam(det_data.trs[t], cluster(
                            det_data.trs[t], gts[t], pts[t], th, v["constrained"], v.get("linkage", "average"))[0]), det_data.cache[t], n_gt[t])
                            for t in tune])
                        if f > best[0]:
                            best = (f, tau, th)
                _, tau, th = best
                rec.update({f"{s}_tau": tau, f"{s}_theta": th, f"{s}_tune_idf1": best[0]})
                trs = det_data.trs[s]
                g = _graph(trs, tau, v, window, vocab, False)
                gid = filter_multicam(trs, cluster(trs, g, predict_edges(net, trs, g, device), th, v["constrained"], v.get("linkage", "average"))[0])
                preds[s] = predictions_from_tracklets(trs, gid, det_data.cache[s])
                write_aic(preds[s], out / f"pred_det_{name}_seed{seed}_{s}.txt")
                tl = tracklet_level(trs, g, gid)
                rec.update({f"{s}_{k}": tl[k] for k in ("FR", "MER")})
            mm_df = mtmc_motmetrics(root, list(test_scenes), preds)
            for sc, r in mm_df.iterrows():
                for k in ("idf1", "idp", "idr", "mota", "num_switches", "num_misses", "num_false_positives",
                          "num_objects", "num_predictions"):
                    rec[f"{sc}_{k}"] = float(r[k])
            rec["minutes"] = (time.time() - t0) / 60
            log(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items()}))
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out / "runs_setting_c.csv", index=False)
    return pd.DataFrame(rows)


def single_camera_quality(det_data, root, scenes=("S02", "S05")):
    """Quality of the Kalman + Hungarian single-camera tracker alone (per camera, before any cross-camera link).
    Note: AI City GT only labels vehicles seen by >= 2 cameras, so per-camera FP counts include real but
    unlabelled vehicles."""
    rows = []
    for s in scenes:
        trs = det_data.trs[s]
        gid = np.arange(len(trs))
        pdf = predictions_from_tracklets(trs, gid, det_data.cache[s])
        pc = per_camera_motmetrics(root, s, pdf)
        pc["scene"] = s
        rows.append(pc)
    return pd.concat(rows)


def summarize(df, cols=("OVERALL_idf1", "OVERALL_idp", "OVERALL_idr", "OVERALL_mota", "OVERALL_num_switches",
                        "S02_idf1", "S05_idf1", "S02_FR", "S05_FR", "S02_MER", "S05_MER")):
    cols = [c for c in cols if c in df.columns]
    g = df.groupby("variant")[cols]
    mean, std = g.mean(), g.std().fillna(0)
    out = mean.copy().astype(object)
    for c in cols:
        out[c] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean[c], std[c])]
    order = [v for v in VARIANTS if v in out.index]
    out = out.loc[order]
    out.insert(0, "runs", df.groupby("variant").size().loc[order].values)
    return out

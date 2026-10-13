"""Evaluation for cross-camera association on CityFlow.

* mtmc_motmetrics : official-style MTMC evaluation. All cameras of a scene go into ONE accumulator
                    (frame key = camera*100000 + frame), so a global identity must be consistent
                    ACROSS cameras to score (the AI City Track 3 protocol). Reports IDF1/IDP/IDR,
                    MOTA with all its components (FN, FP, IDSW, GT), so every number is auditable.
* per_camera_motmetrics : the OLD protocol (each camera separately). Kept only to show why it cannot
                    measure cross-camera association.
* fast_idf1       : exact IDF1 when predictions are the GT boxes (Setting B); used for tuning.
* tracklet_level  : FR, MER, pairwise precision/recall with explicit definitions.
"""
from collections import Counter, defaultdict

import numpy as np

if not hasattr(np, "asfarray"):            # motmetrics 1.4 + NumPy 2 compatibility (Colab ships NumPy 2)
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
import motmetrics as mm                    # noqa: E402
import pandas as pd                        # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from .common import cam_int, read_gt, scene_meta  # noqa: E402
from .tracklets import iou_xywh            # noqa: E402

MM_METRICS = ["idf1", "idp", "idr", "idtp", "idfp", "idfn", "mota", "motp", "num_objects",
              "num_predictions", "num_misses", "num_false_positives", "num_switches",
              "num_fragmentations", "num_unique_objects", "mostly_tracked"]


# ----------------------------------------------------------------------------- predictions
def predictions_from_tracklets(trs, gid, cache):
    """One prediction row for EVERY box of EVERY tracklet: cam frame id x y w h (AI City format)."""
    rows = []
    for t, g in zip(trs, gid):
        if g < 0:                     # removed by filter_multicam
            continue
        d = cache[t["cam"]]
        c = cam_int(t["cam"])
        for r in t["rows"]:
            x, y, w, h = d["xywh"][r]
            rows.append((c, int(d["frame"][r]), int(g), float(x), float(y), float(w), float(h)))
    return pd.DataFrame(rows, columns=["cam", "frame", "id", "x", "y", "w", "h"])


def read_aic(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=range(7),
                     names=["cam", "frame", "id", "x", "y", "w", "h"])
    return df


def write_aic(pred_df, path):
    with open(path, "w") as f:
        for r in pred_df.itertuples(index=False):
            f.write(f"{r.cam} {r.frame} {r.id} {int(round(r.x))} {int(round(r.y))} {int(round(r.w))} {int(round(r.h))} -1 -1\n")


# ----------------------------------------------------------------------------- metrics
def fast_idf1(trs, gid, cache):
    """Exact identity metrics when every GT box carries exactly one prediction (GT-box setting)."""
    gt_ids, pr_ids = [], []
    for t, g in zip(trs, gid):
        gt_ids.append(cache[t["cam"]]["gid"][t["rows"]])
        pr_ids.append(np.full(len(t["rows"]), g))
    gt_ids, pr_ids = np.concatenate(gt_ids), np.concatenate(pr_ids)
    gu, gi = np.unique(gt_ids, return_inverse=True)
    pu, pi = np.unique(pr_ids, return_inverse=True)
    C = np.zeros((len(gu), len(pu)))
    np.add.at(C, (gi, pi), 1)
    r, c = linear_sum_assignment(-C)
    idtp = C[r, c].sum()
    n = len(gt_ids)
    return {"idf1": idtp / n, "idp": idtp / n, "idr": idtp / n, "n_boxes": n}


def fast_gtbox_metrics(trs, gid, cache):
    """Exact MTMC metrics in the GT-box setting (each GT box carries exactly one prediction):
    IDF1/IDP/IDR via the ID bipartite matching; FN = FP = 0; IDSW counted as in motmetrics
    (per GT identity, change of matched prediction id along camera-then-frame order); MOTA = 1 - IDSW/GT.
    Verified against motmetrics in tests/smoke_test.py."""
    m = fast_idf1(trs, gid, cache)
    seq = []
    for t, g in zip(trs, gid):
        d = cache[t["cam"]]
        c = cam_int(t["cam"])
        for r in t["rows"]:
            seq.append((int(d["gid"][r]), c, int(d["frame"][r]), int(g)))
    seq.sort()
    idsw = sum(1 for a, b in zip(seq, seq[1:]) if a[0] == b[0] and a[3] != b[3])
    n = m["n_boxes"]
    return {**m, "num_switches": idsw, "num_misses": 0, "num_false_positives": 0, "num_objects": n,
            "num_predictions": n, "mota": 1 - idsw / max(n, 1)}


def gt_box_count(root, scene):
    return int(sum(len(read_gt(m["gt"])) for m in scene_meta(root, scene).values()))


def fast_idf1_det(trs, gid, cache, n_gt):
    """IDF1 for detection-based predictions, for tuning only (final numbers come from motmetrics).
    Uses the precomputed detection->GT identity (IoU >= 0.5) of every predicted box."""
    g_list, p_list, n_pred = [], [], 0
    for t, g in zip(trs, gid):
        if g < 0:
            continue
        gg = cache[t["cam"]]["gid"][t["rows"]]
        n_pred += len(gg)
        g_list.append(gg[gg >= 0])
        p_list.append(np.full(int((gg >= 0).sum()), g))
    if n_pred == 0:
        return 0.0
    gt_ids, pr_ids = np.concatenate(g_list), np.concatenate(p_list)
    if len(gt_ids) == 0:
        return 0.0
    gu, gi = np.unique(gt_ids, return_inverse=True)
    pu, pi = np.unique(pr_ids, return_inverse=True)
    C = np.zeros((len(gu), len(pu)))
    np.add.at(C, (gi, pi), 1)
    r, c = linear_sum_assignment(-C)
    return 2 * C[r, c].sum() / (n_gt + n_pred)


def filter_multicam(trs, gid):
    """AI City MTMC ground truth contains only vehicles seen by >= 2 cameras, so identities observed in a
    single camera are removed from the output (standard practice). Returns gid with removed ids set to -1."""
    cams = defaultdict(set)
    for t, g in zip(trs, gid):
        cams[g].add(t["cam"])
    return np.array([g if len(cams[g]) >= 2 else -1 for g in gid], dtype=np.int64)


def _acc_update(acc, gt_rows, pr_rows, frameid, iou_thr=0.5):
    gt_ids = gt_rows[:, 1].astype(int).tolist()
    pr_ids = pr_rows[:, 0].astype(int).tolist()
    if len(gt_rows) and len(pr_rows):
        iou = iou_xywh(gt_rows[:, 2:6].astype(np.float64), pr_rows[:, 1:5].astype(np.float64))
        dist = np.where(iou >= iou_thr, 1.0 - iou, np.nan)
    else:
        dist = np.empty((len(gt_rows), len(pr_rows)))
    acc.update(gt_ids, pr_ids, dist, frameid=frameid)


def mtmc_motmetrics(root, scenes, preds_by_scene, iou_thr=0.5):
    """preds_by_scene: {scene: DataFrame(cam, frame, id, x, y, w, h)}. Returns DataFrame incl. OVERALL."""
    accs, names = [], []
    for scene in scenes:
        acc = mm.MOTAccumulator(auto_id=False)
        pdf = preds_by_scene[scene]
        for cam, m in scene_meta(root, scene).items():
            c = cam_int(cam)
            gt = read_gt(m["gt"])
            pr = pdf[pdf.cam == c][["id", "x", "y", "w", "h", "frame"]].to_numpy()
            gt_by = defaultdict(list)
            pr_by = defaultdict(list)
            for k, f in enumerate(gt[:, 0].astype(int)):
                gt_by[f].append(k)
            for k, f in enumerate(pr[:, 5].astype(int)):
                pr_by[f].append(k)
            for f in sorted(set(gt_by) | set(pr_by)):
                _acc_update(acc, gt[gt_by.get(f, [])].reshape(-1, 6), pr[pr_by.get(f, [])].reshape(-1, 6),
                            c * 100000 + f, iou_thr)
        accs.append(acc)
        names.append(scene)
    mh = mm.metrics.create()
    return mh.compute_many(accs, names=names, metrics=MM_METRICS, generate_overall=True)


def per_camera_motmetrics(root, scene, pred_df, iou_thr=0.5):
    """OLD protocol: each camera evaluated alone. Cross-camera identity errors are invisible here."""
    accs, names = [], []
    for cam, m in scene_meta(root, scene).items():
        acc = mm.MOTAccumulator(auto_id=False)
        c = cam_int(cam)
        gt = read_gt(m["gt"])
        pr = pred_df[pred_df.cam == c][["id", "x", "y", "w", "h", "frame"]].to_numpy()
        frames = sorted(set(gt[:, 0].astype(int)) | set(pr[:, 5].astype(int)))
        for f in frames:
            _acc_update(acc, gt[gt[:, 0] == f], pr[pr[:, 5] == f], f, iou_thr)
        accs.append(acc)
        names.append(f"{scene}_{cam}")
    return mm.metrics.create().compute_many(accs, names=names, metrics=["idf1", "idp", "idr", "mota", "num_switches"])


def tracklet_level(trs, g, gid):
    """FR  = Nf / Nt : Nt = GT identities whose tracklets appear in >=2 cameras;
                       Nf = those of them whose tracklets are split over >1 predicted identity.
       MER = Nm / Na : Na = cross-camera tracklet pairs that the prediction puts in one identity;
                       Nm = those pairs whose GT identities differ.
       Pairwise precision/recall over cross-camera tracklet pairs (precision = 1 - MER)."""
    by_gt = defaultdict(list)
    for k, t in enumerate(trs):
        by_gt[t["gt"]].append(k)
    Nt = Nf = 0
    for g, ks in by_gt.items():
        if g >= 0 and len({trs[k]["cam"] for k in ks}) >= 2:
            Nt += 1
            Nf += int(len({gid[k] if gid[k] >= 0 else ("removed", k) for k in ks}) > 1)
    by_pr = defaultdict(list)
    for k, p in enumerate(gid):
        if p >= 0:
            by_pr[p].append(k)
    Na = Nm = 0
    for ks in by_pr.values():
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                if trs[ks[a]]["cam"] != trs[ks[b]]["cam"]:
                    Na += 1
                    Nm += int(trs[ks[a]]["gt"] != trs[ks[b]]["gt"])
    true_pairs = sum(1 for g, ks in by_gt.items() if g >= 0 for a in range(len(ks)) for b in range(a + 1, len(ks))
                     if trs[ks[a]]["cam"] != trs[ks[b]]["cam"])
    tp = Na - Nm
    return {"FR": Nf / max(Nt, 1), "Nf": Nf, "Nt": Nt, "MER": Nm / max(Na, 1), "Nm": Nm, "Na": Na,
            "pair_precision": tp / max(Na, 1), "pair_recall": tp / max(true_pairs, 1), "true_pairs": true_pairs}


def graph_stats(trs, g, tau):
    n = len(trs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(g["I"], g["J"]):
        parent[find(int(i))] = find(int(j))
    sizes = np.array(list(Counter(find(i) for i in range(n)).values()))
    return {"tau": tau, "n_boxes": int(sum(t["n"] for t in trs)), "n_tracklets": n,
            "mean_tracklet_len": float(np.mean([t["n"] for t in trs])),
            "all_pairs": g["n_candidates_all"], "feasible_pairs": g["n_feasible"],
            "edges_kept": int(len(g["I"])), "pos_edges_kept": int(g["y"].sum()),
            "components": int(len(sizes)), "mean_comp_size": float(sizes.mean()),
            "max_comp_size": int(sizes.max())}


# ----------------------------------------------------------------------------- diagnosis of the old 0.074
def diagnose_old_protocol(root, scenes, max_frames=16, min_len=4):
    """Re-creates the V15 prediction coverage WITHOUT any model: V15 wrote predictions only for the
    <=16 frames sampled per tracklet and dropped tracklets shorter than 4 frames."""
    rows = []
    for scene in scenes:
        n_gt = n_pred = 0
        for cam, m in scene_meta(root, scene).items():
            gt = read_gt(m["gt"])
            n_gt += len(gt)
            for oid in np.unique(gt[:, 1]):
                L = int((gt[:, 1] == oid).sum())
                if L >= min_len:
                    n_pred += min(L, max_frames)
        rows.append({"scene": scene, "gt_boxes": n_gt, "boxes_with_prediction": n_pred,
                     "coverage": n_pred / max(n_gt, 1),
                     "MOTA_upper_bound_if_all_predictions_correct": n_pred / max(n_gt, 1)})
    df = pd.DataFrame(rows)
    tot = df[["gt_boxes", "boxes_with_prediction"]].sum()
    df.loc[len(df)] = {"scene": "S02+S05", "gt_boxes": tot.gt_boxes,
                       "boxes_with_prediction": tot.boxes_with_prediction,
                       "coverage": tot.boxes_with_prediction / tot.gt_boxes,
                       "MOTA_upper_bound_if_all_predictions_correct": tot.boxes_with_prediction / tot.gt_boxes}
    return df

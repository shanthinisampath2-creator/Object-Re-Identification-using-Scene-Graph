"""Intra-camera tracklets.

Setting B (GT boxes)
  mode="gt"        : oracle intra-camera tracklets (GT ids within a camera) -> isolates CROSS-camera association.
  mode="hungarian" : our tracker on GT boxes, ignoring GT ids (cost = l1*(1-IoU) + l2*(1-cos)).
Setting C (detections)
  build_det_tracklets : Kalman filter (constant velocity, DeepSORT-style state) + Hungarian matching with
                        cost = alpha*(1-IoU(predicted box, detection)) + beta*(1-cos(track memory, detection)).
Every box assigned to a tracklet receives a prediction later (fixes the old MOTA = 0.074 coverage bug).
"""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import frame_to_sec


def iou_xywh(a, b):
    """a: (N,4), b: (M,4) xywh -> (N,M) IoU"""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def _group_by_frame(frame):
    order = np.argsort(frame, kind="stable")
    uniq, starts = np.unique(frame[order], return_index=True)
    return uniq, np.split(order, starts[1:])


# ----------------------------------------------------------------------------- Setting B tracker
def hungarian_tracks(frame, xywh, emb, l_iou=0.5, l_cos=0.5, max_cost=0.7, min_iou=0.05, max_age=5):
    tid = -np.ones(len(frame), dtype=np.int64)
    active, nxt = [], 0
    for f, idx in zip(*_group_by_frame(frame)):
        active = [t for t in active if f - t["last"] <= max_age]
        used = set()
        if active and len(idx):
            tb = np.stack([t["box"] for t in active])
            te = np.stack([t["emb"] for t in active])
            iou = iou_xywh(tb, xywh[idx])
            cost = l_iou * (1 - iou) + l_cos * (1 - te @ emb[idx].T)
            cost[iou < min_iou] = 1e3
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] <= max_cost:
                    k = idx[j]
                    tid[k] = active[i]["id"]
                    active[i]["box"] = xywh[k]
                    e = 0.9 * active[i]["emb"] + 0.1 * emb[k]
                    active[i]["emb"] = e / (np.linalg.norm(e) + 1e-9)
                    active[i]["last"] = f
                    used.add(j)
        for j in range(len(idx)):
            if j not in used:
                k = idx[j]
                tid[k] = nxt
                active.append({"id": nxt, "box": xywh[k], "emb": emb[k].astype(np.float32), "last": f})
                nxt += 1
    return tid


# ----------------------------------------------------------------------------- Kalman filter (Setting C)
class KalmanXYAH:
    """Constant-velocity Kalman filter on (cx, cy, aspect, h) as in SORT/DeepSORT."""
    _std_pos, _std_vel = 1.0 / 20, 1.0 / 160

    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
               10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov, steps=1):
        for _ in range(max(1, int(steps))):
            h = mean[3]
            q = np.diag(np.square([self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
                                   self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h]))
            mean = self.F @ mean
            cov = self.F @ cov @ self.F.T + q
        return mean, cov

    def update(self, mean, cov, z):
        h = mean[3]
        r = np.diag(np.square([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]))
        S = self.H @ cov @ self.H.T + r
        K = cov @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.H @ mean)
        cov = (np.eye(8) - K @ self.H) @ cov
        return mean, cov


def xywh_to_xyah(b):
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2, b[2] / max(b[3], 1e-6), b[3]])


def xyah_to_xywh(m):
    h = max(m[3], 1.0)
    w = max(m[2], 0.05) * h
    return np.array([m[0] - w / 2, m[1] - h / 2, w, h])


def kalman_hungarian_tracks(frame, xywh, emb, conf, alpha=0.5, beta=0.5, max_cost=0.65, min_iou=0.1,
                            max_age=30, high_conf=0.5, ema=0.9):
    """Returns a local track id per detection (-1 = never confirmed into a track)."""
    kf = KalmanXYAH()
    tid = -np.ones(len(frame), dtype=np.int64)
    tracks, nxt, prev_f = [], 0, None
    for f, idx in zip(*_group_by_frame(frame)):
        steps = 1 if prev_f is None else f - prev_f
        prev_f = f
        for t in tracks:
            t["mean"], t["cov"] = kf.predict(t["mean"], t["cov"], steps)
            t["age"] += steps
        tracks = [t for t in tracks if t["age"] <= max_age]
        used = set()
        if tracks and len(idx):
            pb = np.stack([xyah_to_xywh(t["mean"][:4]) for t in tracks])
            te = np.stack([t["emb"] for t in tracks])
            iou = iou_xywh(pb, xywh[idx])
            cost = alpha * (1 - iou) + beta * (1 - te @ emb[idx].T)
            cost[iou < min_iou] = 1e3
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] <= max_cost:
                    k = idx[j]
                    t = tracks[i]
                    t["mean"], t["cov"] = kf.update(t["mean"], t["cov"], xywh_to_xyah(xywh[k]))
                    e = ema * t["emb"] + (1 - ema) * emb[k]
                    t["emb"] = e / (np.linalg.norm(e) + 1e-9)
                    t["age"] = 0
                    tid[k] = t["id"]
                    used.add(j)
        for j in range(len(idx)):
            k = idx[j]
            if j in used or conf[k] < high_conf:
                continue
            mean, cov = kf.initiate(xywh_to_xyah(xywh[k]))
            tracks.append({"id": nxt, "mean": mean, "cov": cov, "emb": emb[k].astype(np.float32), "age": 0})
            tid[k] = nxt
            nxt += 1
    return tid


# ----------------------------------------------------------------------------- tracklet records
def _make_tracklet(scene, cam, lid, rows, d, emb, seq_len, neg_counter):
    frame, gid = d["frame"], d["gid"]
    off, fps = float(d["offset"]), float(d["fps"])
    rows = rows[np.argsort(frame[rows], kind="stable")]
    fr = frame[rows]
    sel = rows if len(rows) <= seq_len else rows[np.linspace(0, len(rows) - 1, seq_len).astype(int)]
    t_rows = frame_to_sec(frame[sel], off, fps)
    mean = emb[rows].mean(0)
    mean /= (np.linalg.norm(mean) + 1e-9)
    c = Counter(gid[rows].tolist()).most_common(1)[0]
    g = int(c[0])
    if g < 0:                        # false-positive tracklet: unique negative id so it never counts as a match
        neg_counter[0] -= 1
        g = neg_counter[0]
    return {"scene": scene, "cam": cam, "local_id": int(lid), "rows": rows, "frames": fr, "n": int(len(rows)),
            "t0": float(frame_to_sec(fr[0], off, fps)), "t1": float(frame_to_sec(fr[-1], off, fps)),
            "seq": emb[sel], "seq_t": (t_rows - t_rows[0]).astype(np.float32), "mean": mean.astype(np.float32),
            "gt": g, "purity": c[1] / len(rows)}


def build_tracklets(scene, cache, mode="gt", seq_len=16, **hkw):
    """Setting B. cache: {cam: npz-dict}. One tracklet per local track; every GT box is covered."""
    out, neg = [], [-1000000]
    for cam, d in cache.items():
        emb = d["emb"].astype(np.float32)
        if mode == "gt":
            local = d["gid"].astype(np.int64)
        elif mode == "hungarian":
            local = hungarian_tracks(d["frame"], d["xywh"], emb, **hkw)
        else:
            raise ValueError(mode)
        for lid in np.unique(local):
            out.append(_make_tracklet(scene, cam, lid, np.where(local == lid)[0], d, emb, seq_len, neg))
    return out


def build_det_tracklets(scene, det_cache, seq_len=16, min_len=5, remove_static=True, static_ratio=0.5, **kw):
    """Setting C: Kalman + Hungarian tracklets on detections. Detections that never join a confirmed
    tracklet of >= min_len boxes produce no output (standard single-camera tracking practice)."""
    out, neg = [], [-1000000]
    for cam, d in det_cache.items():
        if len(d["frame"]) == 0:
            continue
        emb = d["emb"].astype(np.float32)
        local = kalman_hungarian_tracks(d["frame"], d["xywh"].astype(np.float64), emb, d["conf"], **kw)
        for lid in np.unique(local[local >= 0]):
            rows = np.where(local == lid)[0]
            if len(rows) < min_len:
                continue
            if remove_static:          # parked vehicles never cross cameras; they only create false merges
                b = d["xywh"][rows]
                c = b[:, :2] + b[:, 2:] / 2
                if np.linalg.norm(c[-1] - c[0]) < static_ratio * b[:, 3].mean():
                    continue
            out.append(_make_tracklet(scene, cam, lid, rows, d, emb, seq_len, neg))
    return out

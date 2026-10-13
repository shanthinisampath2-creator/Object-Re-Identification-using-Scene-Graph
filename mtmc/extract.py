"""One sequential pass over each CityFlow video (no per-frame seeking; decoding runs in a background thread).

dump_crops(...)   -> ReID training images from TRAIN scenes only (pid = scene*10000 + id). CPU only.
extract_scene(...) -> per-box 512-D embeddings for every GT box (and, optionally, every detection box)
                      cached as <cache>/gt/<scene>/<cam>.npz and <cache>/det/<scene>/<cam>.npz
"""
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import cam_int, read_gt, read_mot, load_roi, scene_meta

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _frames(video, last_frame, wanted):
    """Yields (frame_idx, bgr) for wanted frames; decoding happens in a thread (cv2 releases the GIL)."""
    q = queue.Queue(maxsize=64)

    def reader():
        cap = cv2.VideoCapture(str(video))
        f = 0
        while f < last_frame:
            ok, img = cap.read()
            if not ok:
                break
            f += 1
            if f in wanted:
                q.put((f, img))
        cap.release()
        q.put(None)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
    t.join()


def _crop_resized(bgr, x, y, w, h, out_hw):
    H, W = bgr.shape[:2]
    x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
    x2, y2 = min(W, int(round(x + w))), min(H, int(round(y + h)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.full((out_hw[0], out_hw[1], 3), 127, np.uint8)
    c = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return cv2.resize(c, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)


def dump_crops(root, scenes, out_dir, stride=5, min_side=20, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for scene in scenes:
        snum = int(scene[1:])
        for cam, m in scene_meta(root, scene).items():
            gt = read_gt(m["gt"])
            by = {}
            for k, r in enumerate(gt):
                by.setdefault(int(r[0]), []).append(k)
            wanted = {f for f in by if f % stride == 0}
            last = max(wanted) if wanted else 0
            for f, img in _frames(m["video"], last, wanted):
                H, W = img.shape[:2]
                for k in by[f]:
                    _, oid, x, y, w, h = gt[k]
                    if min(w, h) < min_side:
                        continue
                    x1, y1 = max(0, int(x)), max(0, int(y))
                    x2, y2 = min(W, int(x + w)), min(H, int(y + h))
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    pid = snum * 10000 + int(oid)
                    cv2.imwrite(str(out / f"{pid:06d}_c{cam_int(cam):03d}_{scene}_f{f:06d}.jpg"),
                                img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 92])
                    n += 1
            log(f"[crops] {scene}/{cam}: total so far {n}")
    return n


def load_detections(m, conf_thr=0.3, min_side=15, use_roi=True):
    """Detections of one camera after confidence / size / ROI filtering. Returns array frame,-1,x,y,w,h,conf."""
    if m.get("det") is None:
        return np.zeros((0, 7))
    d = read_mot(m["det"])
    keep = (d[:, 6] >= conf_thr) & (d[:, 4] >= min_side) & (d[:, 5] >= min_side)
    d = d[keep]
    roi = load_roi(m["dir"]) if use_roi else None
    if roi is not None and len(d):
        H, W = roi.shape
        bx = np.clip((d[:, 2] + d[:, 4] / 2).astype(int), 0, W - 1)
        by = np.clip((d[:, 3] + d[:, 5]).astype(int) - 1, 0, H - 1)
        d = d[roi[by, bx]]
    return d


def match_det_to_gt(det, gt, thr=0.5):
    """GT identity of each detection (IoU >= thr with the best GT box in the same frame), else -1."""
    from .tracklets import iou_xywh
    gid = -np.ones(len(det), dtype=np.int64)
    gby = {}
    for k, f in enumerate(gt[:, 0].astype(int)):
        gby.setdefault(f, []).append(k)
    for f in np.unique(det[:, 0].astype(int)):
        di = np.where(det[:, 0].astype(int) == f)[0]
        gi = gby.get(f, [])
        if not gi:
            continue
        iou = iou_xywh(det[di, 2:6], gt[gi, 2:6])
        best = iou.argmax(1)
        ok = iou[np.arange(len(di)), best] >= thr
        gid[di[ok]] = gt[np.array(gi)[best[ok]], 1].astype(np.int64)
    return gid


@torch.no_grad()
def _embed(model, crops, device):
    x = torch.from_numpy(np.stack(crops)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - MEAN.to(device)) / STD.to(device)
    with torch.autocast("cuda", enabled=str(device).startswith("cuda")):
        f = model(x) + model(torch.flip(x, dims=[3]))
    return F.normalize(f.float(), dim=1).cpu().numpy().astype(np.float16)


@torch.no_grad()
def extract_scene(root, scene, model, img_hw, cache_dir, device="cuda", bs=256, with_det=False,
                  det_conf=0.3, log=print):
    """Embeds every GT box (and every filtered detection if with_det) of a scene in one pass per video."""
    model.eval()
    out_gt = Path(cache_dir) / "gt" / scene
    out_det = Path(cache_dir) / "det" / scene
    out_gt.mkdir(parents=True, exist_ok=True)
    out_det.mkdir(parents=True, exist_ok=True)
    for cam, m in scene_meta(root, scene).items():
        dst_gt, dst_det = out_gt / f"{cam}.npz", out_det / f"{cam}.npz"
        need_gt = not dst_gt.exists()
        need_det = with_det and not dst_det.exists()
        if not (need_gt or need_det):
            log(f"[emb] cached {scene}/{cam}")
            continue
        gt = read_gt(m["gt"])
        det = load_detections(m, conf_thr=det_conf) if need_det else np.zeros((0, 7))
        if need_det and m["det"] is None:
            log(f"[emb] WARNING {scene}/{cam}: no det/ file found - detector setting skipped for this camera")
        boxes = []                                # (kind, row)
        if need_gt:
            boxes += [(0, k) for k in range(len(gt))]
        if need_det:
            boxes += [(1, k) for k in range(len(det))]
        by = {}
        for kind, k in boxes:
            f = int((gt if kind == 0 else det)[k, 0])
            by.setdefault(f, []).append((kind, k))
        emb_gt = np.zeros((len(gt), 512), np.float16)
        emb_det = np.zeros((len(det), 512), np.float16)
        buf, keys = [], []

        def flush():
            if not buf:
                return
            e = _embed(model, buf, device)
            for (kind, k), v in zip(keys, e):
                (emb_gt if kind == 0 else emb_det)[k] = v
            buf.clear()
            keys.clear()

        last = max(by) if by else 0
        n_frames = 0
        for f, img in _frames(m["video"], last, set(by)):
            n_frames += 1
            for kind, k in by[f]:
                r = (gt if kind == 0 else det)[k]
                buf.append(_crop_resized(img, r[2], r[3], r[4], r[5], img_hw))
                keys.append((kind, k))
                if len(buf) >= bs:
                    flush()
        flush()
        if need_gt:
            np.savez_compressed(dst_gt, frame=gt[:, 0].astype(np.int32), gid=gt[:, 1].astype(np.int32),
                                xywh=gt[:, 2:6].astype(np.float32), emb=emb_gt, offset=m["offset"], fps=m["fps"])
        if need_det:
            np.savez_compressed(dst_det, frame=det[:, 0].astype(np.int32), gid=match_det_to_gt(det, gt).astype(np.int32),
                                xywh=det[:, 2:6].astype(np.float32), conf=det[:, 6].astype(np.float32), emb=emb_det,
                                offset=m["offset"], fps=m["fps"])
        log(f"[emb] {scene}/{cam}: gt {len(gt) if need_gt else '-'} det {len(det) if need_det else '-'} "
            f"boxes, {n_frames} frames decoded")


def load_scene_cache(cache_dir, scene, kind="gt"):
    d = {}
    for p in sorted((Path(cache_dir) / kind / scene).glob("c*.npz")):
        z = np.load(p)
        d[p.stem] = {k: z[k] for k in z.files}
    return d

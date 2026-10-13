"""Shared helpers: CityFlow (AI City Track 1/3 MTMC) layout, GT/detection reading, timing, seeds."""
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

# AI City MTMC layout (as in the Kaggle copy of CityFlow)
SCENE_SPLIT = {"S01": "train", "S03": "train", "S04": "train",
               "S02": "validation", "S05": "validation"}
DET_PREFERENCE = ["det_mask_rcnn.txt", "det_yolo3.txt", "det_ssd512.txt", "det.txt"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id, base_seed=0):
    """Module-level (picklable) DataLoader worker init - required on Windows."""
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def find_cityflow_root(base):
    """Return the folder that contains train/ and validation/ (searches below `base`)."""
    base = Path(base)
    if (base / "train").is_dir() and (base / "validation").is_dir():
        return base
    for p in sorted(base.rglob("validation")):
        if p.is_dir() and (p.parent / "train").is_dir():
            return p.parent
    raise FileNotFoundError(f"No CityFlow root (train/ + validation/) found under {base}")


def find_duke_root(base):
    base = Path(base)
    for p in [base] + sorted(base.rglob("bounding_box_train")):
        r = p if (p / "bounding_box_train").is_dir() else p.parent
        if (r / "bounding_box_train").is_dir() and (r / "query").is_dir():
            return r
    raise FileNotFoundError(f"No DukeMTMC-reID root found under {base}")


def cam_int(cam_name: str) -> int:
    """'c006' -> 6"""
    return int(cam_name.lstrip("c"))


def scene_dir(root, scene):
    return Path(root) / SCENE_SPLIT[scene] / scene


def list_cameras(root, scene):
    """Cameras of a scene that have both a video and a gt.txt."""
    out = []
    for cdir in sorted(scene_dir(root, scene).iterdir()):
        if (cdir / "vdo.avi").exists() and (cdir / "gt" / "gt.txt").exists():
            out.append(cdir.name)
    return out


def read_mot(path):
    """MOT-format txt -> float array: frame,id,x,y,w,h,conf (frames 1-based; conf=1 if absent)."""
    rows = []
    with open(path) as f:
        for line in f:
            v = line.replace(" ", "").strip().split(",")
            if len(v) < 6:
                continue
            conf = float(v[6]) if len(v) > 6 else 1.0
            rows.append([float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5]), conf])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7)


def read_gt(path):
    """gt.txt -> array frame,id,x,y,w,h"""
    return read_mot(path)[:, :6]


def find_det_file(cam_dir, preference=None):
    d = Path(cam_dir) / "det"
    if not d.is_dir():
        return None
    for name in (preference or DET_PREFERENCE):
        if (d / name).exists():
            return d / name
    txt = sorted(d.glob("*.txt"))
    return txt[0] if txt else None


def load_roi(cam_dir):
    p = Path(cam_dir) / "roi.jpg"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m > 127)


def load_offsets(root, scene):
    """cam_timestamp/Sxx.txt -> {cam: offset_seconds}. Missing file -> zeros."""
    p = Path(root) / "cam_timestamp" / f"{scene}.txt"
    off = {}
    if p.exists():
        for line in open(p):
            v = line.split()
            if len(v) >= 2:
                off[v[0]] = float(v[1])
    return off


def video_fps(path, default=10.0):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0 or fps > 120:
        return default
    return float(fps)


def scene_meta(root, scene):
    """{cam: {'dir','video','gt','det','offset','fps'}} for one scene."""
    offs = load_offsets(root, scene)
    meta = {}
    for cam in list_cameras(root, scene):
        cdir = scene_dir(root, scene) / cam
        meta[cam] = {
            "dir": str(cdir),
            "video": str(cdir / "vdo.avi"),
            "gt": str(cdir / "gt" / "gt.txt"),
            "det": find_det_file(cdir),
            "offset": offs.get(cam, 0.0),
            "fps": video_fps(cdir / "vdo.avi"),
        }
    return meta


def frame_to_sec(frame, offset, fps):
    """Global scene time (s) of a 1-based frame index."""
    return offset + (np.asarray(frame, dtype=np.float64) - 1.0) / fps

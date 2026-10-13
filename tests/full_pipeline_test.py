"""Runs the REAL scripts (02 -> 03 all stages -> 04 -> 05) on a tiny synthetic CityFlow + Duke dataset.
Use: python tests/full_pipeline_test.py   (about 10-20 min on CPU). Nothing here touches your real data."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SCENES = {"S01": ("train", ["c001", "c002", "c003"]), "S03": ("train", ["c010", "c011", "c012"]),
          "S04": ("train", ["c016", "c017", "c018"]), "S02": ("validation", ["c006", "c007", "c008"]),
          "S05": ("validation", ["c010", "c011", "c012"])}
W, H, FPS, NF = 320, 240, 10, 160


def make_cityflow(root, n_veh=14, seed=0):
    rng = np.random.default_rng(seed)
    (root / "cam_timestamp").mkdir(parents=True, exist_ok=True)
    for scene, (split, cams) in SCENES.items():
        offs = {c: float(k) * 1.5 for k, c in enumerate(cams)}
        (root / "cam_timestamp" / f"{scene}.txt").write_text("".join(f"{c} {offs[c]}\n" for c in cams))
        tracks = {c: [] for c in cams}
        for vid in range(1, n_veh + 1):
            c1, c2 = rng.integers(30, 255, 3), rng.integers(30, 255, 3)
            w, h = int(rng.integers(40, 70)), int(rng.integers(28, 45))
            path = rng.permutation(cams)[: int(rng.integers(2, len(cams) + 1))]
            t = float(rng.uniform(0, 4))
            for c in path:
                dur = float(rng.uniform(3, 6))
                f0 = int(round((t - offs[c]) * FPS)) + 1
                f1 = f0 + int(dur * FPS)
                y = int(rng.integers(20, H - h - 20))
                for f in range(max(1, f0), min(NF, f1) + 1):
                    x = int(10 + (W - w - 20) * (f - f0) / max(1, f1 - f0))
                    tracks[c].append((f, vid, x, y, w, h, c1, c2))
                t += dur + float(rng.uniform(1, 4))
        for c in cams:
            d = root / split / scene / c
            (d / "gt").mkdir(parents=True, exist_ok=True)
            (d / "det").mkdir(parents=True, exist_ok=True)
            roi = np.full((H, W), 255, np.uint8)
            roi[:, :8] = 0
            cv2.imwrite(str(d / "roi.jpg"), roi)
            vw = cv2.VideoWriter(str(d / "vdo.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
            by = {}
            for r in tracks[c]:
                by.setdefault(r[0], []).append(r)
            pc = rng.integers(30, 255, 3)                       # a parked car (not in GT, like real AI City GT)
            gt_lines, det_lines = [], []
            for f in range(1, NF + 1):
                img = np.full((H, W, 3), 90, np.uint8)
                img[200:230, 250:300] = pc
                det_lines.append(f"{f},-1,250,200,50,30,0.9,-1,-1,-1\n")
                for (_, vid, x, y, w, h, a, b) in by.get(f, []):
                    img[y:y + h, x:x + w] = a
                    img[y + h // 2:y + h, x:x + w // 2] = b
                    gt_lines.append(f"{f},{vid},{x},{y},{w},{h},1,-1,-1,-1\n")
                    if rng.random() > 0.08:                     # 8 % missed detections, jittered boxes
                        j = rng.normal(0, 1.5, 4)
                        det_lines.append(f"{f},-1,{x + j[0]:.1f},{y + j[1]:.1f},{w + j[2]:.1f},{h + j[3]:.1f},"
                                         f"{rng.uniform(0.55, 0.99):.3f},-1,-1,-1\n")
                if rng.random() < 0.05:                         # random false positive
                    det_lines.append(f"{f},-1,{rng.integers(20, 200)},{rng.integers(20, 150)},40,30,0.4,-1,-1,-1\n")
                vw.write(img)
            vw.release()
            (d / "gt" / "gt.txt").write_text("".join(gt_lines))
            (d / "det" / "det_yolo3.txt").write_text("".join(det_lines))


def make_duke(src_crops, dst):
    for s in ["bounding_box_train", "query", "bounding_box_test"]:
        (dst / s).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    files = sorted(src_crops.glob("*.jpg"))
    pids = sorted({f.name.split("_")[0] for f in files})
    test = set(pids[::2])
    for f in files:
        pid = f.name.split("_")[0]
        sub = "bounding_box_train" if pid not in test else ("query" if rng.random() < 0.15 else "bounding_box_test")
        shutil.copy(f, dst / sub / f.name)


def run(cmd, env):
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE, env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd)}")


def main():
    base = Path(tempfile.mkdtemp(prefix="reid_test_"))
    env = {**os.environ, "REID_BASE": str(base), "REID_TEST": "1"}
    (base / "data").mkdir(parents=True)
    make_cityflow(base / "data" / "cityflow")
    py = sys.executable
    (base / "data" / "data_paths.json").write_text(json.dumps({"cityflow": str(base / "data" / "cityflow"),
                                                               "duke": str(base / "data" / "duke")}))
    run([py, "02_prepare_tonight.py"], env)
    make_duke(next(p for p in (base / "work").glob("crops_bb*") if p.is_dir()), base / "data" / "duke")
    run([py, "03_run_cityflow.py", "--seeds", "0", "1", "--backbone-epochs", "2", "--assoc-epochs", "10",
         "--batch", "64", "--workers", "0"], env)
    run([py, "03_run_cityflow.py", "--stage", "hung", "--seeds", "0", "--assoc-epochs", "5", "--workers", "0"], env)
    run([py, "04_run_duke.py", "--epochs", "2", "--workers", "0"], env)
    run([py, "05_make_report.py"], env)
    rep = (base / "results" / "REPORT.md").read_text(encoding="utf-8")
    for must in ["## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8."]:
        assert must in rep, f"report section {must} missing"
    print(f"\nPIPELINE TEST PASSED. Synthetic output (numbers are meaningless): {base / 'results'}")


if __name__ == "__main__":
    main()

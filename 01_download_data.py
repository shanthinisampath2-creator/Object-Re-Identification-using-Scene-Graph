"""TONIGHT (no GPU needed). Downloads CityFlow and DukeMTMC-reID from Kaggle INTO this folder (data/),
then checks the CityFlow layout (scenes, cameras, detection files, ROI masks).

Needs Kaggle credentials: kaggle.com -> Settings -> API -> Create New Token, and put kaggle.json in
C:\\Users\\<you>\\.kaggle\\kaggle.json
"""
import json
import os
import time

import settings as S

S.utf8_console()
os.environ["KAGGLEHUB_CACHE"] = str(S.DATA / "kagglehub")     # keep the download under <repo>/data

from mtmc.common import SCENE_SPLIT, find_cityflow_root, find_duke_root, list_cameras, scene_meta  # noqa: E402

log = S.Tee("01_download.log")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cityflow", help="folder of a manually downloaded + extracted CityFlow / AIC22 Track 1 copy")
    ap.add_argument("--duke", help="folder of a manually downloaded + extracted DukeMTMC-reID copy")
    a = ap.parse_args()
    paths = json.loads(S.PATHS_FILE.read_text()) if S.PATHS_FILE.exists() else {}
    for key, handle, finder, manual in [("cityflow", S.CITYFLOW_KAGGLE, find_cityflow_root, a.cityflow),
                                        ("duke", S.DUKE_KAGGLE, find_duke_root, a.duke)]:
        try:
            if manual:
                paths[key] = str(finder(manual))
            elif key in paths and os.path.isdir(paths[key]):
                log(f"{key}: already at {paths[key]}")
                continue
            else:
                t = time.time()
                log(f"Downloading {handle} from Kaggle ...")
                import kagglehub
                paths[key] = str(finder(kagglehub.dataset_download(handle)))
                log(f"  took {(time.time() - t) / 60:.1f} min")
            S.PATHS_FILE.write_text(json.dumps(paths, indent=2))
            log(f"{key}: {paths[key]}")
        except Exception as e:          # one dataset failing must not block the other
            log(f"{key}: NOT AVAILABLE ({type(e).__name__}: {e})")

    if "cityflow" in paths:
        root = paths["cityflow"]
        log("\nCityFlow layout check")
        for s in SCENE_SPLIT:
            meta = scene_meta(root, s)
            n_det = sum(1 for m in meta.values() if m["det"])
            n_roi = sum(1 for m in meta.values() if os.path.exists(os.path.join(m["dir"], "roi.jpg")))
            dets = sorted({os.path.basename(str(m["det"])) for m in meta.values() if m["det"]})
            log(f"  {s} ({SCENE_SPLIT[s]}): {len(list_cameras(root, s))} cameras | det files {n_det} {dets} | roi.jpg {n_roi}")
        for s in ("S01", "S02", "S05"):
            if not all(m["det"] for m in scene_meta(root, s).values()):
                log(f"  WARNING: some cameras of {s} have no det/ file -> Setting C will skip them")
    if "duke" in paths:
        for sub in ("bounding_box_train", "query", "bounding_box_test"):
            n = len([f for f in os.listdir(os.path.join(paths["duke"], sub)) if f.endswith(".jpg")])
            log(f"  Duke {sub}: {n} images (expected 16522 / 2228 / 17661)")
    log("\nAVAILABLE: " + ", ".join(k for k in ("cityflow", "duke") if k in paths))


if __name__ == "__main__":
    main()

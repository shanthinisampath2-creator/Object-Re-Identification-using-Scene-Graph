"""CPU ONLY (no GPU needed, ~20-40 min). Uses the embeddings already computed on 7 October (work\\emb_cache) to
compare single-linkage vs average-linkage identity clustering for the appearance-only (cosine) baseline, in
Setting B (GT boxes) and Setting C (detections), with the same 2-fold cross-validation (S02 <-> S05).
No model is trained. Results: results\\quick_linkage_check.csv and printed tables.

These embeddings come from the 7-Oct backbone (trained on S01+S03+S04), which never saw S02/S05, so the cosine
baseline numbers on S02/S05 are valid; only the learned variants need the leak-free re-run.
"""
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # force CPU
import pandas as pd  # noqa: E402

import settings as S  # noqa: E402

S.utf8_console()
from mtmc.experiments import CV_FOLDS, Data, run_setting_c, run_variant, summarize  # noqa: E402

log = S.Tee("06_quick_linkage.log")


def main():
    root = S.data_paths()["cityflow"]
    cache = Path(os.environ.get("REID_CACHE", S.WORK / "emb_cache"))   # 7-Oct embeddings
    if not cache.exists():
        raise SystemExit(f"{cache} not found - this check needs the 7-October embedding cache")
    out = S.WORK / "quick_linkage_check"
    have = lambda s: any((cache / "gt" / s).glob("c*.npz"))  # noqa: E731
    missing = [s for s in ("S02", "S05") if not have(s)]
    if missing:
        raise SystemExit(f"{cache} has no GT embeddings for {missing}")
    # training scenes are only used for the feasibility window (GT time gaps); no network is trained here
    train_scenes = tuple(s for s in ("S01", "S03", "S04") if have(s))
    log(f"cache: {cache}\nfeasibility window learned from: {train_scenes}")
    t = time.time()
    gt = Data(cache, ["S02", "S05", *train_scenes], tracklet_mode="gt")
    log(f"GT tracklets loaded ({(time.time() - t) / 60:.1f} min)")
    dfs = [run_variant(gt, v, seeds=(0,), device="cpu", out_dir=out, train_scenes=train_scenes, folds=CV_FOLDS,
                       test_scenes=S.TEST_SCENES, gt_data=gt, log=log) for v in ("cosine_single", "cosine")]
    tb = summarize(pd.concat(dfs, ignore_index=True))
    log("\nSetting B (GT boxes), appearance-only baseline\n" + tb.to_string())

    det = Data(cache, ["S02", "S05"], tracklet_mode="det")
    dc = run_setting_c(det, gt, root, ["cosine_single", "cosine"], (0,), out, out, train_scenes=train_scenes,
                       folds=CV_FOLDS, test_scenes=S.TEST_SCENES, device="cpu", log=log)
    cols = ("OVERALL_idf1", "OVERALL_idp", "OVERALL_idr", "OVERALL_mota", "OVERALL_num_switches",
            "OVERALL_num_misses", "OVERALL_num_false_positives", "OVERALL_num_objects", "OVERALL_num_predictions",
            "S02_idf1", "S05_idf1")
    tc = summarize(dc, cols)
    log("\nSetting C (detections), appearance-only baseline\n" + tc.to_string())
    pd.concat([tb.assign(setting="B"), tc.assign(setting="C")]).to_csv(S.RESULTS / "quick_linkage_check.csv")
    log(f"\nDONE in {(time.time() - t) / 60:.1f} min -> results\\quick_linkage_check.csv")


if __name__ == "__main__":
    main()

"""TOMORROW (GPU, can run in a second window once the CityFlow backbone is trained).
Controlled DukeMTMC-reID backbone ablation: same schedule for every row, last epoch reported.

    python 04_run_duke.py                    # 3 configs x seed 0, 30 epochs  (~1-2 h)
    python 04_run_duke.py --seeds 1 2        # add seeds later if time allows (finished runs are kept)
"""
import argparse

import pandas as pd

import settings as S

S.utf8_console()
from mtmc.reid_ablation import CONFIGS, run_duke  # noqa: E402

log = S.Tee("04_duke.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=list(S.DUKE_SEEDS))
    ap.add_argument("--epochs", type=int, default=S.DUKE_EPOCHS)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    duke = S.data_paths()["duke"]
    for seed in a.seeds:
        out = S.WORK / f"duke_seed{seed}"
        done = out / "duke_ablation_runs.csv"
        if done.exists() and len(pd.read_csv(done)) == len(CONFIGS):
            log(f"seed {seed}: done, skipped")
            continue
        run_duke(duke, out, configs=CONFIGS, seeds=(seed,), epochs=a.epochs, monitor_every=10, device=S.device(),
                 workers=a.workers, img_h=S.DUKE_IMG[0], img_w=S.DUKE_IMG[1],
                 pretrained=S.PRETRAINED, log=log)
    runs = pd.concat([pd.read_csv(p) for p in S.WORK.glob("duke_seed*/duke_ablation_runs.csv")])
    curves = pd.concat([pd.read_csv(p) for p in S.WORK.glob("duke_seed*/duke_ablation_curves.csv")])
    runs.to_csv(S.RESULTS / "duke_all_runs.csv", index=False)
    curves.to_csv(S.RESULTS / "duke_curves.csv", index=False)
    tab = runs.groupby("config")[["rank1", "mAP", "auc", "eer"]].agg(["mean", "std", "count"])
    tab.to_csv(S.RESULTS / "duke_table.csv")
    log("\nDukeMTMC-reID controlled ablation (last epoch)\n" + tab.to_string())
    log("DUKE DONE")


if __name__ == "__main__":
    main()

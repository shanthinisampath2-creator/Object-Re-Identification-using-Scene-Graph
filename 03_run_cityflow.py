"""TOMORROW (GPU). All CityFlow experiments, resumable: every stage skips work whose output already exists.

    python 03_run_cityflow.py                 # everything, in the order below
    python 03_run_cityflow.py --stage setC    # one stage only

Stages (time on a mid-range GPU; video decoding dominates 'embed'):
  backbone  vehicle ReID backbone on TRAIN-scene crops ................. 30-45 min
  embed     one pass per video: GT boxes (all scenes) + detections (S01, S02, S05) .. 40-70 min
  setB      Setting B, core variants x 3 seeds (GT boxes) ............. 20-40 min
  tau       tau sensitivity figure with the frozen proposed model ...... 2-5 min
  setC      Setting C, detections + Kalman/Hungarian, official-style MOTA  20-40 min
  protocol  old per-camera protocol vs MTMC protocol ................... 2-5 min
  stats     graph statistics table ...................................... 1-2 min
  hung      Setting B with our Hungarian tracklets (optional) ......... 10-20 min
  extras    extra ablation rows (optional) ............................. 15-25 min
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S

S.utf8_console()
try:  # matplotlib's DLL can be blocked by an app-lock policy; the figure is then skipped (data stays in tau_sweep.csv)
    import matplotlib  # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except Exception as _e:  # noqa: BLE001
    plt = None
    print(f"matplotlib unavailable ({_e.__class__.__name__}) - figures will be skipped, CSVs are still written")
import torch  # noqa: E402

from mtmc.backbone import load_backbone, train_reid  # noqa: E402
from mtmc.evaluate import (graph_stats, mtmc_motmetrics, per_camera_motmetrics, read_aic)  # noqa: E402
from mtmc.experiments import (CORE, CV_FOLDS, EXTRA, Data, context, run_setting_c, run_variant, single_camera_quality,  # noqa: E402
                              summarize, tau_sweep, VARIANTS)
from mtmc.extract import extract_scene  # noqa: E402
from mtmc.assoc import build_graph  # noqa: E402

log = S.Tee("03_cityflow.log")
ROOT = None
DEV = None
TAG = S.SPLIT_TAG
CACHE = S.WORK / f"emb_cache_{TAG}"
BACKBONE = S.WORK / f"cityflow_backbone_{TAG}.pt"
RES_B = S.WORK / f"setting_b_cv_{TAG}"
RES_C = S.WORK / f"setting_c_cv_{TAG}"
RES_H = S.WORK / f"setting_b_hungarian_cv_{TAG}"
ALL_SCENES = list(S.EMBED_SCENES)
_gt = {}


def gt_data():
    if "d" not in _gt:
        _gt["d"] = Data(CACHE, ALL_SCENES, tracklet_mode="gt")
    return _gt["d"]


def final_variant():
    """The proposed variant = better of tgat_lstm / tgat_attn by the cross-validation TUNING score
    (each test scene is judged only by results on the other scene). Fixed in advance for the VeRi run."""
    if S.FINAL_VARIANT:
        return S.FINAL_VARIANT
    rows = [pd.read_csv(RES_B / f"runs_{v}.csv") for v in ("tgat_lstm", "tgat_attn") if (RES_B / f"runs_{v}.csv").exists()]
    if not rows:
        return "tgat_lstm"
    df = pd.concat(rows)
    return df.groupby("variant").tune_idf1.mean().idxmax()


def find_veri():
    """VeRi-776 root = the folder that contains image_train, image_query and image_test (searched under data/)."""
    import json as _j
    p = S.PATHS_FILE
    if p.exists():
        v = _j.loads(p.read_text()).get("veri")
        if v and (Path(v) / "image_train").is_dir():
            return Path(v)
    for d in sorted(S.DATA.rglob("image_train")):
        if d.is_dir() and (d.parent / "image_query").is_dir() and (d.parent / "image_test").is_dir():
            return d.parent
    raise SystemExit("VeRi-776 not found: unzip it under data\\veri so that image_train, image_query and "
                     "image_test are in one folder")


def train_veri_backbone(a):
    from mtmc.backbone import extract_folder, reid_metrics
    root = find_veri()
    log(f"backbone: training on VeRi-776 ({root})")
    model, _ = train_reid(root / "image_train", BACKBONE, epochs=a.backbone_epochs, loss="ce_triplet",
                          triplet_weight=1.0, P=16, K=4, img_h=S.BACKBONE_IMG[0], img_w=S.BACKBONE_IMG[1], seed=0,
                          device=DEV, workers=a.workers, pretrained=S.PRETRAINED, log=log)
    model.eval()
    h, w = S.BACKBONE_IMG
    qf, qp, qc = extract_folder(model, root / "image_query", h, w, DEV, workers=a.workers)
    gf, gp, gc = extract_folder(model, root / "image_test", h, w, DEV, workers=a.workers)
    m = reid_metrics(qf, qp, qc, gf, gp, gc)
    (S.RESULTS / "veri_backbone_metrics.json").write_text(json.dumps({k: float(v) for k, v in m.items()
                                                                       if np.isscalar(v)}, indent=1))
    log("VeRi-776 query/test (last epoch): " + json.dumps({k: round(float(m[k]), 4) for k in ("rank1", "mAP")}))
    del model
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- stages
def stage_backbone(a):
    if BACKBONE.exists():
        log("backbone: exists, skipped")
        return
    if S.BACKBONE_SOURCE == "veri":
        return train_veri_backbone(a)
    crops = S.WORK / f"crops_{TAG}"
    if not (S.WORK / f"crops_{TAG}.done").exists():
        raise SystemExit("crops missing - run 02_prepare_tonight.py first")
    train_reid(crops, BACKBONE, epochs=a.backbone_epochs, loss="ce_triplet", triplet_weight=1.0, P=16, K=4,
               img_h=S.BACKBONE_IMG[0], img_w=S.BACKBONE_IMG[1], seed=0, device=DEV, workers=a.workers,
               pretrained=S.PRETRAINED, log=log)
    log("BACKBONE DONE - if the GPU has >= 12 GB free you can now start RUN_TOMORROW_DUKE.bat in a second window")


def stage_embed(a):
    model, hw = load_backbone(BACKBONE, DEV)
    for s in ALL_SCENES:
        extract_scene(ROOT, s, model, hw, CACHE, device=DEV, bs=a.batch, with_det=s in S.DET_SCENES, log=log)
    del model
    torch.cuda.empty_cache()


def stage_setB(a, variants=None, out=RES_B, mode="gt", same_cam=False):
    variants = variants or S.SETB_VARIANTS or CORE
    data = gt_data() if mode == "gt" else Data(CACHE, ALL_SCENES, tracklet_mode=mode)
    dfs = []
    for v in variants:
        f = out / f"runs_{v}.csv"
        if f.exists():
            dfs.append(pd.read_csv(f))
            log(f"setB[{mode}] {v}: exists, skipped")
            continue
        t = time.time()
        dfs.append(run_variant(data, v, seeds=tuple(a.seeds), epochs=a.assoc_epochs, device=DEV, out_dir=out,
                               train_scenes=S.TRAIN_SCENES, folds=CV_FOLDS, test_scenes=S.TEST_SCENES,
                               same_cam=same_cam, gt_data=gt_data(), log=log))
        log(f"setB[{mode}] {v}: {(time.time() - t) / 60:.1f} min")
    if mode == "gt":     # never drop the extra seeds / ablations written by 07_strengthen.py
        extra = S.WORK / f"setting_b_cv_{TAG}_extra"
        dfs += [pd.read_csv(p) for p in sorted(extra.glob("runs_*_s*.csv"))] if extra.exists() else []
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["variant", "seed"], keep="last")
    name = "setting_b" if mode == "gt" else "setting_b_hungarian"
    df.to_csv(S.RESULTS / f"{name}_all_runs.csv", index=False)
    tab = summarize(df)
    tab.to_csv(S.RESULTS / f"{name}_table.csv")
    log(f"\n{name} (mean ± std over seeds)\n" + tab.to_string())


def stage_tau(a):
    fv = final_variant()
    runs = pd.read_csv(RES_B / f"runs_{fv}.csv")
    row = runs[runs.seed == a.seeds[0]].iloc[0]
    theta = {s: float(row[f"{s}_theta"]) for s in S.TEST_SCENES}
    sw = tau_sweep(gt_data(), fv, a.seeds[0], row, RES_B, train_scenes=S.TRAIN_SCENES, test_scenes=S.TEST_SCENES, folds=CV_FOLDS,
                   device=DEV, out_csv=S.RESULTS / "tau_sweep.csv", log=log)
    for s in S.TEST_SCENES:
        e = sw[sw.scene == s].sort_values("tau").edges_kept.to_numpy()
        assert (np.diff(e) <= 0).all(), "edges must decrease monotonically with tau"
    if plt is None:
        log(f"tau sweep done for {fv} (theta={theta}); figure skipped (no matplotlib) - data in results\\tau_sweep.csv")
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for s, mk in zip(S.TEST_SCENES, ("o", "s")):
        d = sw[sw.scene == s]
        ax[0].plot(d.tau, d.edges_kept, marker=mk, label=s)
        ax[1].plot(d.tau, d.idf1, marker=mk, label=f"IDF1 {s}")
        ax[1].plot(d.tau, d.MER, marker=mk, ls="--", label=f"MER {s}")
        ax[1].plot(d.tau, d.FR, marker=mk, ls=":", label=f"FR {s}")
    ax[0].set(xlabel="graph threshold τ", ylabel="edges kept")
    ax[0].legend()
    ax[1].set(xlabel="graph threshold τ", ylabel="score")
    ax[1].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(S.RESULTS / "fig_tau_sensitivity.pdf")
    plt.savefig(S.RESULTS / "fig_tau_sensitivity.png", dpi=600)
    plt.close()
    log(f"tau sweep done for {fv} (theta={theta}); figure saved")


def stage_setC(a):
    det = Data(CACHE, ["S02", "S05"], tracklet_mode="det")
    for s in ("S02", "S05"):
        log(f"setC tracklets {s}: {len(det.trs[s])}")
    q = single_camera_quality(det, ROOT, S.TEST_SCENES)
    q.to_csv(S.RESULTS / "setting_c_single_camera_quality.csv")
    log("single-camera tracker (per camera)\n" + q.to_string())
    names = list(S.SETTING_C_VARIANTS)
    df = run_setting_c(det, gt_data(), ROOT, names, tuple(a.seeds), RES_B, RES_C, train_scenes=S.TRAIN_SCENES,
                       folds=CV_FOLDS, test_scenes=S.TEST_SCENES, device=DEV, log=log)
    df.to_csv(S.RESULTS / "setting_c_all_runs.csv", index=False)
    cols = ("OVERALL_idf1", "OVERALL_idp", "OVERALL_idr", "OVERALL_mota", "OVERALL_num_switches",
            "OVERALL_num_misses", "OVERALL_num_false_positives", "OVERALL_num_objects", "OVERALL_num_predictions",
            "S02_idf1", "S05_idf1", "S02_mota", "S05_mota")
    tab = summarize(df, cols)
    tab.to_csv(S.RESULTS / "setting_c_table.csv")
    log("\nSetting C (detections, official-style MTMC)\n" + tab.to_string())


def stage_protocol(a):
    fv = final_variant()
    s = S.TEST_SCENES[0]
    pred = read_aic(RES_B / f"pred_{fv}_seed{a.seeds[0]}_{s}.txt")
    broken = pred.copy()
    broken["id"] = broken["cam"] * 100000 + broken["id"]       # every cross-camera link destroyed
    rows = []
    for name, p in [(f"proposed ({fv})", pred), ("cross-camera links destroyed", broken)]:
        pc = per_camera_motmetrics(ROOT, s, p)
        mt = mtmc_motmetrics(ROOT, [s], {s: p}).loc[s]
        rows.append({"prediction": name, "old per-camera mean IDF1": pc.idf1.mean(),
                     "old per-camera mean MOTA": pc.mota.mean(), "MTMC IDF1": mt.idf1, "MTMC MOTA": mt.mota})
    df = pd.DataFrame(rows)
    df.to_csv(S.RESULTS / "protocol_comparison.csv", index=False)
    log("\nprotocol comparison\n" + df.to_string(index=False))


def stage_stats(a):
    window, vocab = context(gt_data(), S.TRAIN_SCENES)
    rows = []
    for s in S.TEST_SCENES:
        for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            g = build_graph(gt_data().trs[s], tau, window=window, campair_vocab=vocab)
            rows.append({"scene": s, **graph_stats(gt_data().trs[s], g, tau)})
    df = pd.DataFrame(rows)
    df.to_csv(S.RESULTS / "graph_stats.csv", index=False)
    (S.RESULTS / "feasibility_window.json").write_text(json.dumps({"window_seconds": window,
                                                                    "n_camera_pairs": len(vocab)}))
    log("\ngraph statistics\n" + df.to_string(index=False))


def stage_hung(a):
    stage_setB(a, variants=["cosine", "tgat_lstm", "tgat_attn"], out=RES_H, mode="hungarian", same_cam=True)


def stage_extras(a):
    stage_setB(a, variants=CORE + EXTRA)


STAGES = {"backbone": stage_backbone, "embed": stage_embed, "setB": stage_setB, "tau": stage_tau,
          "setC": stage_setC, "protocol": stage_protocol, "stats": stage_stats, "hung": stage_hung,
          "extras": stage_extras}
DEFAULT = ["backbone", "embed", "setB", "tau", "setC", "protocol", "stats"]


def main():
    global ROOT, DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="*", default=DEFAULT, help=f"any of {list(STAGES)}")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(S.SEEDS))
    ap.add_argument("--backbone-epochs", type=int, default=S.BACKBONE_EPOCHS)
    ap.add_argument("--assoc-epochs", type=int, default=S.ASSOC_EPOCHS)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ROOT = S.data_paths()["cityflow"]
    DEV = S.device()
    log(f"code version {S.CODE_VERSION}, split {S.SPLIT}, final model {final_variant()}")
    log(f"\n===== {time.strftime('%Y-%m-%d %H:%M')} stages={a.stage} device={DEV} "
        f"{torch.cuda.get_device_name(0) if DEV == 'cuda' else ''}")
    for st in a.stage:
        t = time.time()
        log(f"\n----- stage {st} -----")
        STAGES[st](a)
        log(f"----- stage {st} finished in {(time.time() - t) / 60:.1f} min -----")
    log("ALL REQUESTED STAGES DONE - now run 05_make_report.py")


if __name__ == "__main__":
    main()

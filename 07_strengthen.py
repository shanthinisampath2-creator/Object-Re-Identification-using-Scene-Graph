"""Second GPU day (after the leak-free run). Adds, without touching the finished results:

  seed2   : a third seed (seed 2) for every learned Setting-B variant -> mean ± std over 3 seeds
  attnabl : the mask / cannot-link / single-linkage ablations for the attention version (tgat_attn), 3 seeds
  setCg   : Setting C with a confidence-gated multi-camera filter: an identity is kept only if it spans >= 2
            cameras AND its strongest cross-camera link has probability >= gamma, and tracklets shorter than
            L boxes are dropped. gamma and L are selected on the OTHER scene (same 2-fold CV as everything else).
  merge   : rebuilds results/setting_b_all_runs.csv + setting_b_table.csv from all seeds.

Resumable: finished variants are skipped. Run:  python 07_strengthen.py --workers 0
Then:  python 05_make_report.py
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import settings as S

S.utf8_console()
import torch  # noqa: E402

from mtmc.assoc import AssocNet, cluster, predict_edges  # noqa: E402
from mtmc.evaluate import (fast_idf1_det, gt_box_count, mtmc_motmetrics, predictions_from_tracklets,  # noqa: E402
                           tracklet_level, write_aic)
from mtmc.experiments import (COS_GRID, CV_FOLDS, THETAS, VARIANTS, Data, _fold_tag, _graph, context,  # noqa: E402
                              run_variant, summarize)

log = S.Tee("07_strengthen.log")
TAG = S.SPLIT_TAG
CACHE = S.WORK / f"emb_cache_{TAG}"
RES_B = S.WORK / f"setting_b_cv_{TAG}"            # seeds 0, 1 (finished run)
RES_X = S.WORK / f"setting_b_cv_{TAG}_extra"      # seed 2 + attention ablations
RES_CG = S.WORK / f"setting_c_gated_{TAG}"
LEARNED = ["gat", "gat_lstm", "gat_attn", "tgat", "tgat_lstm", "tgat_attn", "tgat_lstm_nomask", "tgat_lstm_cc",
           "tgat_lstm_single"]
ATTN_ABL = ["tgat_attn_nomask", "tgat_attn_cc", "tgat_attn_single"]
SETC_VARIANTS = ["cosine", "gat_attn", "tgat_lstm", "tgat_attn"]
GATES_COS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)           # on the cosine scale; mapped like THETAS for learned models
MIN_LEN = (0, 10, 20, 40)                             # boxes
DEV = None
_gt = {}


def gt_data():
    if "d" not in _gt:
        _gt["d"] = Data(CACHE, list(S.EMBED_SCENES), tracklet_mode="gt")
    return _gt["d"]


def _run(names, seeds, a):
    RES_X.mkdir(parents=True, exist_ok=True)
    for v in names:
        f = RES_X / f"runs_{v}_s{''.join(map(str, seeds))}.csv"
        if f.exists():
            log(f"{v} seeds {seeds}: exists, skipped")
            continue
        t = time.time()
        df = run_variant(gt_data(), v, seeds=tuple(seeds), epochs=a.assoc_epochs, device=DEV, out_dir=RES_X,
                         train_scenes=S.TRAIN_SCENES, folds=CV_FOLDS, test_scenes=S.TEST_SCENES, gt_data=gt_data(),
                         log=log)
        df.to_csv(f, index=False)
        (RES_X / f"runs_{v}.csv").unlink(missing_ok=True)
        log(f"{v} seeds {seeds}: {(time.time() - t) / 60:.1f} min")


def stage_seed2(a):
    _run(LEARNED, [2], a)


def stage_attnabl(a):
    _run(ATTN_ABL, [0, 1, 2], a)


def stage_merge(a):
    dfs = [pd.read_csv(p) for p in sorted(RES_B.glob("runs_*.csv"))] + \
          [pd.read_csv(p) for p in sorted(RES_X.glob("runs_*_s*.csv"))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["variant", "seed"], keep="last")
    df.to_csv(S.RESULTS / "setting_b_all_runs.csv", index=False)
    tab = summarize(df)
    tab.to_csv(S.RESULTS / "setting_b_table.csv")
    log("\nSetting B, all seeds (mean ± std)\n" + tab.to_string())


def _load_net(name, seed, vocab, tune):
    v = VARIANTS[name]
    if v["enc"] is None:
        return None
    fn = f"assoc_{name}_seed{seed}_{_fold_tag(tune)}.pt"
    p = RES_B / fn if (RES_B / fn).exists() else RES_X / fn
    if not p.exists():
        return "missing"
    net = AssocNet(v["enc"], v["gnn"], n_campairs=len(vocab)).to(DEV)
    net.load_state_dict(torch.load(p, map_location=DEV))
    net.eval()
    return net


def gated(trs, g, p, gid, gamma, min_len):
    """Keep an identity only if it spans >= 2 cameras and its strongest cross-camera link (an edge of the graph
    whose two tracklets received this identity) has probability >= gamma; drop tracklets shorter than min_len."""
    gid = gid.copy()
    lens = np.array([t["n"] for t in trs])
    gid[lens < min_len] = -1
    best = np.full(gid.max() + 2 if len(gid) else 1, -1.0)
    if len(g["I"]):
        I, J = np.asarray(g["I"]), np.asarray(g["J"])
        same = (gid[I] == gid[J]) & (gid[I] >= 0)
        np.maximum.at(best, gid[I][same], np.asarray(p)[same])
    cams = {}
    for t, k in zip(trs, gid):
        if k >= 0:
            cams.setdefault(k, set()).add(t["cam"])
    keep = {k for k, c in cams.items() if len(c) >= 2 and best[k] >= gamma}
    return np.array([k if k in keep else -1 for k in gid], dtype=np.int64)


def stage_setCg(a):
    RES_CG.mkdir(parents=True, exist_ok=True)
    out_csv = S.RESULTS / "setting_c_gated_all_runs.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["variant", "seed"])
    det = Data(CACHE, ["S02", "S05"], tracklet_mode="det")
    window, vocab = context(gt_data(), S.TRAIN_SCENES)
    n_gt = {s: gt_box_count(ROOT, s) for s in S.TEST_SCENES}
    rows = done.to_dict("records")
    for name in SETC_VARIANTS:
        v = VARIANTS[name]
        lin = v.get("linkage", "average")
        for seed in ((0, 1, 2) if v["enc"] is not None else (0,)):
            if ((done.variant == name) & (done.seed == seed)).any():
                log(f"setCg {name} seed {seed}: exists, skipped")
                continue
            t0 = time.time()
            rec, preds, ok = {"variant": name, "seed": seed}, {}, True
            for s in S.TEST_SCENES:
                tune = CV_FOLDS[s]
                net = _load_net(name, seed, vocab, tune)
                if isinstance(net, str):
                    ok = False
                    break
                thetas = (THETAS if net is not None else COS_GRID)[::2]     # every 2nd grid point (speed)
                gates = [0.0] + [float(1 / (1 + np.exp(-10 * (c - 0.5)))) if net is not None else c for c in GATES_COS[1:]]
                best = (-1, None)
                for tau in (0.1, 0.3):                                       # IDF1 is flat in tau (sweep)
                    for t in tune:
                        trs = det.trs[t]
                        g = _graph(trs, tau, v, window, vocab, False)
                        p = predict_edges(net, trs, g, DEV)
                        for th in thetas:
                            gid0 = cluster(trs, g, p, th, v["constrained"], lin)[0]
                            for ga in gates:
                                for L in MIN_LEN:
                                    f = fast_idf1_det(trs, gated(trs, g, p, gid0, ga, L), det.cache[t], n_gt[t])
                                    if f > best[0]:
                                        best = (f, (tau, th, ga, L))
                tau, th, ga, L = best[1]
                rec.update({f"{s}_tau": tau, f"{s}_theta": th, f"{s}_gamma": ga, f"{s}_minlen": L,
                            f"{s}_tune_idf1": best[0]})
                trs = det.trs[s]
                g = _graph(trs, tau, v, window, vocab, False)
                p = predict_edges(net, trs, g, DEV)
                gid = gated(trs, g, p, cluster(trs, g, p, th, v["constrained"], lin)[0], ga, L)
                preds[s] = predictions_from_tracklets(trs, gid, det.cache[s])
                write_aic(preds[s], RES_CG / f"pred_det_{name}_seed{seed}_{s}.txt")
                tl = tracklet_level(trs, g, gid)
                rec.update({f"{s}_{k}": tl[k] for k in ("FR", "MER")})
            if not ok:
                log(f"setCg {name} seed {seed}: model file missing, skipped")
                continue
            mm_df = mtmc_motmetrics(ROOT, list(S.TEST_SCENES), preds)
            for sc, r in mm_df.iterrows():
                for k in ("idf1", "idp", "idr", "mota", "num_switches", "num_misses", "num_false_positives",
                          "num_objects", "num_predictions"):
                    rec[f"{sc}_{k}"] = float(r[k])
            rec["minutes"] = (time.time() - t0) / 60
            log(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items()}))
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    cols = ("OVERALL_idf1", "OVERALL_idp", "OVERALL_idr", "OVERALL_mota", "OVERALL_num_switches",
            "OVERALL_num_misses", "OVERALL_num_false_positives", "OVERALL_num_objects", "OVERALL_num_predictions",
            "S02_idf1", "S05_idf1", "S02_mota", "S05_mota")
    tab = summarize(df, cols)
    tab.to_csv(S.RESULTS / "setting_c_gated_table.csv")
    log("\nSetting C with confidence-gated multi-camera filter\n" + tab.to_string())


STAGES = {"seed2": stage_seed2, "attnabl": stage_attnabl, "merge": stage_merge, "setCg": stage_setCg}


def main():
    global DEV, ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="*", default=["seed2", "attnabl", "merge", "setCg"])
    ap.add_argument("--assoc-epochs", type=int, default=S.ASSOC_EPOCHS)
    ap.add_argument("--workers", type=int, default=0)     # kept for symmetry with 03; nothing here uses workers
    a = ap.parse_args()
    DEV = S.device()
    ROOT = S.data_paths()["cityflow"]
    log(f"\n===== {time.strftime('%Y-%m-%d %H:%M')} stages={a.stage} device={DEV}")
    for st in a.stage:
        t = time.time()
        STAGES[st](a)
        log(f"stage {st} done in {(time.time() - t) / 60:.1f} min")
    log("ALL DONE - now run 05_make_report.py")


ROOT = None
if __name__ == "__main__":
    main()

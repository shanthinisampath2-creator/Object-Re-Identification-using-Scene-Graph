"""Robustness + significance experiments (no retraining: the Setting-B models of each fold are used as trained,
with the tau/theta each run selected on the other scene). Everything goes to results/ and REPORT.md.

  occ_embed : re-embeds the GT boxes of S02/S05 with synthetic PARTIAL OCCLUSION (random-erasing rectangle):
                occ30 = 50% of boxes occluded, 30% of the box area
                occ50 = every box occluded, 50% of the box area
              (one video pass for both levels, GPU, ~40-60 min)
  temporal  : PARTIAL OBSERVATION - every tracklet is seen only during a random contiguous part of its
              lifetime (100 / 50 / 25 / 10 % of its boxes); evaluation still covers every GT box.
  spatial   : the same evaluation on the occluded embeddings (needs occ_embed).
  boot      : paired bootstrap over GT identities (10,000 resamples) of the clean Setting-B result:
              95% confidence interval and p-value of the IDF1 difference vs the appearance-only baseline.

Run:  python 08_robustness.py      then   python 05_make_report.py
"""
import argparse
import time
import zlib

import numpy as np
import pandas as pd

import settings as S

S.utf8_console()
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from mtmc.assoc import AssocNet, cluster, predict_edges  # noqa: E402
from mtmc.common import read_gt, scene_meta  # noqa: E402
from mtmc.evaluate import fast_gtbox_metrics  # noqa: E402
from mtmc.experiments import CV_FOLDS, VARIANTS, Data, _fold_tag, _graph, context  # noqa: E402
from mtmc.extract import _crop_resized, _embed, _frames  # noqa: E402

log = S.Tee("08_robustness.log")
TAG = S.SPLIT_TAG
CACHE = S.WORK / f"emb_cache_{TAG}"
BACKBONE = S.WORK / f"cityflow_backbone_{TAG}.pt"
RES_B = S.WORK / f"setting_b_cv_{TAG}"
RES_X = S.WORK / f"setting_b_cv_{TAG}_extra"
OCC = {"occ30": (0.5, 0.3), "occ50": (1.0, 0.5)}          # name: (fraction of boxes occluded, area fraction)
EVAL_VARIANTS = ["cosine", "gat_attn", "tgat", "tgat_lstm", "tgat_attn"]
RATIOS = (1.0, 0.5, 0.25, 0.1)
DEV = None
_gt = {}


def gt_data():
    if "d" not in _gt:
        _gt["d"] = Data(CACHE, list(S.EMBED_SCENES), tracklet_mode="gt")
    return _gt["d"]


def occlude(crop, rng, area):
    """Random-erasing style occluder: a rectangle of `area` x box area, random aspect and position, filled with
    uniform noise (a different object / clutter in front of the vehicle)."""
    h, w = crop.shape[:2]
    ar = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
    eh = int(min(h, round(np.sqrt(area * h * w * ar))))
    ew = int(min(w, round(area * h * w / max(eh, 1))))
    y0, x0 = rng.integers(0, h - eh + 1), rng.integers(0, w - ew + 1)
    out = crop.copy()
    out[y0:y0 + eh, x0:x0 + ew] = rng.integers(0, 256, (eh, ew, 3), dtype=np.uint8)
    return out


@torch.no_grad()
def stage_occ_embed(a):
    root = S.data_paths()["cityflow"]
    from mtmc.backbone import load_backbone      # imported here: needs torchvision/Pillow (only this stage)
    model, hw = load_backbone(BACKBONE, DEV)
    model.eval()
    for s in S.TEST_SCENES:
        for cam, m in scene_meta(root, s).items():
            dst = {k: S.WORK / f"emb_cache_{TAG}_{k}" / "gt" / s / f"{cam}.npz" for k in OCC}
            if all(p.exists() for p in dst.values()):
                log(f"[occ] cached {s}/{cam}")
                continue
            gt = read_gt(m["gt"])
            by = {}
            for k in range(len(gt)):
                by.setdefault(int(gt[k, 0]), []).append(k)
            emb = {k: np.zeros((len(gt), 512), np.float16) for k in OCC}
            rng = np.random.default_rng(zlib.crc32(f"{s}/{cam}".encode()))   # reproducible
            buf = {k: [] for k in OCC}
            keys = []

            def flush():
                if not keys:
                    return
                for k in OCC:
                    e = _embed(model, buf[k], DEV)
                    emb[k][keys] = e
                    buf[k].clear()
                keys.clear()

            for f, img in _frames(m["video"], max(by) if by else 0, set(by)):
                for k in by[f]:
                    r = gt[k]
                    crop = _crop_resized(img, r[2], r[3], r[4], r[5], hw)
                    for name, (frac, area) in OCC.items():
                        buf[name].append(occlude(crop, rng, area) if rng.random() < frac else crop)
                    keys.append(k)
                    if len(keys) >= a.batch:
                        flush()
            flush()
            for name in OCC:
                dst[name].parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dst[name], frame=gt[:, 0].astype(np.int32), gid=gt[:, 1].astype(np.int32),
                                    xywh=gt[:, 2:6].astype(np.float32), emb=emb[name], offset=m["offset"],
                                    fps=m["fps"])
            log(f"[occ] {s}/{cam}: {len(gt)} boxes")
    del model
    torch.cuda.empty_cache()


def partial_view(trs, cache, ratio, seed=0, seq_len=16):
    """Each tracklet observed only in a random contiguous window covering `ratio` of its boxes.
    Appearance (sampled sequence + mean) comes from the window; timing and the evaluated boxes are unchanged."""
    if ratio >= 1.0:
        return trs
    rng = np.random.default_rng(seed)
    out = []
    for t in trs:
        rows = t["rows"]
        n = len(rows)
        k = max(1, int(round(ratio * n)))
        s0 = int(rng.integers(0, n - k + 1))
        win = rows[s0:s0 + k]
        d = cache[t["cam"]]
        emb = d["emb"][win].astype(np.float32)
        sel = np.arange(k) if k <= seq_len else np.linspace(0, k - 1, seq_len).astype(int)
        fr = d["frame"][win][sel].astype(np.float32) / float(d["fps"])
        mean = emb.mean(0)
        mean /= (np.linalg.norm(mean) + 1e-9)
        out.append({**t, "seq": emb[sel], "seq_t": (fr - fr[0]).astype(np.float32), "mean": mean.astype(np.float32)})
    return out


def _runs():
    p = S.RESULTS / "setting_b_all_runs.csv"
    if not p.exists():
        raise SystemExit("results/setting_b_all_runs.csv missing - run 07_strengthen.py first")
    return pd.read_csv(p)


def _net(name, seed, vocab, tune):
    v = VARIANTS[name]
    if v["enc"] is None:
        return None
    fn = f"assoc_{name}_seed{seed}_{_fold_tag(tune)}.pt"
    p = RES_B / fn if (RES_B / fn).exists() else RES_X / fn
    net = AssocNet(v["enc"], v["gnn"], n_campairs=len(vocab)).to(DEV)
    net.load_state_dict(torch.load(p, map_location=DEV))
    net.eval()
    return net


def evaluate(trs_by_scene, cache_by_scene, condition, rows_out, keep_assign=None):
    window, vocab = context(gt_data(), S.TRAIN_SCENES)
    runs = _runs()
    for name in EVAL_VARIANTS:
        v = VARIANTS[name]
        for _, r in runs[runs.variant == name].iterrows():
            seed = int(r.seed)
            tot_tp = tot_n = 0.0
            rec = {"condition": condition, "variant": name, "seed": seed}
            for s in S.TEST_SCENES:
                net = _net(name, seed, vocab, CV_FOLDS[s])
                trs, cache = trs_by_scene[s], cache_by_scene[s]
                t0 = time.time()
                g = _graph(trs, float(r[f"{s}_tau"]), v, window, vocab, False)
                p = predict_edges(net, trs, g, DEV)
                gid = cluster(trs, g, p, float(r[f"{s}_theta"]), v["constrained"], v.get("linkage", "average"))[0]
                rec[f"{s}_seconds"] = time.time() - t0
                m = fast_gtbox_metrics(trs, gid, cache)
                rec[f"{s}_idf1"] = m["idf1"]
                tot_tp += m["idf1"] * m["n_boxes"]
                tot_n += m["n_boxes"]
                if keep_assign is not None:
                    keep_assign[(name, seed, s)] = (trs, gid, cache)
            rec["OVERALL_idf1"] = tot_tp / max(tot_n, 1)
            rows_out.append(rec)
            log(f"[{condition}] {name} seed {seed}: IDF1 {rec['OVERALL_idf1']:.4f}")


def _table(df, name):
    g = df.groupby(["condition", "variant"]).OVERALL_idf1
    tab = (g.mean().map("{:.4f}".format) + " ± " + g.std().fillna(0).map("{:.4f}".format)).unstack("condition")
    tab = tab.reindex([v for v in EVAL_VARIANTS if v in tab.index])
    tab.to_csv(S.RESULTS / f"{name}_table.csv")
    df.to_csv(S.RESULTS / f"{name}_all_runs.csv", index=False)
    log(f"\n{name}\n" + tab.to_string())
    return tab


def stage_temporal(a):
    rows = []
    gd = gt_data()
    for ratio in RATIOS:
        trs = {s: partial_view(gd.trs[s], gd.cache[s], ratio) for s in S.TEST_SCENES}
        evaluate(trs, gd.cache, f"observed {int(ratio * 100)}%", rows)
    df = pd.DataFrame(rows)
    cond = [f"observed {int(r * 100)}%" for r in RATIOS]
    tab = _table(df, "robust_temporal")
    tab[[c for c in cond if c in tab.columns]].to_csv(S.RESULTS / "robust_temporal_table.csv")


def stage_spatial(a):
    rows = []
    gd = gt_data()
    evaluate({s: gd.trs[s] for s in S.TEST_SCENES}, gd.cache, "clean", rows)
    for name in OCC:
        oc = S.WORK / f"emb_cache_{TAG}_{name}" / "gt"
        if not all(any((oc / s).glob("c*.npz")) for s in S.TEST_SCENES):
            log(f"spatial: {name} embeddings missing (stage occ_embed did not run) - skipped")
            continue
        d = Data(S.WORK / f"emb_cache_{TAG}_{name}", list(S.TEST_SCENES), tracklet_mode="gt")
        evaluate(d.trs, d.cache, name, rows)
    _table(pd.DataFrame(rows), "robust_occlusion")


def per_identity(trs, gid, cache):
    """IDTP and box count per GT identity under the optimal ID matching of this prediction."""
    g_all, p_all = [], []
    for t, k in zip(trs, gid):
        gg = cache[t["cam"]]["gid"][t["rows"]]
        g_all.append(gg)
        p_all.append(np.full(len(gg), k))
    G, P = np.concatenate(g_all), np.concatenate(p_all)
    gu, gi = np.unique(G, return_inverse=True)
    pu, pi = np.unique(P, return_inverse=True)
    C = np.zeros((len(gu), len(pu)))
    np.add.at(C, (gi, pi), 1)
    r, c = linear_sum_assignment(-C)
    tp = np.zeros(len(gu))
    tp[r] = C[r, c]
    return dict(zip(gu.tolist(), tp)), dict(zip(gu.tolist(), C.sum(1)))


def stage_boot(a):
    gd = gt_data()
    assign = {}
    evaluate({s: gd.trs[s] for s in S.TEST_SCENES}, gd.cache, "clean", [], keep_assign=assign)
    prof = {}                                      # variant -> (tp per identity averaged over seeds, n)
    for name in EVAL_VARIANTS:
        keys = [k for k in assign if k[0] == name]
        seeds = sorted({k[1] for k in keys})
        tp_s, n = [], {}
        for seed in seeds:
            tp = {}
            for s in S.TEST_SCENES:
                t_, n_ = per_identity(*assign[(name, seed, s)])
                tp.update({(s, i): v for i, v in t_.items()})
                n.update({(s, i): v for i, v in n_.items()})
            tp_s.append(tp)
        ids = sorted(n)
        prof[name] = (np.array([np.mean([tp.get(i, 0) for tp in tp_s]) for i in ids]), np.array([n[i] for i in ids]))
    rng = np.random.default_rng(0)
    nid = len(prof["cosine"][1])
    B = 10000
    idx = rng.integers(0, nid, (B, nid))
    rows = []
    for a_, b_ in [("tgat_attn", "cosine"), ("gat_attn", "cosine"), ("tgat_lstm", "cosine"), ("tgat", "cosine"),
                   ("tgat_attn", "tgat_lstm"), ("tgat_attn", "gat_attn")]:
        (ta, n), (tb, _) = prof[a_], prof[b_]
        diff = (ta - tb)[idx].sum(1) / n[idx].sum(1)
        point = (ta - tb).sum() / n.sum()
        rows.append({"comparison": f"{a_} - {b_}", "IDF1 difference": point,
                     "95% CI low": np.percentile(diff, 2.5), "95% CI high": np.percentile(diff, 97.5),
                     "p (one-sided, diff <= 0)": float((diff <= 0).mean()), "identities": nid})
    df = pd.DataFrame(rows)
    df.to_csv(S.RESULTS / "bootstrap_significance.csv", index=False)
    log("\npaired bootstrap over GT identities\n" + df.to_string(index=False))


STAGES = {"occ_embed": stage_occ_embed, "temporal": stage_temporal, "spatial": stage_spatial, "boot": stage_boot}


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="*", default=["boot", "temporal", "occ_embed", "spatial"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    DEV = S.device()
    log(f"\n===== {time.strftime('%Y-%m-%d %H:%M')} stages={a.stage} device={DEV}")
    for st in a.stage:
        t = time.time()
        try:
            STAGES[st](a)
        except ImportError as e:
            log(f"stage {st} SKIPPED - a library could not be loaded: {e}")
            continue
        except FileNotFoundError as e:
            log(f"stage {st} SKIPPED - missing input: {e}")
            continue
        log(f"stage {st} done in {(time.time() - t) / 60:.1f} min")
    log("ALL DONE - now run 05_make_report.py")


if __name__ == "__main__":
    main()

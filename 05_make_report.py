"""Collects every result into results/REPORT.md (tables ready to copy into the paper + factual checks)."""
import json

import pandas as pd

import settings as S

S.utf8_console()
R = S.RESULTS


def md(df, index=True, floatfmt=".4f"):
    try:
        return df.to_markdown(index=index, floatfmt=floatfmt)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def mean_std(df, variant, col):
    v = df[df.variant == variant][col]
    return v.mean(), (v.std() if len(v) > 1 else 0.0), len(v)


def compare(df, a, b, col="OVERALL_idf1"):
    if a not in set(df.variant) or b not in set(df.variant):
        return None
    ma, sa, _ = mean_std(df, a, col)
    mb, sb, _ = mean_std(df, b, col)
    d = ma - mb
    verdict = "clearly better" if d > sa + sb else ("clearly worse" if d < -(sa + sb) else "within seed variation")
    return f"- {a} vs {b}: {col} {ma:.4f} vs {mb:.4f} (diff {d:+.4f}) -> **{verdict}**"


def main():
    out = ["# Revision results (auto-generated)\n",
           "Every number below comes from the code in this folder. Copy tables from here into the paper.\n"]
    p = R / "step1_old_protocol_coverage.csv"
    if p.exists():
        out += ["## 1. Why the submitted MOTA was 0.074\n",
                "The V15 evaluation wrote predictions only for <=16 sampled frames per tracklet, so MOTA cannot exceed "
                "this coverage:\n", md(pd.read_csv(p), index=False), "\n"]
    p = R / "veri_backbone_metrics.json"
    if p.exists():
        m = json.loads(p.read_text())
        out += ["## 0. Appearance backbone on VeRi-776 (query/test, last epoch, no model selection)\n",
                f"Rank-1 {m['rank1']:.4f}, mAP {m['mAP']:.4f}, ROC-AUC {m.get('auc', float('nan')):.4f}\n"]
    p = R / "setting_b_all_runs.csv"
    if p.exists():
        df = pd.read_csv(p)
        out += ["## 2. Setting B: cross-camera association with GT boxes (mean ± std over seeds)\n",
                f"Leak-free split: appearance backbone trained on "
                f"{'VeRi-776 (external dataset, no CityFlow vehicle)' if S.BACKBONE_SOURCE == 'veri' else '+'.join(S.BACKBONE_SCENES) + ' only'}; "
                f"association network trained on {'+'.join(S.TRAIN_SCENES)} (vehicles unseen by the backbone); 2-fold cross-validation over "
                "S02/S05: early-stopping epoch, tau and theta for S02 are selected on S05 and vice versa. "
                "Identity clustering: average linkage (rows *_single: single linkage). "
                "FN = FP = 0 by construction, so IDP = IDR = IDF1 and MOTA = 1 - IDSW/GT.\n",
                md(pd.read_csv(R / "setting_b_table.csv", index_col=0)), "\n", "### Factual checks\n"]
        for a, b in [("tgat_lstm", "cosine"), ("tgat_attn", "cosine"), ("tgat_lstm", "gat_lstm"),
                     ("tgat_attn", "gat_attn"), ("tgat_lstm", "tgat"), ("tgat_attn", "tgat"),
                     ("tgat_lstm", "tgat_attn"), ("tgat_lstm", "tgat_lstm_nomask"), ("tgat_lstm", "tgat_lstm_cc"),
                     ("gat", "cosine"), ("tgat_attn", "tgat_attn_nomask"), ("tgat_attn", "tgat_attn_cc"),
                     ("tgat_attn", "tgat_attn_single"), ("tgat_lstm", "tgat_lstm_single"), ("cosine", "cosine_single"),
                     ("gat_attn", "cosine")]:
            c = compare(df, a, b)
            if c:
                out.append(c)
        if "best_epoch" in df.columns:
            be = df.dropna(subset=["best_epoch"]).groupby("variant").best_epoch.apply(lambda x: ", ".join(str(v) for v in x))
            out += ["\nEpoch kept by early stopping on the tuning scene, per seed, shown as S02-fold/S05-fold (0 = the untrained "
                    "model, which equals the cosine baseline; if a variant always keeps 0, its learning did not help):\n", md(be.to_frame()), "\n"]
        sel = df[df.variant.isin(["tgat_lstm", "tgat_attn"])].groupby("variant").tune_idf1.mean()
        if S.FINAL_VARIANT:
            msg = (f"\nFinal model: **{S.FINAL_VARIANT}**"
                   + (" (fixed before this run, from the S04-backbone study)." if S.SPLIT == "veri" else "."))
            if len(sel) and sel.idxmax() != S.FINAL_VARIANT:
                msg += (f" Disclosure: the cross-validation tuning score alone would select {sel.idxmax()} "
                        f"(mean tuning IDF1 {sel.max():.4f}); all variants are reported above.")
            out.append(msg + "\n")
        elif len(sel):
            out.append(f"\nVariant chosen by the cross-validation tuning score (report this one as 'proposed'): "
                       f"**{sel.idxmax()}** (mean tuning IDF1 {sel.max():.4f}).\n")
    for name, title in [("tau_sweep.csv", "## 3. tau sensitivity (one frozen model, fixed embeddings)\n"),
                        ("graph_stats.csv", "## 4. Graph statistics (reviewer minor comment 4)\n")]:
        p = R / name
        if p.exists():
            d = pd.read_csv(p)
            keep = [c for c in ["scene", "tau", "n_boxes", "n_tracklets", "mean_tracklet_len", "all_pairs",
                                "feasible_pairs", "edges_kept", "pos_edges_kept", "components", "mean_comp_size",
                                "max_comp_size", "idf1", "FR", "MER"] if c in d.columns]
            out += [title, md(d[keep], index=False), "\n"]
    if (R / "fig_tau_sensitivity.png").exists():
        out.append("Figure: `fig_tau_sensitivity.pdf` (vector) / `.png` (600 dpi)\n")
    p = R / "feasibility_window.json"
    if p.exists():
        out.append(f"Learned feasibility window (seconds, from S03+S04 positives): {json.loads(p.read_text())}\n")
    p = R / "setting_c_all_runs.csv"
    if p.exists():
        df = pd.read_csv(p)
        out += ["## 5. Setting C: end-to-end MTMCT with detections (official-style motmetrics)\n",
                "Provided detections -> ROI/score filter -> Kalman + Hungarian (cost = 0.5(1-IoU) + 0.5(1-cos)) -> "
                "association (Setting-B model of the same fold) -> identities in >= 2 cameras kept. tau/theta for S02 selected on "
                "S05 detections and vice versa.\n",
                md(pd.read_csv(R / "setting_c_table.csv", index_col=0)), "\n"]
        for a, b in [("tgat_lstm", "cosine"), ("tgat_attn", "cosine"), ("tgat_lstm", "gat_lstm")]:
            c = compare(df, a, b)
            if c:
                out.append(c)
        q = R / "setting_c_single_camera_quality.csv"
        if q.exists():
            out += ["\nSingle-camera tracker quality per camera (before cross-camera linking; note AI City GT only "
                    "labels multi-camera vehicles, so per-camera FP include real unlabelled vehicles):\n",
                    md(pd.read_csv(q, index_col=0)), "\n"]
    p = R / "setting_c_gated_all_runs.csv"
    if p.exists():
        df = pd.read_csv(p)
        out += ["## 5b. Setting C with the confidence-gated multi-camera filter\n",
                "As Section 5, but an identity is output only if it spans >= 2 cameras AND its strongest cross-camera "
                "link has probability >= gamma; tracklets shorter than L boxes are dropped. tau, theta, gamma and L for "
                "S02 are selected on S05 detections and vice versa.\n",
                md(pd.read_csv(R / "setting_c_gated_table.csv", index_col=0)), "\n"]
        for a, b in [("tgat_attn", "cosine"), ("tgat_lstm", "cosine"), ("gat_attn", "cosine"), ("tgat_attn", "gat_attn")]:
            c = compare(df, a, b)
            if c:
                out.append(c)
        sel = [c for c in df.columns if c.endswith(("_gamma", "_minlen", "_theta", "_tau"))]
        out += ["\nSelected parameters per run:\n", md(df[["variant", "seed"] + sel], index=False), "\n"]
    p = R / "protocol_comparison.csv"
    if p.exists():
        out += ["## 6. Old per-camera protocol vs MTMC protocol (same predictions)\n", md(pd.read_csv(p), index=False),
                "\nThe old protocol gives the same score when every cross-camera link is destroyed, so it cannot "
                "measure cross-camera association.\n"]
    p = R / "setting_b_hungarian_table.csv"
    if p.exists():
        out += ["## 7. Setting B with our Hungarian intra-camera tracklets\n", md(pd.read_csv(p, index_col=0)), "\n"]
    p = R / "duke_table.csv"
    if p.exists():
        out += ["## 8. DukeMTMC-reID backbone ablation (identical schedule, last epoch)\n",
                md(pd.read_csv(p, header=[0, 1], index_col=0)), "\n",
                "These rows contain no graph or temporal module; report them as an appearance-backbone ablation.\n"]
    p = R / "bootstrap_significance.csv"
    if p.exists():
        out += ["## 9. Statistical significance (paired bootstrap over GT identities, 10,000 resamples)\n",
                "IDF1 difference on S02+S05 (clean Setting B; learned models averaged over seeds). p = share of "
                "resamples in which the first method is not better.\n", md(pd.read_csv(p), index=False), "\n"]
    p = R / "robust_temporal_table.csv"
    if p.exists():
        out += ["## 10. Robustness to partial observation (no retraining)\n",
                "Every tracklet is observed only during a random contiguous part of its lifetime; all GT boxes are "
                "still evaluated. Overall IDF1, mean ± std over seeds.\n", md(pd.read_csv(p, index_col=0)), "\n"]
    p = R / "robust_occlusion_table.csv"
    if p.exists():
        out += ["## 11. Robustness to partial occlusion (no retraining)\n",
                "GT boxes of S02/S05 re-embedded with a random-erasing occluder: occ30 = 50% of boxes, 30% of the box "
                "area; occ50 = all boxes, 50% of the area. Overall IDF1, mean ± std over seeds.\n",
                md(pd.read_csv(p, index_col=0)), "\n"]
    (R / "REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print(f"written {R / 'REPORT.md'}")


if __name__ == "__main__":
    main()

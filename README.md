# Tracklet Scene Graphs with Transition-Aware Graph Attention for Cross-Camera Vehicle Association

Code and result files for the revised manuscript

> J. Shanthini *et al.*, "Tracklet Scene Graphs with Transition-Aware Graph Attention for Cross-Camera Vehicle
> Association in Non-Overlapping Camera Networks", *Scientific Reports* (under revision).

This repository replaces the notebook released with the first submission. Every number in the revised manuscript
comes from the scripts here, and the exact output files are in `results/` (Study A) and `results_veri/` (Study B).
`MANIFEST.sha256` lists their checksums.

## What the code does

1. **Appearance backbone.** ResNet-50 (ImageNet), last stride 1, BNNeck, 512-D, trained with cross-entropy plus a
   batch-hard triplet loss.
2. **Single-camera tracklets.** These are either ground-truth tracklets (Setting B), our own Hungarian tracker on
   ground-truth boxes (Setting B-H), or a Kalman + Hungarian tracker on the provided detections (Setting C).
3. **Tracklet scene graph.** Nodes are tracklets. A cross-camera edge is kept if the pair lies in the learned
   feasibility time window, is among the k = 20 nearest neighbours and has cosine similarity > τ.
4. **Association network.**
   - A temporal encoder (attention pooling or LSTM) produces each tracklet's node features.
   - A transition-aware GATv2 (edge features + camera-pair embedding) updates them.
   - An edge classifier predicts a residual over the cosine prior (zero-initialised).
5. **Identity clustering.** Average linkage at threshold θ, with cannot-link constraints for the same camera and
   overlapping time.
6. **Evaluation.** Official-style multi-camera evaluation (motmetrics; all cameras of a scene in one accumulator).
   Metrics are IDF1/IDP/IDR/MOTA, reported over 3 seeds and tested with a paired bootstrap.

## Two leak-free studies on CityFlowV2 (AI City Challenge 2022, Track 1)

| | Backbone trained on | Association network trained on | Selection + test |
|---|---|---|---|
| **Study A** (`results/`) | CityFlow S04 | S01 + S03 | S02 ↔ S05 (2-fold cross-validation) |
| **Study B** (`results_veri/`) | VeRi-776 | S01 + S03 + S04 | S02 ↔ S05 (2-fold cross-validation) |

Within a study, no scene is used for more than one role. Epoch, τ and θ are always chosen on the *other* test scene.

## Repository layout

```
settings.py               all paths, splits and hyper-parameters (REID_SPLIT=veri switches to Study B)
01_download_data.py       locates/downloads the datasets and checks the CityFlow layout
02_prepare_tonight.py     CPU: old-protocol coverage diagnosis (Table 4) + backbone training crops
03_run_cityflow.py        main pipeline, stages: backbone embed setB tau setC protocol stats hung
04_run_duke.py            DukeMTMC-reID backbone ablation (Table 12)
05_make_report.py         collects every table into results*/REPORT.md
06_quick_linkage_check.py CPU check: single vs average linkage
07_strengthen.py          extra seed, attention ablation, gated Setting C
08_robustness.py          synthetic occlusion, partial observation, paired bootstrap
mtmc/                     library (backbone, tracklets, graph + association models, clustering, evaluation)
tests/full_pipeline_test.py  end-to-end test on a tiny synthetic dataset (no data download needed)
paper_figures/make_figures.py  Figures 1-3 of the manuscript
tools/collect_results.py  copies a finished run's results into this repository
results/  results_veri/   the result files the paper was written from (REPORT.md + CSV/JSON + logs)
```

## Installation

Python 3.10 or 3.11, an NVIDIA GPU (≥ 12 GB; we used an RTX 4500 Ada, 24 GB).

```
conda create -n mtmc python=3.11 -y
conda activate mtmc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tests/full_pipeline_test.py        # must end with PIPELINE TEST PASSED
```

matplotlib is optional. Without it, everything runs but the τ figure is not drawn.

## Data

The datasets are **not** included and must be obtained from their owners (see `DATA.md`):

* **CityFlowV2 / AIC22 Track 1.** From the AI City Challenge organisers (registration required).
* **VeRi-776.** From its authors, on request (Study B only).
* **DukeMTMC-reID.** Backbone ablation only (Table 12).

Place them under `data/` (or anywhere) and register the paths once:

```
python 01_download_data.py --cityflow <path to AIC22_Track1_MTMC_Tracking> --duke <path to DukeMTMC-reID>
```

For VeRi-776, unzip it under `data/veri/`. The folder that contains `image_train`, `image_query` and `image_test` is
found automatically.

## Reproducing the paper

Every command is resumable: finished stages are skipped when it is run again. On Windows, use `set VAR=value`
instead of `export`. `--workers 0` avoids DataLoader problems on Windows.

**Study A (S04 backbone) → `results/`**

```
python 02_prepare_tonight.py                     # Table 4 + crops of S04              (CPU, ~15 min)
python 03_run_cityflow.py --workers 0            # backbone, embeddings, Setting B, τ, Setting C, protocol, graph stats  (GPU, ~4.5 h)
python 03_run_cityflow.py --stage hung           # Setting B-H (own tracker on GT boxes)
python 07_strengthen.py                          # 3rd seed, attention ablation, gated Setting C
python 08_robustness.py                          # occlusion, partial observation, paired bootstrap
python 04_run_duke.py                            # Table 12 (independent of the rest)
python 05_make_report.py                         # -> results/REPORT.md
```

**Study B (VeRi-776 backbone) → `results_veri/`**

```
export REID_SPLIT=veri                           # Windows: set REID_SPLIT=veri
python 03_run_cityflow.py --workers 0 --seeds 0 1 2
python 07_strengthen.py --stage setCg
python 08_robustness.py
python 05_make_report.py                         # -> results_veri/REPORT.md
```

The final model (`tgat_attn`: transition-aware GAT + attention pooling) was fixed after Study A and *before* Study B
was run (`settings.FINAL_VARIANT`). `REPORT.md` states this, along with the variant that the cross-validation score
alone would have chosen.

## Where each table and figure comes from

| Manuscript | File |
|---|---|
| Table 4: coverage of the previous evaluation | `results/step1_old_protocol_coverage.csv` |
| Table 5: per-camera vs multi-camera evaluation | `results/protocol_comparison.csv` |
| Table 6: scene-graph statistics | `results/graph_stats.csv`, `results/feasibility_window.json` |
| Table 7: Setting B, Study A | `results/setting_b_table.csv` (per seed and fold: `setting_b_all_runs.csv`) |
| Table 8: ablations | `results/setting_b_table.csv`, `results/REPORT.md` §§ ablation |
| Table 9: paired bootstrap | `results/bootstrap_significance.csv` |
| Table 10: Setting B, Study B | `results_veri/setting_b_table.csv`, `results_veri/veri_backbone_metrics.json`, `results_veri/bootstrap_significance.csv` |
| Table 11: relaxing ground truth (B-H, C) | `results/setting_b_hungarian_table.csv`, `results/setting_c_table.csv`, `results/setting_c_gated_table.csv`, `results/setting_c_single_camera_quality.csv` |
| Table 12: DukeMTMC-reID backbone ablation | `results/duke_table.csv` (`duke_all_runs.csv`, `duke_curves.csv`) |
| Figure 2: τ sensitivity | `results/tau_sweep.csv` |
| Figure 3: robustness | `results/robust_occlusion_table.csv`, `results/robust_temporal_table.csv` |

`results*/REPORT.md` gathers all of these in one human-readable file, with the code version used.

## Honest-reporting notes (as in the paper)

* The proposed model is clearly better than appearance-only association with **ground-truth tracklets**: Study A
  gives +1.85 IDF1 (p < 0.001), and Study B gives +1.82 for `tgat` (p = 0.012). In Study B the final `tgat_attn`
  variant does **not** beat the cosine baseline.
* With our own tracklets (Setting B-H) and with detections (Setting C), the advantage is not maintained (Table 11).
  All results are released, including those that do not favour the proposed model.
* The LSTM temporal encoder is worse than the appearance-only baseline in Study A.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licences and are not redistributed here.

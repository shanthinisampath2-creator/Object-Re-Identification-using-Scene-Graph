# Datasets (not redistributed)

| Dataset | Used for | How to obtain |
|---|---|---|
| CityFlowV2: AI City Challenge 2022, Track 1 (MTMC vehicle tracking), scenes S01–S05 | everything except Table 12 | Register with the AI City Challenge organisers (https://www.aicitychallenge.org/) and accept the data licence. |
| VeRi-776 | Study B backbone (`results_veri/`) | Request it from the dataset authors (non-commercial research use). |
| DukeMTMC-reID | Backbone ablation, Table 12 only | The original release was withdrawn by its creators in 2019. We used a public mirror (https://www.kaggle.com/datasets/whurobin/dukemtmcreid). |

## Expected layout

`01_download_data.py --cityflow <dir>` accepts the folder that contains the scene folders
(`train/S01`, `train/S03`, `train/S04`, `validation/S02`, `validation/S05`, ...). Each camera folder needs
`vdo.avi`, `gt/gt.txt`, `roi.jpg` and, for Setting C, `det/det_*.txt`. The script checks this and writes the
resolved paths to `data/data_paths.json`.

For VeRi-776, put it under `data/veri/` so that `image_train/`, `image_query/` and `image_test/` are in one folder.

None of the datasets, the extracted crops or the embedding caches are stored in this repository.

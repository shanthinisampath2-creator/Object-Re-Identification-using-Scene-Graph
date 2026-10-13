"""Controlled DukeMTMC-reID ablation: identical schedule for every row, several seeds, last-epoch
reporting (no model selection on query/gallery). Answers Reviewer 2 comments 6 and minor 7."""
import json
from pathlib import Path

import pandas as pd

from .backbone import extract_folder, reid_metrics, train_reid

CONFIGS = {
    "R1_CE":              dict(loss="ce"),
    "R2_CE+Triplet(1.0)": dict(loss="ce_triplet", triplet_weight=1.0),
    "R3_CE+Triplet(1.2)": dict(loss="ce_triplet", triplet_weight=1.2),
}


def run_duke(duke_root, out_dir, configs=CONFIGS, seeds=(0,), epochs=60, monitor_every=10,
             device="cuda", img_h=256, img_w=128, workers=4, pretrained=True, log=print):
    root = Path(duke_root)
    tr, q, g = root / "bounding_box_train", root / "query", root / "bounding_box_test"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for name, cfg in configs.items():
        for seed in seeds:
            def monitor(model, ep, name=name, seed=seed):
                if ep % monitor_every and ep != epochs:
                    return {}
                model.eval()
                qf, qp, qc = extract_folder(model, q, img_h, img_w, device, workers=workers)
                gf, gp, gc = extract_folder(model, g, img_h, img_w, device, workers=workers)
                m = reid_metrics(qf, qp, qc, gf, gp, gc)
                curves.append({"config": name, "seed": seed, "epoch": ep, **m})
                return {k: m[k] for k in ("rank1", "mAP", "auc")}

            ck = out / f"duke_{name}_seed{seed}.pt"
            _, hist = train_reid(tr, ck, epochs=epochs, seed=seed, device=device, img_h=img_h, img_w=img_w,
                                 workers=workers, eval_fn=monitor, log=log, pretrained=pretrained, **cfg)
            final = [c for c in curves if c["config"] == name and c["seed"] == seed and c["epoch"] == epochs][-1]
            rec = {"config": name, "seed": seed, "epochs": epochs, **cfg,
                   **{k: final[k] for k in ("rank1", "mAP", "auc", "eer")}}
            rows.append(rec)
            log("FINAL " + json.dumps(rec))
            pd.DataFrame(rows).to_csv(out / "duke_ablation_runs.csv", index=False)
            pd.DataFrame(curves).to_csv(out / "duke_ablation_curves.csv", index=False)
    df = pd.DataFrame(rows)
    agg = df.groupby("config")[["rank1", "mAP", "auc", "eer"]].agg(["mean", "std"])
    return df, agg

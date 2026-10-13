"""All paths and knobs in one place. Everything lives under the folder that contains this file
(override with the REID_BASE environment variable). Nothing is written to Google Drive."""
import json
import os
import sys
from pathlib import Path

BASE = Path(os.environ.get("REID_BASE", Path(__file__).resolve().parent))
SPLIT = os.environ.get("REID_SPLIT", "cityflow").lower()     # "cityflow" (backbone on S04) | "veri" (backbone on VeRi-776)
DATA = BASE / "data"                     # datasets (kagglehub cache lives here)
WORK = BASE / "work"                     # crops, embedding cache, models
RESULTS = BASE / ("results_veri" if SPLIT == "veri" else "results")   # tables, figures, REPORT.md
LOGS = BASE / "logs"
for d in (DATA, WORK, RESULTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

PATHS_FILE = DATA / "data_paths.json"

CITYFLOW_KAGGLE = "shanthinisampath1/cityflow-multi-camera-tracking-dataset"
DUKE_KAGGLE = "whurobin/dukemtmcreid"

# --- experiment knobs
# Leak-free scene split: the appearance backbone and the association network never see the same vehicles,
# and neither sees the reported scenes.
#   backbone (vehicle ReID)        : S04            (25 cameras, most identities)
#   association network (GAT/LSTM) : S01 + S03      (vehicles NOT seen by the backbone)
#   selection + test               : S02 <-> S05    (2-fold cross-validation; unseen by both models)
BACKBONE_SCENES = ("S04",)
TRAIN_SCENES = ("S01", "S03")
TEST_SCENES = ("S02", "S05")
EMBED_SCENES = ("S01", "S02", "S03", "S05")   # S04 is only needed for backbone crops
DET_SCENES = ("S02", "S05")
SPLIT_TAG = "bb" + "".join(BACKBONE_SCENES)    # keeps today's (leaky) outputs separate from the new ones
BACKBONE_EPOCHS = 30          # vehicle ReID backbone (CE + triplet), last epoch kept
BACKBONE_IMG = (256, 256)
ASSOC_EPOCHS = 100            # association network (best epoch chosen on the other test scene, never on itself)
SEEDS = (0, 1, 2)
SETTING_C_VARIANTS = ("cosine", "cosine_single", "gat_lstm", "tgat_lstm", "tgat_attn")
DUKE_EPOCHS = 30
DUKE_SEEDS = (0,)
DUKE_IMG = (256, 128)
PRETRAINED = True             # ImageNet weights for ResNet-50 (downloaded once by torchvision)
BACKBONE_SOURCE = "cityflow"
# Final model, fixed after the S04-backbone study (10 Oct 2026): transition-aware GAT + attention pooling.
# The cross-validation tuning score alone would have picked tgat_lstm; REPORT.md states both.
FINAL_VARIANT = os.environ.get("REID_FINAL") or "tgat_attn"
CODE_VERSION = "2026-10-12"
SETB_VARIANTS = None          # None: the full CORE list

if SPLIT == "veri":
    # External, leak-free backbone: VeRi-776 (different city, different vehicles). No CityFlow vehicle is seen by
    # the backbone, so the association network can use S01+S03+S04; S02/S05 stay the 2-fold CV test scenes.
    # The final model was fixed BEFORE this run (from the S04-backbone study): tgat_attn.
    BACKBONE_SOURCE = "veri"
    BACKBONE_SCENES = ()
    TRAIN_SCENES = ("S01", "S03", "S04")
    EMBED_SCENES = ("S01", "S02", "S03", "S04", "S05")
    SPLIT_TAG = "veri"
    FINAL_VARIANT = "tgat_attn"
    SETB_VARIANTS = ["cosine", "cosine_single", "gat_attn", "tgat", "tgat_lstm", "tgat_attn",
                     "tgat_attn_nomask", "tgat_attn_cc", "tgat_attn_single"]
    SETTING_C_VARIANTS = ("cosine", "gat_attn", "tgat_lstm", "tgat_attn")

if os.environ.get("REID_TEST"):   # tiny settings used only by tests/full_pipeline_test.py
    BACKBONE_IMG, DUKE_IMG, PRETRAINED = (64, 64), (64, 64), False


def utf8_console():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def data_paths():
    if not PATHS_FILE.exists():
        raise SystemExit("data/data_paths.json missing - run 01_download_data.py first")
    return json.loads(PATHS_FILE.read_text())


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Tee:
    """Print to console and append to a log file."""

    def __init__(self, name):
        self.f = open(LOGS / name, "a", encoding="utf-8")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

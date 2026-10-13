"""Copy the result files of a finished run into this repository (run once, before `git add`).

    python tools/collect_results.py --src F:\\path\\to\\your\\run\\folder

Copies (small text files only, so the repository stays small):
  <src>/results/       -> results/        Study A (S04 backbone): REPORT.md, every *.csv / *.json, figures
  <src>/results_veri/  -> results_veri/   Study B (VeRi-776 backbone)
  <src>/logs/*.log     -> results/logs/   full console logs of every stage (with the code-version line)
Optional:
  --predictions  also copies the per-run prediction / score files from <src>/work/setting_*  -> predictions/
  --weights      copies the trained models (*.pt) to release_assets/ (git-ignored) for upload as a GitHub Release

Datasets, crops and embedding caches are never copied. A SHA-256 manifest of everything copied is written to
MANIFEST.sha256 so reviewers can check that the uploaded tables are the ones the paper was written from.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT = {".csv", ".json", ".md", ".txt", ".png", ".pdf", ".log"}
MAX_MB_TEXT = 20        # skip anything larger (GitHub warns at 50 MB, refuses at 100 MB)
MAX_MB_PRED = 45


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_tree(src, dst, exts, max_mb, copied, skipped):
    if not src.is_dir():
        print(f"  (missing, skipped) {src}")
        return
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        mb = f.stat().st_size / 2**20
        if mb > max_mb:
            skipped.append((f, mb))
            continue
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        copied.append(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the folder in which 03_run_cityflow.py etc. were run")
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--weights", action="store_true")
    a = ap.parse_args()
    src = Path(a.src)
    copied, skipped = [], []

    copy_tree(src / "results", REPO / "results", TEXT, MAX_MB_TEXT, copied, skipped)
    copy_tree(src / "results_veri", REPO / "results_veri", TEXT, MAX_MB_TEXT, copied, skipped)
    copy_tree(src / "logs", REPO / "results" / "logs", {".log", ".txt"}, MAX_MB_TEXT, copied, skipped)

    if a.predictions:
        for d in sorted((src / "work").glob("setting_*")):
            if d.is_dir():
                copy_tree(d, REPO / "predictions" / d.name, {".csv", ".json", ".txt", ".npz"}, MAX_MB_PRED,
                          copied, skipped)

    if a.weights:
        rel = REPO / "release_assets"
        rel.mkdir(exist_ok=True)
        for f in sorted((src / "work").rglob("*.pt")):
            out = rel / f"{f.parent.name}__{f.name}" if f.parent != src / "work" else rel / f.name
            shutil.copy2(f, out)
            print(f"  weight: {out.name}  ({f.stat().st_size / 2**20:.1f} MB)")

    man = REPO / "MANIFEST.sha256"
    with open(man, "w", encoding="utf-8") as fh:
        for p in sorted(copied):
            fh.write(f"{sha(p)}  {p.relative_to(REPO).as_posix()}\n")
    print(f"\ncopied {len(copied)} files; manifest -> {man.name}")
    for f, mb in skipped:
        print(f"  SKIPPED (too large, {mb:.0f} MB): {f}")
    if not (REPO / "results" / "REPORT.md").exists():
        print("WARNING: results/REPORT.md not found - run 05_make_report.py first")
    if not (REPO / "results_veri" / "REPORT.md").exists():
        print("WARNING: results_veri/REPORT.md not found - run 05_make_report.py with REID_SPLIT=veri first")


if __name__ == "__main__":
    main()

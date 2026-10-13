"""CPU only. 1) Diagnoses the old MOTA = 0.074 from the GT files alone.
2) Writes ReID training crops from the BACKBONE scenes (settings.BACKBONE_SCENES). Both are resumable."""
import time

import settings as S

S.utf8_console()
from mtmc.evaluate import diagnose_old_protocol  # noqa: E402
from mtmc.extract import dump_crops  # noqa: E402

log = S.Tee("02_prepare.log")


def main():
    root = S.data_paths()["cityflow"]
    out = S.RESULTS / "step1_old_protocol_coverage.csv"
    diag = diagnose_old_protocol(root, list(S.TEST_SCENES))
    diag.to_csv(out, index=False)
    log("Old V15 protocol: prediction coverage (= upper bound of MOTA)\n" + diag.to_string(index=False))

    crops = S.WORK / f"crops_{S.SPLIT_TAG}"
    done = S.WORK / f"crops_{S.SPLIT_TAG}.done"
    if done.exists():
        log(f"crops already written: {len(list(crops.glob('*.jpg')))}")
        return
    t = time.time()
    n = dump_crops(root, list(S.BACKBONE_SCENES), crops, stride=5, log=log)
    done.write_text(str(n))
    log(f"crops written: {n} in {(time.time() - t) / 60:.1f} min")
    log("TONIGHT PREPARATION DONE")


if __name__ == "__main__":
    main()

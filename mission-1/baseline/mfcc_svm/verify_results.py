"""Read-only consistency checks for the frozen MFCC baseline results."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
MISSION = HERE.parents[1]
RESULTS = HERE / "results"
SPLIT = MISSION / "validation/split_assignments.csv"
F0_PREDICTIONS = MISSION / "baseline/f0_lr/results/val_predictions.csv"
EXPECTED_HASH = "04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    assert digest(SPLIT) == EXPECTED_HASH
    with SPLIT.open(encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    split = {row["call_id"]: (row["gender"], row["partition"]) for row in split_rows}
    assert len(split_rows) == len(split) == 27985

    features = pd.read_csv(RESULTS / "call_features.csv")
    predictions = pd.read_csv(RESULTS / "val_predictions.csv")
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    assert len(features) == len(set(features.call_id)) == 27985
    assert {row.call_id: (row.gender, row.partition) for row in features.itertuples()} == split
    assert (features.partition == "train").sum() == 22388
    assert (features.partition == "internal_validation").sum() == 5597
    assert int(features.n_segments.sum()) == 442639
    numeric = features.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric).all()

    expected_validation = {key for key, (_, part) in split.items() if part == "internal_validation"}
    assert len(predictions) == len(set(predictions.call_id)) == 5597
    assert set(predictions.call_id) == expected_validation
    assert predictions.prediction.isin(["M", "F"]).all()
    accuracy = float((predictions.gender == predictions.prediction).mean())
    matrix = confusion_matrix(predictions.gender, predictions.prediction, labels=["M", "F"]).tolist()
    assert abs(accuracy - metrics["overall"]["accuracy"]) < 1e-15
    assert matrix == metrics["confusion_matrix"]["matrix"]
    assert metrics["dataset"]["split_sha256"] == EXPECTED_HASH
    assert metrics["preprocessing"]["scaler_fit_samples"] == 22388
    assert metrics["features"]["dimension"] == 54

    f0 = pd.read_csv(F0_PREDICTIONS, usecols=["call_id"])
    assert len(f0) == len(set(f0.call_id)) == 5597
    assert set(f0.call_id) == set(predictions.call_id)
    print(json.dumps({"status": "PASS", "feature_calls": len(features),
                      "reporter_segments": int(features.n_segments.sum()),
                      "validation_predictions": len(predictions),
                      "accuracy_recomputed": accuracy,
                      "confusion_matrix_recomputed": matrix,
                      "split_sha256": EXPECTED_HASH,
                      "scaler_fit_samples": metrics["preprocessing"]["scaler_fit_samples"]},
                     indent=2))


if __name__ == "__main__":
    main()

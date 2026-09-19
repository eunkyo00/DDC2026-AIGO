"""Recalculate persisted full-run metrics; fail clearly when full extraction was not run."""
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix

HERE = Path(__file__).resolve().parent
metrics_path = HERE / "results/metrics.json"
predictions_path = HERE / "results/val_predictions.csv"
if not metrics_path.exists() or not predictions_path.exists():
    raise SystemExit("Full-run metrics are absent. Review results/REPORT.md and benchmark_results.json.")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
with predictions_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 5_597
y_true = [r["gender"] for r in rows]
y_pred = [r["prediction"] for r in rows]
assert abs(accuracy_score(y_true, y_pred) - metrics["overall"]["accuracy"]) < 1e-15
assert confusion_matrix(y_true, y_pred, labels=["M", "F"]).tolist() == metrics["confusion_matrix"]["matrix"]
assert metrics["encoder"]["trainable_parameters"] == 0
print("PASS: 5,597 predictions, Accuracy, confusion matrix, and frozen encoder metadata agree.")

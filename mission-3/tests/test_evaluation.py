import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import TARGETS, load_split, tune, save_predictions

spec = importlib.util.spec_from_file_location("ensemble", Path(__file__).resolve().parents[1] / "ensemble/run_ensemble.py")
ensemble = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ensemble)


class EvaluationTests(unittest.TestCase):
    def test_perfect_predictions_and_round_trip(self):
        y = np.eye(9, dtype=int)
        probs = y * .8 + .1
        _, scores, macro = tune(y, probs)
        self.assertEqual(macro, 1.)
        self.assertEqual(scores, [1.] * 9)
        frame = pd.DataFrame(y, columns=TARGETS)
        frame["file_name"] = [f"{i}.json" for i in range(9)]
        with tempfile.TemporaryDirectory() as tmp:
            save_predictions(Path(tmp), frame, probs)
            with np.load(Path(tmp) / "dev_predictions.npz", allow_pickle=False) as saved:
                ensemble.aligned(saved, saved)
                corrupt = dict(saved)
                corrupt["file_names"] = saved["file_names"][::-1]
                with self.assertRaisesRegex(ValueError, "alignment"):
                    ensemble.aligned(saved, corrupt)

    def test_fixed_threshold_does_not_tune(self):
        y = np.eye(9, dtype=int)
        frame = pd.DataFrame(y, columns=TARGETS)
        frame["file_name"] = [f"{i}.json" for i in range(9)]
        with tempfile.TemporaryDirectory() as tmp:
            save_predictions(Path(tmp), frame, y * .8 + .1, tune_thresholds=False)
            metrics = __import__("json").loads((Path(tmp) / "metrics.json").read_text())
            self.assertEqual(metrics["fixed_threshold"], .5)
            self.assertEqual(metrics["fixed_threshold_macro_f1"], 1.)
            self.assertNotIn("tuned_macro_f1", metrics)

    def test_original_manifest_order_and_overlap_rejection(self):
        frame = pd.DataFrame([{ "file_name": f"{i}.json", "text": "sample",
                               **{target: 0 for target in TARGETS}} for i in range(3)])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frame.to_json(tmp / "training.jsonl", orient="records", lines=True)
            manifest = pd.DataFrame({"file_name": ["2.json", "0.json", "1.json"],
                                     "split": ["train", "dev_valid", "dev_valid"]})
            manifest.to_csv(tmp / "split.csv", index=False)
            train, valid = load_split(tmp, tmp / "split.csv")
            self.assertEqual(valid.file_name.tolist(), ["0.json", "1.json"])
            self.assertEqual(train.file_name.tolist(), ["2.json"])
            pd.concat([manifest, manifest.iloc[:1]]).to_csv(tmp / "split.csv", index=False)
            with self.assertRaises(ValueError):
                load_split(tmp, tmp / "split.csv")


if __name__ == "__main__":
    unittest.main()

"""Blend aligned prediction exports; output remains compatible with this command."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import TARGETS, fresh_dir, tune, save_predictions, write_json, file_sha256


def aligned(a, b):
    for key in ["file_names", "targets", "y_true"]:
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"Prediction alignment mismatch: {key}")
    if list(a["targets"]) != TARGETS:
        raise ValueError("Target order mismatch")
    if len(set(a["file_names"])) != len(a["file_names"]):
        raise ValueError("Duplicate prediction IDs")
    for data in (a, b):
        probs = data["probs"]
        if probs.shape != data["y_true"].shape or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
            raise ValueError("Invalid probabilities")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fixed-threshold", action="store_true",
                   help="Evaluate one pre-specified blend at threshold 0.5; do not search weights or thresholds.")
    p.add_argument("--weight-a", type=float, default=.5,
                   help="Pre-specified weight for --a when --fixed-threshold is set (default: 0.5).")
    args = p.parse_args()
    a, b = np.load(args.a, allow_pickle=False), np.load(args.b, allow_pickle=False)
    aligned(a, b)
    out = fresh_dir(args.out_dir)
    if not 0 <= args.weight_a <= 1:
        raise ValueError("--weight-a must be between 0 and 1")
    frame = pd.DataFrame(a["y_true"], columns=TARGETS)
    frame["file_name"] = a["file_names"]
    if args.fixed_threshold:
        probs = args.weight_a * a["probs"] + (1 - args.weight_a) * b["probs"]
        save_predictions(out, frame, probs, threshold=.5, tune_thresholds=False)
        write_json(out / "blend_config.json", {"a": str(args.a), "b": str(args.b),
            "sha256_a": file_sha256(args.a), "sha256_b": file_sha256(args.b),
            "weight_a": args.weight_a, "weight_b": 1 - args.weight_a,
            "selection": "pre-specified; fixed threshold 0.5"})
        raise SystemExit(0)
    records, best = [], None
    for weight in np.arange(0., 1.01, .05):
        probs = weight * a["probs"] + (1 - weight) * b["probs"]
        _, _, score = tune(a["y_true"], probs)
        records.append({"weight_a": float(weight), "macro_f1": score})
        if best is None or score > best[0]:
            best = score, float(weight), probs
    save_predictions(out, frame, best[2])
    pd.DataFrame(records).to_csv(out / "search.csv", index=False)
    write_json(out / "blend_config.json", {"a": str(args.a), "b": str(args.b),
        "sha256_a": file_sha256(args.a), "sha256_b": file_sha256(args.b),
        "weight_a": best[1], "weight_b": 1 - best[1]})

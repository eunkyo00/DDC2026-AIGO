"""Prefer the original E0 manifest; random seeds alone do not prove identical splits."""
import argparse
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT, TARGETS, fresh_dir, write_json, environment, file_sha256

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/calls")
    p.add_argument("--manifest", type=Path, help="Original E0 split_manifest.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/splits/fixed")
    args = p.parse_args()
    df = pd.read_json(args.data_dir / "training.jsonl", lines=True).sort_values("file_name").reset_index(drop=True)
    if df.file_name.duplicated().any():
        raise ValueError("Duplicate file names")
    if args.manifest:
        split = pd.read_csv(args.manifest)
        source = "original manifest"
    else:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        train, valid = next(splitter.split(df, df[TARGETS]))
        split = pd.concat([df.iloc[train][["file_name"]].assign(split="train"),
                           df.iloc[valid][["file_name"]].assign(split="dev_valid")], ignore_index=True)
        source = "regenerated; historical equivalence NOT confirmed"
    if split.file_name.duplicated().any() or set(split.file_name) != set(df.file_name):
        raise ValueError("Invalid manifest cohort")
    if set(split.split) != {"train", "dev_valid"}:
        raise ValueError("Invalid split values")
    out = fresh_dir(args.out_dir)
    split.to_csv(out / "split_manifest.csv", index=False)
    merged = split.merge(df, on="file_name", validate="one_to_one")
    rates = merged.groupby("split")[TARGETS].mean()
    rates.to_csv(out / "label_rates.csv")
    info = {"source": source, "seed": 42, "counts": split.split.value_counts().to_dict(),
            "sha256": file_sha256(out / "split_manifest.csv"), "environment": environment()}
    write_json(out / "summary.json", info)
    print(info)

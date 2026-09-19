"""Text and target audit only. No raw conversations are included in the report."""
import argparse
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT, TARGETS, fresh_dir, write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/calls")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    out = fresh_dir(args.out_dir)
    train = pd.read_json(args.data_dir / "training.jsonl", lines=True)
    valid = pd.read_json(args.data_dir / "validation.jsonl", lines=True)
    normalized = [df.text.fillna("").str.replace(r"\s+", " ", regex=True).str.strip() for df in (train, valid)]
    info = {"training_rows": len(train), "validation_rows": len(valid),
            "within_duplicates": [int(text.duplicated().sum()) for text in normalized],
            "cross_exact_texts": len(set(normalized[0]) & set(normalized[1])),
            "empty_texts": [int(text.eq("").sum()) for text in normalized],
            "training_label_count_distribution": train[TARGETS].sum(axis=1).value_counts().sort_index().to_dict()}
    train[TARGETS].sum().rename("positive_count").to_csv(out / "training_label_counts.csv")
    write_json(out / "summary.json", info)
    print(info)

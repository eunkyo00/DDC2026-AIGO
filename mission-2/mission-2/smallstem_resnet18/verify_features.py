#!/usr/bin/env python3
"""사전 생성된 log-Mel 특징 폴더의 개수·형태·범위를 검사한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def verify_split(
    feature_root: Path, split: str, split_info: dict, full: bool, feature_variant: str
) -> None:
    split_dir = feature_root / split
    total = 0
    label_counts = {0: 0, 1: 0}
    for index, shard in enumerate(split_info["shards"]):
        features = np.load(split_dir / shard["features"], mmap_mode="r", allow_pickle=False)
        labels = np.load(split_dir / shard["labels"], mmap_mode="r", allow_pickle=False)
        expected = int(shard["samples"])
        expected_shape = (expected, 64, 48 if feature_variant == "first_plus_whole" else 24)
        if features.shape != expected_shape:
            raise ValueError(f"{split} shard {index}: features shape={features.shape}")
        if labels.shape != (expected,):
            raise ValueError(f"{split} shard {index}: labels shape={labels.shape}")
        expected_dtype = np.uint8 if feature_variant == "first_plus_whole" else np.float16
        if features.dtype != expected_dtype or labels.dtype != np.int8:
            raise ValueError(
                f"{split} shard {index}: dtype={features.dtype}/{labels.dtype}"
            )
        if full:
            sample = np.asarray(features)
            sample_labels = np.asarray(labels)
        else:
            # 기본 검사는 3GB 전체를 다시 읽지 않고 각 shard의 앞·중간·끝만 확인한다.
            sample_indices = sorted({0, expected // 2, expected - 1})
            sample = np.asarray(features[sample_indices])
            sample_labels = np.asarray(labels[sample_indices])
        max_value = 255 if feature_variant == "first_plus_whole" else 1
        if not np.isfinite(sample).all() or sample.min() < 0 or sample.max() > max_value:
            raise ValueError(f"{split} shard {index}: 특징값 범위 오류")
        if not np.isin(sample_labels, [0, 1]).all():
            raise ValueError(f"{split} shard {index}: 라벨 오류")
        if full:
            label_counts[0] += int(np.sum(sample_labels == 0))
            label_counts[1] += int(np.sum(sample_labels == 1))
        total += expected
    if total != int(split_info["samples"]):
        raise ValueError(f"{split}: shard 합 {total:,} != metadata {split_info['samples']:,}")
    suffix = f", labels={label_counts}" if full else ""
    print(f"{split}: {total:,} samples, {len(split_info['shards'])} shards{suffix} — OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission 2 precomputed features 검증")
    parser.add_argument("feature_root", type=Path)
    parser.add_argument("--full", action="store_true", help="모든 특징값과 라벨을 읽어 검사")
    args = parser.parse_args()
    metadata_path = args.feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for split in ("Training", "Validation"):
        split_info = metadata["splits"][split]
        if not split_info.get("complete", False):
            raise RuntimeError(f"{split} 생성이 완료되지 않았습니다.")
        verify_split(
            args.feature_root, split, split_info, args.full,
            metadata.get("feature_variant", "first_1p5"),
        )


if __name__ == "__main__":
    main()

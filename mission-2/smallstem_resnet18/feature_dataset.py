"""사전 생성된 Mission 2 log-Mel shard를 읽는 PyTorch Dataset."""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PrecomputedFeatureDataset(Dataset):
    def __init__(self, feature_root: str | Path, split: str):
        self.feature_root = Path(feature_root)
        metadata_path = self.feature_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"전처리 특징 metadata가 없습니다: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("format_version", -1)) != 1:
            raise ValueError(f"지원하지 않는 feature format: {metadata.get('format_version')}")
        self.feature_variant = str(metadata.get("feature_variant", "first_1p5"))
        if self.feature_variant not in ("first_1p5", "first_plus_whole"):
            raise ValueError(f"지원하지 않는 feature variant: {self.feature_variant}")
        try:
            split_info = metadata["splits"][split]
        except KeyError as error:
            raise KeyError(f"metadata에 {split} split이 없습니다.") from error
        if not split_info.get("complete", False):
            raise RuntimeError(f"{split} 특징 생성이 완료되지 않았습니다.")

        self.split = split
        self.split_dir = self.feature_root / split
        self.shards = list(split_info["shards"])
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["samples"])
            self.cumulative.append(total)
        self.length = total
        if self.length != int(split_info["samples"]):
            raise ValueError(f"{split} metadata sample 수가 일치하지 않습니다.")
        self._feature_arrays: dict[int, np.ndarray] = {}
        self._label_arrays: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_feature_arrays"] = {}
        state["_label_arrays"] = {}
        return state

    def _arrays(self, shard_index: int) -> tuple[np.ndarray, np.ndarray]:
        if shard_index not in self._feature_arrays:
            shard = self.shards[shard_index]
            feature_path = self.split_dir / shard["features"]
            label_path = self.split_dir / shard["labels"]
            self._feature_arrays[shard_index] = np.load(feature_path, mmap_mode="r", allow_pickle=False)
            self._label_arrays[shard_index] = np.load(label_path, mmap_mode="r", allow_pickle=False)
        return self._feature_arrays[shard_index], self._label_arrays[shard_index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative, index)
        shard_start = self.cumulative[shard_index - 1] if shard_index else 0
        local_index = index - shard_start
        features, labels = self._arrays(shard_index)
        feature = np.array(features[local_index], dtype=np.float32, copy=True)
        if self.feature_variant == "first_plus_whole":
            feature /= 255.0
        label = int(labels[local_index])
        return torch.from_numpy(feature).unsqueeze(0), torch.tensor(label, dtype=torch.long)

    def set_epoch(self, epoch: int) -> None:
        # raw IterableDataset과 같은 학습 인터페이스를 유지한다.
        del epoch

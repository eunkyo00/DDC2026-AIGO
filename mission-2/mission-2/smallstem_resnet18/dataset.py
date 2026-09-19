"""GitHub Mission 2 manifest를 읽어 실시간으로 음향 특징을 만드는 Dataset.

발화별 파일을 미리 저장하지 않는다. 통화 WAV를 한 번 로드한 뒤 같은 통화의 발화를
연속 처리하여 GitHub README의 crop -> pad/trim -> log-Mel 흐름을 그대로 적용한다.
"""

from __future__ import annotations

import math
import random
import unicodedata
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from github_preprocessing import (
    crop_segment,
    extract_melspectrogram,
    load_audio,
    pad_or_trim_to_length,
)


TARGET_SECONDS = 1.5
N_MELS = 64


def normalized_stem(path: Path) -> str:
    return unicodedata.normalize("NFC", path.stem)


def find_split_dir(data_root: Path, split: str) -> Path:
    direct = data_root / split
    if direct.is_dir():
        return direct
    matches = sorted(
        (path for path in data_root.rglob(split) if path.is_dir()),
        key=lambda path: (len(path.parts), str(path)),
    )
    if not matches:
        raise FileNotFoundError(f"{data_root} 아래에서 {split}/ 폴더를 찾지 못했습니다.")
    return matches[0]


def build_wav_index(data_root: Path, split: str) -> dict[str, Path]:
    split_dir = find_split_dir(data_root, split)
    result: dict[str, Path] = {}
    for path in split_dir.rglob("*.wav"):
        if path.is_file() and not path.name.startswith("._") and "__MACOSX" not in path.parts:
            result[normalized_stem(path)] = path
    return result


class Mission2ManifestDataset(IterableDataset):
    """한 통화의 WAV를 한 번만 읽고 그 안의 발화를 개별 샘플로 반환한다."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        split: str,
        shuffle: bool,
        seed: int = 42,
        limit: int = 0,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.frame = pd.read_csv(
            self.manifest_path,
            encoding="utf-8-sig",
            usecols=["stem", "startAt", "endAt", "speaker"],
        )
        if limit:
            self.frame = self.frame.head(limit).copy()
        self.frame["stem"] = self.frame["stem"].astype(str).map(
            lambda value: unicodedata.normalize("NFC", value)
        )
        grouped = self.frame.groupby("stem", sort=False).indices.values()
        self.call_groups = [indices.tolist() for indices in grouped]
        self.wav_index = build_wav_index(self.data_root, split)
        needed_stems = {str(self.frame.iloc[indices[0]]["stem"]) for indices in self.call_groups}
        missing = sorted(needed_stems - self.wav_index.keys())
        if missing:
            raise FileNotFoundError(f"{split} WAV를 {len(missing)}개 찾지 못했습니다. 예: {missing[:3]}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.frame)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        rng = random.Random(self.seed + self.epoch * 1009 + worker_id * 97)

        group_order = list(range(len(self.call_groups)))
        if self.shuffle:
            rng.shuffle(group_order)
        group_order = group_order[worker_id::num_workers]

        for group_number in group_order:
            row_indices = list(self.call_groups[group_number])
            if self.shuffle:
                rng.shuffle(row_indices)
            first_row = self.frame.iloc[row_indices[0]]
            stem = str(first_row["stem"])
            waveform, sr = load_audio(str(self.wav_index[stem]))
            target_len = int(round(TARGET_SECONDS * sr))

            for row_index in row_indices:
                row = self.frame.iloc[row_index]
                segment = crop_segment(waveform, sr, float(row["startAt"]), float(row["endAt"]))
                segment = pad_or_trim_to_length(segment, target_len)
                mel = extract_melspectrogram(segment, sr, n_mels=N_MELS, to_db=True)
                feature = np.clip((mel.astype(np.float32) + 80.0) / 80.0, 0.0, 1.0)
                yield torch.from_numpy(feature).unsqueeze(0), torch.tensor(int(row["speaker"]), dtype=torch.long)


def steps_per_epoch(dataset: Mission2ManifestDataset, batch_size: int) -> int:
    return math.ceil(len(dataset) / batch_size)

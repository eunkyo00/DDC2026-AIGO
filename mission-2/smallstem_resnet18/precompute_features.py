#!/usr/bin/env python3
"""원본 WAV를 Colab 학습용 log-Mel shard로 한 번만 변환한다.

GitHub 저장소와 동일한 ``crop_segment -> pad_or_trim_to_length ->
extract_melspectrogram`` 함수를 사용한다. first_plus_whole은 전체 발화 특징을
추가하는 별도 실험이다. 각 shard를 저장할 때 진행 상태도 기록하므로
중단된 명령을 같은 인자로 다시 실행하면 마지막으로 완성된 shard부터 이어서 처리한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from github_preprocessing import (
    crop_segment,
    extract_melspectrogram,
    load_audio,
    pad_or_trim_to_length,
)
from multiview_preprocessing import make_multiview_feature, quantize_feature


FORMAT_VERSION = 1
TARGET_SECONDS = 1.5
N_MELS = 64
SOURCE_COMMIT = "c83727276141377cfda7e10a39eaf61630c927a3"
FEATURE_VARIANTS = ("first_1p5", "first_plus_whole")


def normalized_stem(value: str) -> str:
    return unicodedata.normalize("NFC", value)


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
    result: dict[str, Path] = {}
    for path in find_split_dir(data_root, split).rglob("*.wav"):
        if path.is_file() and not path.name.startswith("._") and "__MACOSX" not in path.parts:
            result[normalized_stem(path.stem)] = path
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def make_call_jobs(
    manifest_path: Path, data_root: Path, split: str
) -> tuple[list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], int]:
    frame = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        usecols=["stem", "startAt", "endAt", "speaker"],
    )
    frame["stem"] = frame["stem"].astype(str).map(normalized_stem)
    wav_index = build_wav_index(data_root, split)
    jobs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    missing: list[str] = []

    for stem, rows in frame.groupby("stem", sort=False):
        wav_path = wav_index.get(str(stem))
        if wav_path is None:
            missing.append(str(stem))
            continue
        labels = rows["speaker"].to_numpy(dtype=np.int8, copy=True)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{stem}: speaker는 0 또는 1이어야 합니다.")
        jobs.append(
            (
                str(wav_path),
                rows["startAt"].to_numpy(dtype=np.float64, copy=True),
                rows["endAt"].to_numpy(dtype=np.float64, copy=True),
                labels,
            )
        )

    if missing:
        raise FileNotFoundError(f"{split} WAV {len(missing)}개가 없습니다. 예: {missing[:3]}")
    return jobs, len(frame)


def batched(values: list, size: int) -> list[list]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def process_job_batch(
    payload: tuple[list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], str],
) -> tuple[np.ndarray, np.ndarray, int]:
    jobs, feature_variant = payload
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None

    for wav_path, starts, ends, labels in jobs:
        waveform, sample_rate = load_audio(wav_path)
        target_length = int(round(TARGET_SECONDS * sample_rate))
        call_features: list[np.ndarray] = []
        for start_at, end_at in zip(starts, ends):
            if feature_variant == "first_plus_whole":
                feature = make_multiview_feature(
                    waveform, sample_rate, float(start_at), float(end_at)
                )
            else:
                segment = crop_segment(waveform, sample_rate, float(start_at), float(end_at))
                segment = pad_or_trim_to_length(segment, target_length)
                mel = extract_melspectrogram(segment, sample_rate, n_mels=N_MELS, to_db=True)
                feature = np.clip((mel.astype(np.float32) + 80.0) / 80.0, 0.0, 1.0)
            if expected_shape is None:
                expected_shape = tuple(feature.shape)
            elif tuple(feature.shape) != expected_shape:
                raise ValueError(
                    f"특징 크기가 일정하지 않습니다: {wav_path} {feature.shape} != {expected_shape}"
                )
            call_features.append(
                quantize_feature(feature) if feature_variant == "first_plus_whole"
                else feature.astype(np.float16)
            )
        feature_blocks.append(np.stack(call_features))
        label_blocks.append(labels)

    return np.concatenate(feature_blocks), np.concatenate(label_blocks), len(jobs)


def result_iterator(
    tasks: list[list], workers: int, feature_variant: str
) -> Iterable[tuple[np.ndarray, np.ndarray, int]]:
    payloads = ((task, feature_variant) for task in tasks)
    if workers <= 1:
        return map(process_job_batch, payloads)
    context = mp.get_context("spawn")
    pool = context.Pool(processes=workers)

    def generate():
        try:
            yield from pool.imap(process_job_batch, payloads, chunksize=1)
        finally:
            pool.close()
            pool.join()

    return generate()


def initial_progress(
    split: str, manifest_path: Path, manifest_hash: str, manifest_rows: int,
    total_tasks: int, feature_variant: str,
) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "split": split,
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_hash,
        "manifest_rows": manifest_rows,
        "total_tasks": total_tasks,
        "next_task": 0,
        "samples": 0,
        "complete": False,
        "feature_shape": None,
        "dtype": "uint8" if feature_variant == "first_plus_whole" else "float16",
        "feature_variant": feature_variant,
        "shards": [],
    }


def load_or_create_progress(
    split_dir: Path,
    split: str,
    manifest_path: Path,
    manifest_hash: str,
    manifest_rows: int,
    total_tasks: int,
    feature_variant: str,
) -> dict:
    progress_path = split_dir / "progress.json"
    if not progress_path.exists():
        progress = initial_progress(
            split, manifest_path, manifest_hash, manifest_rows, total_tasks, feature_variant
        )
        atomic_json(progress_path, progress)
        return progress
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    expected = (manifest_hash, manifest_rows, total_tasks)
    actual = (
        progress.get("manifest_sha256"),
        progress.get("manifest_rows"),
        progress.get("total_tasks"),
    )
    if actual != expected:
        raise RuntimeError(
            f"{split_dir}의 기존 진행 정보가 현재 manifest와 다릅니다. "
            "다른 output-dir를 사용하거나 --overwrite를 지정하세요."
        )
    if progress.get("feature_variant", "first_1p5") != feature_variant:
        raise RuntimeError(
            f"{split_dir}의 기존 특징 종류와 요청한 {feature_variant}가 다릅니다. "
            "다른 output-dir를 사용하세요."
        )
    return progress


def save_shard(
    split_dir: Path,
    progress: dict,
    features: np.ndarray,
    labels: np.ndarray,
    next_task: int,
) -> None:
    shard_index = len(progress["shards"])
    feature_name = f"features_{shard_index:04d}.npy"
    label_name = f"labels_{shard_index:04d}.npy"
    atomic_npy(split_dir / feature_name, features)
    atomic_npy(split_dir / label_name, labels.astype(np.int8, copy=False))
    progress["shards"].append(
        {"features": feature_name, "labels": label_name, "samples": int(len(labels))}
    )
    progress["next_task"] = next_task
    progress["samples"] = int(progress["samples"] + len(labels))
    progress["feature_shape"] = list(features.shape[1:])
    atomic_json(split_dir / "progress.json", progress)


def process_split(
    data_root: Path,
    manifest_path: Path,
    output_root: Path,
    split: str,
    workers: int,
    calls_per_task: int,
    shard_size: int,
    overwrite: bool,
    feature_variant: str,
) -> dict:
    split_dir = output_root / split
    if overwrite and split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{split}] manifest와 WAV 인덱스 읽는 중...", flush=True)
    jobs, manifest_rows = make_call_jobs(manifest_path, data_root, split)
    tasks = batched(jobs, calls_per_task)
    manifest_hash = sha256_file(manifest_path)
    progress = load_or_create_progress(
        split_dir, split, manifest_path, manifest_hash, manifest_rows, len(tasks), feature_variant
    )
    if progress["complete"]:
        print(f"[{split}] 이미 완료됨: {progress['samples']:,} samples", flush=True)
        return progress

    start_task = int(progress["next_task"])
    pending_features: list[np.ndarray] = []
    pending_labels: list[np.ndarray] = []
    pending_samples = 0
    pending_start_task = start_task
    started = time.perf_counter()

    print(
        f"[{split}] calls={len(jobs):,} samples={manifest_rows:,} "
        f"resume_task={start_task:,}/{len(tasks):,} workers={workers}",
        flush=True,
    )
    remaining_tasks = tasks[start_task:]
    for offset, (features, labels, _) in enumerate(
        result_iterator(remaining_tasks, workers, feature_variant)
    ):
        absolute_next_task = start_task + offset + 1
        pending_features.append(features)
        pending_labels.append(labels)
        pending_samples += len(labels)

        if pending_samples >= shard_size:
            feature_array = np.concatenate(pending_features)
            label_array = np.concatenate(pending_labels)
            save_shard(split_dir, progress, feature_array, label_array, absolute_next_task)
            elapsed = time.perf_counter() - started
            print(
                f"[{split}] shard={len(progress['shards']):03d} "
                f"saved={progress['samples']:,}/{manifest_rows:,} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )
            pending_features.clear()
            pending_labels.clear()
            pending_samples = 0
            pending_start_task = absolute_next_task

    if pending_samples:
        save_shard(
            split_dir,
            progress,
            np.concatenate(pending_features),
            np.concatenate(pending_labels),
            len(tasks),
        )
    elif pending_start_task == len(tasks):
        progress["next_task"] = len(tasks)

    if int(progress["samples"]) != manifest_rows:
        raise RuntimeError(
            f"{split}: 저장한 sample 수 {progress['samples']:,} != manifest {manifest_rows:,}"
        )
    progress["complete"] = True
    atomic_json(split_dir / "progress.json", progress)
    print(f"[{split}] 완료: {progress['samples']:,} samples", flush=True)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission 2 log-Mel 특징 사전 생성")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--calls-per-task", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--feature-variant", choices=FEATURE_VARIANTS, default="first_1p5")
    args = parser.parse_args()

    if args.workers < 1 or args.calls_per_task < 1 or args.shard_size < 1:
        parser.error("workers, calls-per-task, shard-size는 1 이상이어야 합니다.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = [
        ("Training", args.train_manifest),
        ("Validation", args.valid_manifest),
    ]
    completed: dict[str, dict] = {}
    for split, manifest_path in split_specs:
        completed[split] = process_split(
            args.data_root,
            manifest_path,
            args.output_dir,
            split,
            args.workers,
            args.calls_per_task,
            args.shard_size,
            args.overwrite,
            args.feature_variant,
        )

    metadata = {
        "format_version": FORMAT_VERSION,
        "description": "DCC Mission 2 precomputed normalized log-Mel features",
        "github_source_commit": SOURCE_COMMIT,
        "feature_variant": args.feature_variant,
        "preprocessing": {
            "pipeline": (
                ["crop_segment(startAt, endAt)",
                 "first 1.5 seconds + whole utterance Mel pooled to 24 frames",
                 "extract_melspectrogram(n_mels=64, to_db=True)",
                 "clip((mel + 80) / 80, 0, 1)", "quantize uint8 (round(feature * 255))"]
                if args.feature_variant == "first_plus_whole"
                else ["crop_segment(startAt, endAt)",
                      "pad_or_trim_to_length(1.5 seconds)",
                      "extract_melspectrogram(n_mels=64, to_db=True)",
                      "clip((mel + 80) / 80, 0, 1)"]
            ),
            "target_seconds": TARGET_SECONDS,
            "n_mels": N_MELS,
            "feature_shape": [N_MELS, 48 if args.feature_variant == "first_plus_whole" else 24],
            "storage_dtype": "uint8" if args.feature_variant == "first_plus_whole" else "float16",
        },
        "splits": completed,
    }
    atomic_json(args.output_dir / "metadata.json", metadata)
    total_bytes = sum(path.stat().st_size for path in args.output_dir.rglob("*.npy"))
    print(f"metadata={args.output_dir / 'metadata.json'}")
    print(f"feature_files={total_bytes / (1024 ** 3):.2f} GiB")


if __name__ == "__main__":
    main()

"""Frozen Wav2Vec2 call embeddings and one fixed Logistic Regression baseline."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SOURCE_SAMPLE_RATE = 8_000
MODEL_SAMPLE_RATE = 16_000
CHECKPOINT = "facebook/wav2vec2-base"
REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
HIDDEN_DIM = 768
EXPECTED_CALLS = 27_985
EXPECTED_SEGMENTS = 442_639
EXPECTED_SPLIT_SHA256 = "04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06"
MIN_MODEL_SAMPLES = 400
MAX_CHUNK_SECONDS = 15.0
MAX_CHUNK_SAMPLES = round(MAX_CHUNK_SECONDS * MODEL_SAMPLE_RATE)
FULL_RUNTIME_LIMIT_HOURS = 12.0
BASELINES = {"majority": 0.5318920850455601, "f0": 0.9120957655887082,
             "mfcc": 0.9506878685009826}


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value) -> None:
    """Write small state files with fsync + atomic replace in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def durable_memmap_flush(array: np.memmap, path: Path) -> None:
    array.flush()
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def append_jsonl_durable(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def acquire_cache_lock(cache_dir: Path, purpose: str):
    """Prevent two extraction/evaluation writers from using the same cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    handle = (cache_dir / "run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("Another extraction/evaluation process holds cache/run.lock") from error
    handle.seek(0); handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "purpose": purpose, "started_utc": utc_now()}) + "\n")
    handle.flush(); os.fsync(handle.fileno())
    return handle


def load_jobs(split_path: Path, manifest_path: Path, segments_path: Path):
    split_hash = sha256_file(split_path)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"Frozen split checksum changed: {split_hash}")
    split_rows = list(read_csv(split_path))
    if len(split_rows) != EXPECTED_CALLS or len({r["call_id"] for r in split_rows}) != EXPECTED_CALLS:
        raise ValueError("Frozen split must contain 27,985 unique calls")
    split = {r["call_id"]: (r["gender"], r["partition"]) for r in split_rows}
    counts = Counter(r["partition"] for r in split_rows)
    if counts != {"train": 22_388, "internal_validation": 5_597}:
        raise ValueError(f"Unexpected split sizes: {counts}")
    if set(k for k, v in split.items() if v[1] == "train") & set(k for k, v in split.items() if v[1] == "internal_validation"):
        raise ValueError("Call leakage in frozen split")

    manifests = {r["call_id"]: r for r in read_csv(manifest_path)}
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in read_csv(segments_path):
        call_id = row["file_ref"]
        if row["split"] == "Training" and call_id in split and row["speaker"] == "1":
            if row["matched"] != "True" or row["exceeds_wav"] != "False":
                raise ValueError(f"Invalid indexed caller segment: {call_id}")
            intervals[call_id].append((float(row["start_s"]), float(row["end_s"])))
    if set(split) != set(manifests) or set(split) != set(intervals):
        raise ValueError("Split, call manifest, and caller segment index coverage differ")
    if sum(map(len, intervals.values())) != EXPECTED_SEGMENTS:
        raise ValueError("Expected 442,639 indexed caller segments")

    jobs = []
    for call_id in sorted(split):
        gender, partition = split[call_id]
        manifest = manifests[call_id]
        if (manifest["gender"], manifest["partition"]) != (gender, partition):
            raise ValueError(f"Manifest differs from split: {call_id}")
        if len(intervals[call_id]) != int(manifest["n_segments"]):
            raise ValueError(f"Segment count differs from manifest: {call_id}")
        jobs.append({"call_id": call_id, "gender": gender, "partition": partition,
                     "wav_path": manifest["wav_path"],
                     "wav_relative_path": manifest["wav_relative_path"],
                     "intervals": intervals[call_id],
                     "n_segments": len(intervals[call_id]),
                     "total_duration_s": sum(b - a for a, b in intervals[call_id])})
    return jobs, split_hash


def crop_segment(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start, end = round(start_s * sample_rate), round(end_s * sample_rate)
    if audio.ndim != 1 or start < 0 or end <= start or end > len(audio):
        raise ValueError(f"Invalid segment bounds: {start_s}, {end_s}, samples={len(audio)}")
    return np.asarray(audio[start:end], dtype=np.float32)


def resample_8k_to_16k(samples: np.ndarray) -> np.ndarray:
    if not len(samples):
        raise ValueError("Cannot resample an empty segment")
    result = resample_poly(np.asarray(samples, dtype=np.float32), up=2, down=1,
                           window=("kaiser", 5.0))
    return np.asarray(result, dtype=np.float32)


def masked_temporal_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean [batch,time,dim] without allowing padded frames into the result."""
    mask = attention_mask.to(dtype=hidden.dtype, device=hidden.device).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denominator


def aggregate_segment_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("A call requires at least one successful segment")
    matrix = np.stack(embeddings).astype(np.float32, copy=False)
    result = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    if result.shape != (HIDDEN_DIM,) or not np.isfinite(result).all():
        raise ValueError("Invalid call embedding")
    return result


def freeze_encoder(model: torch.nn.Module) -> tuple[int, int]:
    model.eval()
    model.requires_grad_(False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def load_encoder(model_path: str | None, cache_dir: Path, device: torch.device):
    source = model_path or CHECKPOINT
    kwargs = {"local_files_only": bool(model_path)}
    if not model_path:
        kwargs.update({"revision": REVISION, "cache_dir": str(cache_dir)})
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(source, **kwargs)
    model, loading = Wav2Vec2Model.from_pretrained(source, output_loading_info=True, **kwargs)
    if int(extractor.sampling_rate) != MODEL_SAMPLE_RATE:
        raise ValueError(f"Checkpoint expects {extractor.sampling_rate} Hz")
    if int(model.config.hidden_size) != HIDDEN_DIM:
        raise ValueError(f"Unexpected hidden dimension: {model.config.hidden_size}")
    total, trainable = freeze_encoder(model)
    model.to(device)
    return extractor, model, total, trainable, loading


def encode_segment(samples_16k: np.ndarray, extractor, model, device: torch.device):
    """Last-layer temporal mean; long segments use frame-count weighted chunk means."""
    too_short = len(samples_16k) < MIN_MODEL_SAMPLES
    if too_short:
        samples_16k = np.pad(samples_16k, (0, MIN_MODEL_SAMPLES - len(samples_16k)))
    # Normalize once per complete segment, then divide evenly so a tiny trailing chunk cannot occur.
    normalized = extractor(samples_16k, sampling_rate=MODEL_SAMPLE_RATE,
                           return_tensors="pt").input_values[0]
    n_chunks = math.ceil(len(normalized) / MAX_CHUNK_SAMPLES)
    chunks = torch.tensor_split(normalized, n_chunks)
    weighted = np.zeros(HIDDEN_DIM, dtype=np.float64)
    frames = 0
    with torch.inference_mode():
        for chunk in chunks:
            inputs = chunk.unsqueeze(0).to(device)
            hidden = model(input_values=inputs).last_hidden_state
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != HIDDEN_DIM:
                raise ValueError(f"Unexpected hidden shape: {tuple(hidden.shape)}")
            mean = hidden.mean(dim=1)[0].float().cpu().numpy()
            weighted += mean * hidden.shape[1]
            frames += hidden.shape[1]
            del inputs, hidden
    result = (weighted / frames).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite segment embedding")
    return result, {"frames": frames, "chunks": len(chunks), "too_short": too_short}


def encode_call(job, extractor, model, device: torch.device):
    audio, sample_rate = sf.read(job["wav_path"], dtype="float32", always_2d=False)
    if sample_rate != SOURCE_SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"Unexpected WAV format: {sample_rate} Hz, shape={audio.shape}")
    embeddings, short, chunks = [], 0, 0
    for start_s, end_s in job["intervals"]:
        waveform = crop_segment(audio, sample_rate, start_s, end_s)
        resampled = resample_8k_to_16k(waveform)
        embedding, info = encode_segment(resampled, extractor, model, device)
        embeddings.append(embedding)
        short += int(info["too_short"])
        chunks += int(info["chunks"])
    result = aggregate_segment_embeddings(embeddings)
    return result, {"successful_segments": len(embeddings), "failed_segments": 0,
                    "too_short_segments": short, "model_chunks": chunks}


def select_representative(jobs, n_per_group: int):
    """Deterministic duration quantile coverage for both partitions and genders."""
    selected = []
    for partition in ("train", "internal_validation"):
        for gender in ("M", "F"):
            group = sorted((j for j in jobs if j["partition"] == partition and j["gender"] == gender),
                           key=lambda j: (j["total_duration_s"], j["call_id"]))
            # Interior quantiles avoid letting the extreme calls dominate a small timing sample.
            indices = [round((i + 0.5) * (len(group) - 1) / n_per_group)
                       for i in range(n_per_group)]
            selected.extend(group[i] for i in indices)
    return sorted(selected, key=lambda j: j["call_id"])


def select_smoke(jobs):
    selected = select_representative(jobs, 1)
    shortest_segment_call = min(
        jobs, key=lambda j: (min(end - start for start, end in j["intervals"]), j["call_id"]))
    most_segments_call = max(jobs, key=lambda j: (j["n_segments"], j["call_id"]))
    by_id = {j["call_id"]: j for j in selected + [shortest_segment_call, most_segments_call]}
    return sorted(by_id.values(), key=lambda j: j["call_id"])


def remap_wav_paths(jobs, data_root: Path | None):
    if data_root is not None:
        for job in jobs:
            job["wav_path"] = str(data_root / job["wav_relative_path"])


def validate_audio_availability(jobs):
    missing = [j["wav_path"] for j in jobs if not Path(j["wav_path"]).is_file()]
    if missing:
        examples = "\n".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} selected WAV files are unavailable. Mount the original data volume "
            f"or pass --data-root. Examples:\n{examples}")


def memory_snapshot(device: torch.device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    data = {"peak_rss_bytes": rss_bytes}
    if device.type == "mps":
        data.update({"mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
                     "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory())})
    return data


def versions():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "numpy": np.__version__, "scipy": importlib.metadata.version("scipy"),
            "soundfile": sf.__version__,
            "scikit-learn": importlib.metadata.version("scikit-learn")}


def run_subset(jobs, extractor, model, device, mode: str):
    parameter_versions = [p._version for p in model.parameters()]
    started = time.perf_counter()
    rows, embeddings = [], []
    maxima = memory_snapshot(device)
    for job in jobs:
        embedding, quality = encode_call(job, extractor, model, device)
        embeddings.append(embedding)
        rows.append({**{k: job[k] for k in ("call_id", "gender", "partition", "n_segments", "total_duration_s")},
                     **quality})
        if device.type == "mps":
            torch.mps.synchronize()
        snapshot = memory_snapshot(device)
        maxima = {key: max(maxima.get(key, 0), value) for key, value in snapshot.items()}
    elapsed = time.perf_counter() - started
    if parameter_versions != [p._version for p in model.parameters()]:
        raise RuntimeError("Frozen encoder parameter version changed during inference")
    matrix = np.stack(embeddings)
    if matrix.shape != (len(jobs), HIDDEN_DIM) or not np.isfinite(matrix).all():
        raise ValueError("Subset embeddings failed shape/finiteness checks")

    # End-to-end smoke classifier. It is diagnostic only and is not reported as baseline accuracy.
    train_idx = [i for i, j in enumerate(jobs) if j["partition"] == "train"]
    val_idx = [i for i, j in enumerate(jobs) if j["partition"] == "internal_validation"]
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                                           class_weight=None, random_state=SEED))])
    pipeline.fit(matrix[train_idx], [jobs[i]["gender"] for i in train_idx])
    diagnostic_predictions = pipeline.predict(matrix[val_idx])
    if len(diagnostic_predictions) != len(val_idx):
        raise ValueError("Smoke classifier did not produce one prediction per diagnostic validation call")
    return rows, matrix, {"mode": mode, "calls": len(jobs),
                          "segments": sum(j["n_segments"] for j in jobs),
                          "caller_audio_seconds": sum(j["total_duration_s"] for j in jobs),
                          "elapsed_seconds": elapsed,
                          "seconds_per_call": elapsed / len(jobs),
                          "seconds_per_segment": elapsed / sum(j["n_segments"] for j in jobs),
                          "audio_realtime_factor": elapsed / sum(j["total_duration_s"] for j in jobs),
                          "embedding_shape": list(matrix.shape), "finite": True,
                          "frozen_parameter_versions_unchanged": True,
                          "diagnostic_classifier": "end-to-end completed; accuracy intentionally not used",
                          "memory": maxima}


def initialize_cache(cache_dir: Path, jobs, split_hash: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    fingerprint = hashlib.sha256("\n".join(j["call_id"] for j in jobs).encode()).hexdigest()
    expected = {"format_version": 2, "checkpoint": CHECKPOINT, "revision": REVISION,
                "split_sha256": split_hash, "job_order_sha256": fingerprint,
                "rows": len(jobs), "dimension": HIDDEN_DIM, "dtype": "float32",
                "source_sample_rate": SOURCE_SAMPLE_RATE, "model_sample_rate": MODEL_SAMPLE_RATE,
                "hidden_layer": "last_hidden_state",
                "segment_pooling": "temporal mean", "call_pooling": "equal segment mean",
                "batch_size": 1}
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != expected:
            raise ValueError("Existing cache metadata does not match this run")
        embeddings = np.lib.format.open_memmap(cache_dir / "call_embeddings.npy", mode="r+")
        status = np.lib.format.open_memmap(cache_dir / "status.npy", mode="r+")
    else:
        embeddings = np.lib.format.open_memmap(cache_dir / "call_embeddings.npy", mode="w+",
                                                dtype="float32", shape=(len(jobs), HIDDEN_DIM))
        status = np.lib.format.open_memmap(cache_dir / "status.npy", mode="w+",
                                           dtype="uint8", shape=(len(jobs),))
        status[:] = 0
        durable_memmap_flush(embeddings, cache_dir / "call_embeddings.npy")
        durable_memmap_flush(status, cache_dir / "status.npy")
        atomic_write_json(metadata_path, expected)
    if embeddings.shape != (len(jobs), HIDDEN_DIM) or embeddings.dtype != np.float32:
        raise ValueError("Embedding cache shape/dtype is invalid")
    if status.shape != (len(jobs),) or status.dtype != np.uint8:
        raise ValueError("Status cache shape/dtype is invalid")
    if not np.isin(np.asarray(status), [0, 1, 2]).all():
        raise ValueError("Status cache contains an invalid value")
    return embeddings, status, expected


def load_quality_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                rows[row["call_id"]] = row
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Corrupt quality.jsonl line {line_number}") from error
    return rows


def validate_completed_cache(cache_dir: Path, jobs, embeddings, status, quality_by_id) -> None:
    """Never skip a completed row unless its embedding and quality record are valid."""
    completed = np.flatnonzero(np.asarray(status) == 1)
    if len(completed):
        if not np.isfinite(np.asarray(embeddings[completed])).all():
            raise ValueError("Completed cache contains NaN/inf")
        missing_quality = [jobs[index]["call_id"] for index in completed
                           if jobs[index]["call_id"] not in quality_by_id]
        if missing_quality:
            raise ValueError(f"Completed cache lacks quality rows: {missing_quality[:3]}")


def cache_progress(cache_dir: Path, total_calls: int) -> dict:
    status_path = cache_dir / "status.npy"
    if not status_path.exists():
        return {"completed_calls": 0, "total_calls": total_calls,
                "failed_calls": 0, "pending_calls": total_calls,
                "estimated_remaining_seconds": None}
    status = np.load(status_path, mmap_mode="r")
    if status.shape != (total_calls,):
        raise ValueError("Status cache row count differs from fixed cohort")
    summary = {"completed_calls": int(np.sum(status == 1)), "total_calls": total_calls,
               "failed_calls": int(np.sum(status == 2)),
               "pending_calls": int(np.sum(status == 0))}
    progress_path = cache_dir / "progress.json"
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        summary["estimated_remaining_seconds"] = saved.get("estimated_remaining_seconds")
        summary["last_updated_utc"] = saved.get("updated_utc")
        summary["last_call_id"] = saved.get("last_call_id")
    else:
        summary["estimated_remaining_seconds"] = None
    return summary


def require_storylink(job) -> None:
    wav_path = Path(job["wav_path"])
    if str(wav_path).startswith("/Volumes/STORYLiNK/") and not os.path.ismount("/Volumes/STORYLiNK"):
        raise FileNotFoundError("External volume /Volumes/STORYLiNK is not mounted")
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV is unavailable: {wav_path}")


def group_metric(rows):
    if not rows:
        return {"n": 0, "accuracy": None, "male_n": 0, "female_n": 0}
    return {"n": len(rows), "accuracy": sum(r["gender"] == r["prediction"] for r in rows) / len(rows),
            "male_n": sum(r["gender"] == "M" for r in rows),
            "female_n": sum(r["gender"] == "F" for r in rows)}


def evaluate_full(jobs, embeddings: np.ndarray, quality_rows, split_hash: str, mfcc_predictions: Path):
    train_idx = [i for i, j in enumerate(jobs) if j["partition"] == "train"]
    val_idx = [i for i, j in enumerate(jobs) if j["partition"] == "internal_validation"]
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                                           class_weight=None, random_state=SEED))])
    pipeline.fit(embeddings[train_idx], [jobs[i]["gender"] for i in train_idx])
    predictions = pipeline.predict(embeddings[val_idx])
    scores = pipeline.decision_function(embeddings[val_idx])
    rows = []
    for index, prediction, score in zip(val_idx, predictions, scores):
        job, quality = jobs[index], quality_rows[index]
        rows.append({"call_id": job["call_id"], "gender": job["gender"],
                     "prediction": prediction, "decision_score": float(score),
                     "total_duration_s": job["total_duration_s"], "n_segments": job["n_segments"],
                     "too_short_segments": quality["too_short_segments"],
                     "model_chunks": quality["model_chunks"]})
    if len(rows) != 5_597:
        raise ValueError("Validation prediction count must be 5,597")
    overall = group_metric(rows)
    gender = {g: group_metric([r for r in rows if r["gender"] == g]) for g in ("M", "F")}
    cm = confusion_matrix([r["gender"] for r in rows], predictions, labels=["M", "F"]).tolist()
    duration_bins = [("<8s", 0, 8), ("8–16s", 8, 16), ("16–32s", 16, 32),
                     ("32–64s", 32, 64), (">=64s", 64, math.inf)]
    segment_bins = [("1–5", 1, 6), ("6–10", 6, 11), ("11–20", 11, 21),
                    ("21–40", 21, 41), (">=41", 41, math.inf)]
    by_duration = {name: group_metric([r for r in rows if low <= r["total_duration_s"] < high])
                   for name, low, high in duration_bins}
    by_segments = {name: group_metric([r for r in rows if low <= r["n_segments"] < high])
                   for name, low, high in segment_bins}

    mfcc = {r["call_id"]: r for r in read_csv(mfcc_predictions)}
    if set(mfcc) != {r["call_id"] for r in rows}:
        raise ValueError("MFCC and Wav2Vec2 validation call IDs differ")
    overlap = Counter()
    for row in rows:
        w2v_ok = row["prediction"] == row["gender"]
        mfcc_ok = mfcc[row["call_id"]]["prediction"] == row["gender"]
        overlap["both_correct" if w2v_ok and mfcc_ok else
                "wav2vec2_only_correct" if w2v_ok else
                "mfcc_only_correct" if mfcc_ok else "both_wrong"] += 1
    metrics = {"dataset": {"branch": "mission1/eunkyo", "split_sha256": split_hash,
                            "seed": SEED, "train_calls": len(train_idx),
                            "internal_validation_calls": len(val_idx)},
               "overall": overall, "gender_accuracy": gender,
               "confusion_matrix": {"labels": ["M", "F"], "matrix": cm},
               "prediction_counts": dict(Counter(predictions)),
               "duration_groups": by_duration, "segment_count_groups": by_segments,
               "comparison": {"majority_accuracy": BASELINES["majority"],
                              "f0_accuracy": BASELINES["f0"], "mfcc_accuracy": BASELINES["mfcc"],
                              "wav2vec2_minus_majority": overall["accuracy"] - BASELINES["majority"],
                              "wav2vec2_minus_f0": overall["accuracy"] - BASELINES["f0"],
                              "wav2vec2_minus_mfcc": overall["accuracy"] - BASELINES["mfcc"],
                              "mfcc_error_overlap": dict(overlap)}}
    return rows, metrics


def write_rows(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def render_full_report(metrics: dict) -> str:
    overall = metrics["overall"]["accuracy"]
    male = metrics["gender_accuracy"]["M"]["accuracy"]
    female = metrics["gender_accuracy"]["F"]["accuracy"]
    cm = metrics["confusion_matrix"]["matrix"]
    overlap = metrics["comparison"]["mfcc_error_overlap"]
    duration_lines = "\n".join(
        f'| {name} | {row["n"]:,} | {row["accuracy"] * 100:.3f}% |'
        for name, row in metrics["duration_groups"].items())
    segment_lines = "\n".join(
        f'| {name} | {row["n"]:,} | {row["accuracy"] * 100:.3f}% |'
        for name, row in metrics["segment_count_groups"].items())
    direction = ("A: Frozen SSL이 MFCC보다 높음; partial fine-tuning 검토 가치가 있다."
                 if overall > BASELINES["mfcc"] else
                 "B: Frozen SSL과 MFCC가 1%p 이내; 오류 겹침을 먼저 해석해야 한다."
                 if overall >= BASELINES["mfcc"] - 0.01 else
                 "C: Frozen SSL이 MFCC보다 낮음; resampling, domain mismatch와 pooling 원인을 먼저 검토해야 한다.")
    return f"""# Frozen Wav2Vec2 + Logistic Regression baseline

## 고정 설정과 실행 결과

`facebook/wav2vec2-base` revision `{REVISION}`의 encoder를 완전히 frozen하고 고정
call-level split SHA256 `{metrics['dataset']['split_sha256']}`를 사용했다. Train / Internal
Validation은 {metrics['dataset']['train_calls']:,} / {metrics['dataset']['internal_validation_calls']:,} calls다.
8kHz `speaker=1` 구간을 polyphase 방식으로 16kHz로 resampling했다. 마지막 hidden state의
시간 평균으로 768차원 segment embedding을 만들고 segment embedding을 동일 가중 평균해
call embedding을 만들었다. StandardScaler는 Train에만 fit했고 LogisticRegression은
`C=1`, `solver=lbfgs`, `max_iter=1000`, `class_weight=None`, `random_state=42`다.

encoder parameter는 {metrics['encoder']['total_parameters']:,}개이며 trainable parameter는
**{metrics['encoder']['trainable_parameters']}개**다. `eval()`과 `torch.inference_mode()`를 사용했다.

## 고정 Internal Validation 성능

| Model | Accuracy |
|---|---:|
| Majority Female | {BASELINES['majority'] * 100:.6f}% |
| F0 + Acoustic + Logistic Regression | {BASELINES['f0'] * 100:.6f}% |
| MFCC + RBF SVM | {BASELINES['mfcc'] * 100:.6f}% |
| Frozen Wav2Vec2 + Logistic Regression | **{overall * 100:.6f}%** |

- Male Accuracy: {male * 100:.6f}%
- Female Accuracy: {female * 100:.6f}%
- Majority 대비: {(overall - BASELINES['majority']) * 100:+.6f}%p
- F0 대비: {(overall - BASELINES['f0']) * 100:+.6f}%p
- MFCC 대비: {(overall - BASELINES['mfcc']) * 100:+.6f}%p

행은 실제 `[M,F]`, 열은 예측 `[M,F]`이다.

| | Pred M | Pred F |
|---|---:|---:|
| True M | {cm[0][0]:,} | {cm[0][1]:,} |
| True F | {cm[1][0]:,} | {cm[1][1]:,} |

## 그룹별 결과

| 신고자 총 발화 시간 | n | Accuracy |
|---|---:|---:|
{duration_lines}

| 신고자 segment 수 | n | Accuracy |
|---|---:|---:|
{segment_lines}

`<8s`는 MFCC/F0 REPORT와 같은 bin이며 n이 작으면 일반화하지 않는다.

## MFCC 오류 겹침

- 둘 다 정답: {overlap['both_correct']:,}
- MFCC만 정답: {overlap['mfcc_only_correct']:,}
- Wav2Vec2만 정답: {overlap['wav2vec2_only_correct']:,}
- 둘 다 오답: {overlap['both_wrong']:,}

현재 결과가 지지하는 방향은 **{direction}** 이다. Accuracy와 오류 겹침은 이 고정 Internal
Validation에 한정되며 독립 화자 일반화나 공식 Validation 성능을 뜻하지 않는다.

## 품질과 known risks

embedding 성공/실패 call은 {metrics['quality']['embedding_success_calls']:,} /
{metrics['quality']['embedding_failed_calls']:,}, 성공/실패 segment는
{metrics['quality']['successful_segments']:,} / {metrics['quality']['failed_segments']:,}다.
NaN/inf embedding 값은 {metrics['quality']['nan_embeddings']} / {metrics['quality']['inf_embeddings']}다.

- 약 96% 통화에서 caller/operator annotation 시간이 겹쳐 caller crop에 상담원 음성이
  섞일 수 있다.
- person ID/phone hash가 없어 동일 사람이 양 partition에 있는지 확인할 수 없다.
- 8→16kHz upsampling은 새 고주파 정보를 만들지 않는다. LibriSpeech와 긴급 신고 전화의
  domain mismatch가 있다.
- 15초 초과 segment는 memory 보호를 위해 균등 chunk로 나누어 frame 수 가중 평균한다.
  chunk 경계 문맥 손실은 알려진 근사 오차다.

HuBERT, fine-tuning, augmentation, layer/pooling search, hybrid/ensemble은 실행하지 않았다.
"""


def common_metadata(args, split_hash, total_parameters, trainable_parameters, loading, device):
    return {"dataset": {"branch": "mission1/eunkyo", "fixed_split": str(args.split_path),
                         "split_sha256": split_hash, "expected_calls": EXPECTED_CALLS,
                         "expected_segments": EXPECTED_SEGMENTS, "seed": SEED},
            "encoder": {"checkpoint": CHECKPOINT, "revision": REVISION,
                        "pretraining_source": "LibriSpeech 960 hours, self-supervised audio pretraining",
                        "input_sample_rate": SOURCE_SAMPLE_RATE,
                        "expected_sample_rate": MODEL_SAMPLE_RATE,
                        "hidden_layer": "last_hidden_state", "hidden_dimension": HIDDEN_DIM,
                        "total_parameters": total_parameters,
                        "trainable_parameters": trainable_parameters,
                        "frozen": trainable_parameters == 0, "eval_mode": True,
                        "gradient_mode": "torch.inference_mode",
                        "loading_info": {"missing_keys": sorted(loading.get("missing_keys", [])),
                                         "unexpected_keys": sorted(loading.get("unexpected_keys", []))}},
            "audio": {"resampling": "scipy.signal.resample_poly up=2/down=1, Kaiser beta=5.0",
                      "segment_rule": "speaker=1, indexed start_s/end_s",
                      "short_input_rule": "zero-pad only inputs shorter than 400 samples at 16kHz",
                      "max_model_chunk_seconds": MAX_CHUNK_SECONDS,
                      "long_segment_rule": "non-overlapping chunks; output-frame-count weighted temporal mean"},
            "pooling": {"segment": "mean of last hidden state over time",
                        "call": "equal mean of segment embeddings"},
            "classifier": {"scaler": "StandardScaler fit on Train only",
                           "type": "LogisticRegression", "solver": "lbfgs", "C": 1.0,
                           "max_iter": 1000, "class_weight": None, "random_state": SEED},
            "runtime": {"os": platform.platform(), "device": str(device), "versions": versions()}}


def progress_row(status, total_elapsed: float, last_call_id: str | None) -> dict:
    completed = int(np.sum(status == 1))
    failed = int(np.sum(status == 2))
    remaining = len(status) - completed
    rate = total_elapsed / completed if completed else None
    return {"updated_utc": utc_now(), "completed_calls": completed,
            "total_calls": len(status), "failed_calls": failed,
            "pending_or_retry_calls": remaining,
            "elapsed_extraction_seconds": total_elapsed,
            "seconds_per_completed_call": rate,
            "estimated_remaining_seconds": rate * remaining if rate is not None else None,
            "last_call_id": last_call_id}


def print_progress(row: dict) -> None:
    eta = row["estimated_remaining_seconds"]
    eta_text = "unknown" if eta is None else f"{eta / 3600:.2f}h"
    print(f'[{row["updated_utc"]}] completed={row["completed_calls"]}/{row["total_calls"]} '
          f'failed={row["failed_calls"]} pending_or_retry={row["pending_or_retry_calls"]} '
          f'elapsed={row["elapsed_extraction_seconds"] / 3600:.2f}h eta={eta_text} '
          f'last_call={row["last_call_id"]}', flush=True)


def extract_all(args, jobs, split_hash, extractor, model, device, metadata):
    lock_handle = acquire_cache_lock(args.cache_dir, "extract")
    try:
        return _extract_all_locked(args, jobs, split_hash, extractor, model, device, metadata)
    finally:
        lock_handle.close()


def _extract_all_locked(args, jobs, split_hash, extractor, model, device, metadata):
    embeddings, status, cache_metadata = initialize_cache(args.cache_dir, jobs, split_hash)
    quality_path = args.cache_dir / "quality.jsonl"
    quality_by_id = load_quality_rows(quality_path)
    validate_completed_cache(args.cache_dir, jobs, embeddings, status, quality_by_id)
    progress_path = args.cache_dir / "progress.json"
    previous_elapsed = 0.0
    if progress_path.exists():
        previous_elapsed = float(json.loads(progress_path.read_text(encoding="utf-8")).get(
            "elapsed_extraction_seconds", 0.0))
    started = time.perf_counter()
    initial_completed = int(np.sum(status == 1))
    initial = progress_row(status, previous_elapsed, None)
    print("Resume-safe extraction started. Completed cache rows were validated before skipping.", flush=True)
    print_progress(initial)

    for index, job in enumerate(jobs):
        if status[index] == 1:
            continue
        try:
            require_storylink(job)
            embedding, quality = encode_call(job, extractor, model, device)
            embeddings[index] = embedding
            durable_memmap_flush(embeddings, args.cache_dir / "call_embeddings.npy")
            quality_row = {"call_id": job["call_id"], **quality}
            append_jsonl_durable(quality_path, quality_row)
            quality_by_id[job["call_id"]] = quality_row
            # Status=1 is the commit marker and is persisted only after embedding + quality.
            status[index] = 1
            durable_memmap_flush(status, args.cache_dir / "status.npy")
        except Exception as error:
            failure = {"timestamp_utc": utc_now(), "call_id": job["call_id"],
                       "wav_path": job["wav_path"], "error_type": type(error).__name__,
                       "error": str(error)}
            append_jsonl_durable(args.cache_dir / "failures.jsonl", failure)
            status[index] = 2
            durable_memmap_flush(status, args.cache_dir / "status.npy")
            elapsed = previous_elapsed + time.perf_counter() - started
            state = progress_row(status, elapsed, job["call_id"])
            atomic_write_json(progress_path, state)
            print_progress(state)
            print(f'FATAL extraction error: {json.dumps(failure, ensure_ascii=False)}',
                  file=sys.stderr, flush=True)
            raise RuntimeError("Extraction stopped immediately; reconnect/check storage and rerun the same command") from error

        elapsed = previous_elapsed + time.perf_counter() - started
        state = progress_row(status, elapsed, job["call_id"])
        atomic_write_json(progress_path, state)
        completed_this_run = state["completed_calls"] - initial_completed
        if completed_this_run % args.progress_every == 0 or state["completed_calls"] == len(jobs):
            print_progress(state)

    validate_completed_cache(args.cache_dir, jobs, embeddings, status, quality_by_id)
    if not np.all(status == 1):
        raise RuntimeError("Extraction ended with non-complete cache rows")
    finished = progress_row(status, previous_elapsed + time.perf_counter() - started,
                            jobs[-1]["call_id"])
    atomic_write_json(progress_path, finished)
    summary = {**metadata, "cache": cache_metadata, "extraction": finished,
               "quality": {"embedding_success_calls": len(jobs), "embedding_failed_calls": 0,
                           "successful_segments": sum(quality_by_id[j["call_id"]]["successful_segments"] for j in jobs),
                           "failed_segments": sum(quality_by_id[j["call_id"]]["failed_segments"] for j in jobs),
                           "too_short_segments": sum(quality_by_id[j["call_id"]]["too_short_segments"] for j in jobs),
                           "model_chunks": sum(quality_by_id[j["call_id"]]["model_chunks"] for j in jobs)}}
    atomic_write_json(args.results_dir / "extraction_summary.json", summary)
    print("Extraction complete. Classifier/evaluation has not been run.", flush=True)
    return summary


def evaluate_cached(args, jobs, split_hash):
    lock_handle = acquire_cache_lock(args.cache_dir, "evaluate")
    try:
        return _evaluate_cached_locked(args, jobs, split_hash)
    finally:
        lock_handle.close()


def _evaluate_cached_locked(args, jobs, split_hash):
    if not (args.cache_dir / "metadata.json").exists():
        raise FileNotFoundError("Embedding cache does not exist; run --mode extract first")
    embeddings, status, cache_metadata = initialize_cache(args.cache_dir, jobs, split_hash)
    quality_by_id = load_quality_rows(args.cache_dir / "quality.jsonl")
    validate_completed_cache(args.cache_dir, jobs, embeddings, status, quality_by_id)
    if not np.all(status == 1):
        raise RuntimeError(f"Extraction is incomplete: {json.dumps(cache_progress(args.cache_dir, len(jobs)))}")
    summary_path = args.results_dir / "extraction_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("extraction_summary.json is missing")
    extraction_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    quality_rows = [quality_by_id[j["call_id"]] for j in jobs]
    started = time.perf_counter()
    predictions, metrics = evaluate_full(jobs, np.asarray(embeddings), quality_rows,
                                         split_hash, args.mfcc_predictions)
    for key in ("dataset", "encoder", "audio", "pooling", "classifier", "runtime"):
        if key in extraction_summary:
            metrics[key] = {**extraction_summary[key], **metrics.get(key, {})}
    metrics["quality"] = {"embedding_success_calls": EXPECTED_CALLS, "embedding_failed_calls": 0,
                          "successful_segments": sum(r["successful_segments"] for r in quality_rows),
                          "failed_segments": sum(r["failed_segments"] for r in quality_rows),
                          "too_short_segments": sum(r["too_short_segments"] for r in quality_rows),
                          "model_chunks": sum(r["model_chunks"] for r in quality_rows),
                          "nan_embeddings": int(np.isnan(embeddings).sum()),
                          "inf_embeddings": int(np.isinf(embeddings).sum())}
    metrics["runtime"].update({"evaluation_seconds": time.perf_counter() - started,
                               "cache": cache_metadata})
    write_json(args.results_dir / "metrics.json", metrics)
    write_rows(args.results_dir / "val_predictions.csv", predictions)
    (args.results_dir / "REPORT.md").write_text(render_full_report(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return metrics


def parse_args():
    root = Path(__file__).resolve().parents[3]
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "benchmark", "extract", "evaluate", "status", "full"), default="smoke")
    parser.add_argument("--split-path", type=Path, default=root / "mission-1/validation/split_assignments.csv")
    parser.add_argument("--manifest-path", type=Path, default=root / "mission-1/validation/manifests/calls.csv")
    parser.add_argument("--segments-path", type=Path, default=root / "mission-1/eda/eda_outputs/segments.csv")
    parser.add_argument("--mfcc-predictions", type=Path,
                        default=root / "mission-1/baseline/mfcc_svm/results/val_predictions.csv")
    parser.add_argument("--results-dir", type=Path, default=experiment / "results")
    parser.add_argument("--cache-dir", type=Path, default=experiment / "cache")
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("/private/tmp/ddc-hf-cache"))
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--data-root", type=Path,
                        help="Optional replacement root containing Training/...; manifest is unchanged")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--benchmark-per-group", type=int, default=5,
                        help="Calls per partition/gender; default benchmark has 20 calls")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Print progress every N newly completed calls; cache commits every call")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(SEED); torch.manual_seed(SEED)
    jobs, split_hash = load_jobs(args.split_path, args.manifest_path, args.segments_path)
    remap_wav_paths(jobs, args.data_root)
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")
    if args.mode == "status":
        print(json.dumps(cache_progress(args.cache_dir, len(jobs)), ensure_ascii=False, indent=2,
                         allow_nan=False), flush=True)
        return
    if args.mode == "evaluate":
        evaluate_cached(args, jobs, split_hash)
        return

    device = resolve_device(args.device)
    if args.mode in ("smoke", "benchmark"):
        subset = select_smoke(jobs) if args.mode == "smoke" else select_representative(jobs, args.benchmark_per_group)
        validate_audio_availability(subset)
    else:  # extract or backwards-compatible combined full mode
        try:
            validate_audio_availability(jobs)
        except Exception as error:
            append_jsonl_durable(args.cache_dir / "failures.jsonl", {
                "timestamp_utc": utc_now(), "phase": "preflight",
                "error_type": type(error).__name__, "error": str(error)})
            print(f"FATAL preflight error: {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
            raise
    extractor, model, total, trainable, loading = load_encoder(args.model_path, args.hf_cache_dir, device)
    if trainable != 0:
        raise RuntimeError(f"Encoder has {trainable} trainable parameters")
    metadata = common_metadata(args, split_hash, total, trainable, loading, device)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("smoke", "benchmark"):
        rows, _, measured = run_subset(subset, extractor, model, device, args.mode)
        result = {**metadata, "measurement": measured,
                  "checks": {"wav_loading": "pass", "resampling_8k_to_16k": "pass",
                             "caller_crop": "pass", "forward": "pass", "embedding_shape": "pass",
                             "finite": "pass", "frozen_unchanged": "pass",
                             "call_aggregation": "pass", "diagnostic_classifier": "pass"}}
        if args.mode == "benchmark":
            total_audio = sum(j["total_duration_s"] for j in jobs)
            projected = measured["audio_realtime_factor"] * total_audio
            result["projection"] = {"dataset_calls": EXPECTED_CALLS,
                                    "dataset_segments": EXPECTED_SEGMENTS,
                                    "dataset_caller_audio_seconds": total_audio,
                                    "projected_full_seconds": projected,
                                    "projected_full_hours": projected / 3600,
                                    "runtime_limit_hours": FULL_RUNTIME_LIMIT_HOURS,
                                    "full_run_recommended": projected <= FULL_RUNTIME_LIMIT_HOURS * 3600,
                                    "decision_rule": "Run full only when measured projection is <=12 hours and memory is stable",
                                    "call_embedding_cache_bytes": EXPECTED_CALLS * HIDDEN_DIM * 4,
                                    "segment_embeddings_persisted": False}
        write_json(args.results_dir / f"{args.mode}_results.json", result)
        write_rows(args.results_dir / f"{args.mode}_calls.csv", rows)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return

    extract_all(args, jobs, split_hash, extractor, model, device, metadata)
    if args.mode == "full":
        evaluate_cached(args, jobs, split_hash)


if __name__ == "__main__":
    main()

"""MFCC + one fixed RBF SVC baseline on the frozen Mission 1 call split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import librosa
import numpy as np
import pandas as pd
from scipy.fft import dct
import soundfile as sf
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
SAMPLE_RATE = 8000
N_MFCC = 13
N_MELS = 26
N_FFT = 256
WIN_LENGTH = 200
HOP_LENGTH = 80
FMIN_HZ = 50.0
FMAX_HZ = 3800.0
WINDOW = "hamming"
PRE_EMPHASIS = 0.97
SILENCE_RMS_THRESHOLD = 0.001
HIGH_SILENCE_RATIO = 0.50
SHORT_SEGMENT_S = WIN_LENGTH / SAMPLE_RATE
SVC_CACHE_MB = 512
EXPECTED_SPLIT_SHA256 = "04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06"

SEGMENT_FEATURES = tuple(
    f"mfcc_{index:02d}_{stat}"
    for index in range(1, N_MFCC + 1)
    for stat in ("temporal_mean", "temporal_std")
)
MODEL_FEATURES = tuple(
    f"{name}_segment_{stat}"
    for name in SEGMENT_FEATURES
    for stat in ("mean", "std")
) + ("n_segments", "total_duration_s")

_MEL_FILTER = None
_ANALYSIS_WINDOW = None
_DCT_BASIS = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def analysis_constants():
    global _MEL_FILTER, _ANALYSIS_WINDOW, _DCT_BASIS
    if _MEL_FILTER is None:
        _MEL_FILTER = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
            fmin=FMIN_HZ, fmax=FMAX_HZ, htk=False, norm="slaney",
            dtype=np.float64,
        )
        short_window = np.hamming(WIN_LENGTH)
        left = (N_FFT - WIN_LENGTH) // 2
        _ANALYSIS_WINDOW = np.pad(short_window, (left, N_FFT - WIN_LENGTH - left))
        _DCT_BASIS = dct(np.eye(N_MELS), type=2, axis=0, norm="ortho")[:N_MFCC]
    return _MEL_FILTER, _ANALYSIS_WINDOW, _DCT_BASIS


def frame_signal(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n_frames = 1 + (len(x) - N_FFT) // HOP_LENGTH
    shape = (n_frames, N_FFT)
    strides = (x.strides[0] * HOP_LENGTH, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()


def extract_segments(sample_arrays: list[np.ndarray]):
    """Batch one call's frames, then restore segment boundaries for 26-D summaries."""
    frame_batches, frame_counts, diagnostics = [], [], []
    failures = 0
    for samples in sample_arrays:
        x = np.asarray(samples, dtype=np.float64)
        if not len(x):
            failures += 1
            continue
        raw_frames = frame_signal(x)
        frame_rms = np.sqrt(np.mean(raw_frames * raw_frames, axis=1))
        silence_ratio = float(np.mean(frame_rms < SILENCE_RMS_THRESHOLD))
        emphasized = np.empty_like(x)
        emphasized[0] = x[0]
        emphasized[1:] = x[1:] - PRE_EMPHASIS * x[:-1]
        frames = frame_signal(emphasized)
        frame_batches.append(frames)
        frame_counts.append(len(frames))
        diagnostics.append({
            "duration_s": len(x) / SAMPLE_RATE,
            "short_segment": len(x) < WIN_LENGTH,
            "silence_ratio": silence_ratio,
            "high_silence": silence_ratio >= HIGH_SILENCE_RATIO,
        })
    if not frame_batches:
        return [], [], failures
    frames = np.concatenate(frame_batches, axis=0)
    mel_filter, window, dct_basis = analysis_constants()
    power = np.abs(np.fft.rfft(frames * window, n=N_FFT, axis=1)) ** 2 / N_FFT
    mel_power = np.maximum(power @ mel_filter.T, np.finfo(np.float64).tiny)
    log_mel_db = 10.0 * np.log10(mel_power)
    mfcc = log_mel_db @ dct_basis.T
    vectors, offset = [], 0
    for count in frame_counts:
        segment_mfcc = mfcc[offset:offset + count]
        offset += count
        feature = np.column_stack(
            (segment_mfcc.mean(axis=0), segment_mfcc.std(axis=0))
        ).reshape(-1)
        if feature.shape != (N_MFCC * 2,) or not np.isfinite(feature).all():
            failures += 1
            diagnostics.pop(len(vectors))
            continue
        vectors.append(feature)
    return vectors, diagnostics, failures


def extract_segment(samples: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Synthetic-test convenience wrapper around the call-batched implementation."""
    vectors, diagnostics, failures = extract_segments([samples])
    if failures or len(vectors) != 1:
        raise ValueError("MFCC extraction failed")
    return vectors[0], diagnostics[0]


def aggregate_call(call_id: str, gender: str, partition: str, wav_path: str,
                   intervals: list[tuple[float, float]]) -> dict[str, float | int | str]:
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim != 1 or sample_rate != SAMPLE_RATE:
        raise ValueError(f"Unexpected WAV format for {call_id}: {sample_rate} Hz, shape={audio.shape}")
    samples = []
    for start_s, end_s in intervals:
        start, end = round(start_s * sample_rate), round(end_s * sample_rate)
        if start < 0 or end <= start or end > len(audio):
            raise ValueError(f"Invalid segment bounds for {call_id}: {start_s}, {end_s}")
        samples.append(audio[start:end])
    try:
        vectors, diagnostics, failures = extract_segments(samples)
    except (ValueError, FloatingPointError):
        vectors, diagnostics, failures = [], [], len(samples)
    matrix = np.stack(vectors) if vectors else np.full((1, len(SEGMENT_FEATURES)), np.nan)
    result: dict[str, float | int | str] = {
        "call_id": call_id,
        "gender": gender,
        "partition": partition,
        "n_segments": len(intervals),
        "total_duration_s": float(sum(end - start for start, end in intervals)),
        "successful_segments": len(vectors),
        "failed_segments": failures,
        "mfcc_success": int(bool(vectors)),
        "short_segments": sum(bool(row["short_segment"]) for row in diagnostics),
        "high_silence_segments": sum(bool(row["high_silence"]) for row in diagnostics),
        "mean_silence_ratio": float(np.mean([row["silence_ratio"] for row in diagnostics])) if diagnostics else math.nan,
    }
    for column, name in enumerate(SEGMENT_FEATURES):
        values = matrix[:, column]
        values = values[np.isfinite(values)]
        result[f"{name}_segment_mean"] = float(np.mean(values)) if len(values) else math.nan
        result[f"{name}_segment_std"] = float(np.std(values)) if len(values) else math.nan
    return result


def load_jobs(split_path: Path, manifest_path: Path, segments_path: Path,
              f0_metrics_path: Path, f0_features_path: Path):
    split_hash = sha256_file(split_path)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"Frozen split checksum changed: {split_hash}")
    f0_metrics = json.loads(f0_metrics_path.read_text(encoding="utf-8"))
    if f0_metrics["dataset"]["split_sha256"] != split_hash:
        raise ValueError("F0 baseline used a different split checksum")
    split_rows = list(read_csv(split_path))
    if len(split_rows) != 27985 or len({row["call_id"] for row in split_rows}) != 27985:
        raise ValueError("Frozen split must contain 27,985 unique calls")
    split = {row["call_id"]: (row["gender"], row["partition"]) for row in split_rows}
    f0_membership = {
        row["call_id"]: (row["gender"], row["partition"])
        for row in read_csv(f0_features_path)
    }
    if f0_membership != split:
        raise ValueError("F0 and MFCC membership metadata differ")
    manifests = {row["call_id"]: row for row in read_csv(manifest_path)}
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in read_csv(segments_path):
        key = row["file_ref"]
        if row["split"] == "Training" and key in split and row["speaker"] == "1":
            intervals[key].append((float(row["start_s"]), float(row["end_s"])))
    if set(split) != set(manifests) or set(split) != set(intervals):
        raise ValueError("Split, manifest, and reporter segment index coverage differ")
    jobs = []
    for key in sorted(split):
        gender, partition = split[key]
        manifest = manifests[key]
        if (manifest["gender"], manifest["partition"]) != (gender, partition):
            raise ValueError(f"Frozen manifest mismatch for {key}")
        jobs.append((key, gender, partition, manifest["wav_path"], intervals[key]))
    return jobs, split_hash


def group_metric(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "accuracy": None, "male_n": 0, "female_n": 0}
    return {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(frame.gender, frame.prediction)),
        "male_n": int((frame.gender == "M").sum()),
        "female_n": int((frame.gender == "F").sum()),
    }


def fit_and_evaluate(features: pd.DataFrame, split_hash: str, f0_predictions_path: Path):
    train = features[features.partition == "train"].copy()
    valid = features[features.partition == "internal_validation"].copy()
    if (len(train), len(valid)) != (22388, 5597):
        raise ValueError(f"Unexpected split sizes: {len(train)}, {len(valid)}")
    x_train = train.loc[:, MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    x_valid = valid.loc[:, MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    if x_train.isna().all().any():
        raise ValueError("A model feature is entirely missing in Train")

    # One resource-only fit on a deterministic prefix; it does not inspect Validation.
    benchmark_n = min(5000, len(train))
    benchmark = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight=None, cache_size=SVC_CACHE_MB)),
    ])
    benchmark_started = time.perf_counter()
    benchmark.fit(x_train.iloc[:benchmark_n], train.gender.iloc[:benchmark_n])
    benchmark_seconds = time.perf_counter() - benchmark_started
    projected_seconds = benchmark_seconds * (len(train) / benchmark_n) ** 2

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight=None, cache_size=SVC_CACHE_MB)),
    ])
    fit_started = time.perf_counter()
    pipeline.fit(x_train, train.gender)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    prediction = pipeline.predict(x_valid)
    predict_seconds = time.perf_counter() - predict_started
    pred = valid[["call_id", "gender", "total_duration_s", "n_segments", "mfcc_success",
                  "failed_segments", "short_segments", "high_silence_segments",
                  "mean_silence_ratio"]].copy()
    pred["prediction"] = prediction
    pred["decision_score"] = pipeline.decision_function(x_valid)

    duration_edges = [0, 8, 16, 32, 64, np.inf]
    duration_labels = ["<8s", "8–16s", "16–32s", "32–64s", ">=64s"]
    pred["duration_group"] = pd.cut(pred.total_duration_s, duration_edges,
                                     labels=duration_labels, right=False)
    segment_edges = [0, 6, 11, 21, 41, np.inf]
    segment_labels = ["1–5", "6–10", "11–20", "21–40", ">=41"]
    pred["segment_count_group"] = pd.cut(pred.n_segments, segment_edges,
                                          labels=segment_labels, right=False)
    by_duration = {label: group_metric(pred[pred.duration_group == label]) for label in duration_labels}
    by_segments = {label: group_metric(pred[pred.segment_count_group == label]) for label in segment_labels}
    quality_masks = {
        "mfcc_success": pred.mfcc_success == 1,
        "mfcc_failure": pred.mfcc_success == 0,
        "has_failed_segment": pred.failed_segments > 0,
        "all_segments_succeeded": pred.failed_segments == 0,
        "has_short_segment_lt_25ms": pred.short_segments > 0,
        "no_short_segment_lt_25ms": pred.short_segments == 0,
        "has_high_silence_segment": pred.high_silence_segments > 0,
        "no_high_silence_segment": pred.high_silence_segments == 0,
    }
    by_quality = {name: group_metric(pred[mask]) for name, mask in quality_masks.items()}
    overall = group_metric(pred)
    by_gender = {gender: group_metric(pred[pred.gender == gender]) for gender in ("M", "F")}
    cm = confusion_matrix(pred.gender, pred.prediction, labels=["M", "F"])

    f0 = pd.read_csv(f0_predictions_path, usecols=["call_id", "gender", "prediction"])
    f0 = f0.rename(columns={"prediction": "f0_prediction"})
    paired = pred.merge(f0, on=["call_id", "gender"], validate="one_to_one")
    if len(paired) != 5597:
        raise ValueError("F0 and MFCC Validation predictions do not match one-to-one")
    paired["mfcc_correct"] = paired.gender == paired.prediction
    paired["f0_correct"] = paired.gender == paired.f0_prediction
    comparison = {
        "majority_accuracy": 0.5318920850455601,
        "f0_overall_accuracy": 0.9120957655887082,
        "f0_male_accuracy": 0.899618320610687,
        "f0_female_accuracy": 0.9230769230769231,
        "mfcc_minus_f0_accuracy": overall["accuracy"] - 0.9120957655887082,
        "mfcc_minus_f0_male_accuracy": by_gender["M"]["accuracy"] - 0.899618320610687,
        "mfcc_minus_f0_female_accuracy": by_gender["F"]["accuracy"] - 0.9230769230769231,
        "both_correct": int((paired.mfcc_correct & paired.f0_correct).sum()),
        "mfcc_only_correct": int((paired.mfcc_correct & ~paired.f0_correct).sum()),
        "f0_only_correct": int((~paired.mfcc_correct & paired.f0_correct).sum()),
        "both_wrong": int((~paired.mfcc_correct & ~paired.f0_correct).sum()),
        "short_lt_8s_f0_accuracy": float(
            (paired.loc[paired.total_duration_s < 8, "gender"] ==
             paired.loc[paired.total_duration_s < 8, "f0_prediction"]).mean()
        ),
        "short_lt_8s_mfcc_accuracy": by_duration["<8s"]["accuracy"],
    }
    total_segments = int(features.n_segments.sum())
    successful_segments = int(features.successful_segments.sum())
    quality = {
        "calls": len(features),
        "mfcc_success_calls": int(features.mfcc_success.sum()),
        "mfcc_failure_calls": int((features.mfcc_success == 0).sum()),
        "segments": total_segments,
        "successful_segments": successful_segments,
        "failed_segments": int(features.failed_segments.sum()),
        "short_segments_lt_25ms": int(features.short_segments.sum()),
        "short_segment_rate": float(features.short_segments.sum() / total_segments),
        "high_silence_segments": int(features.high_silence_segments.sum()),
        "high_silence_segment_rate": float(features.high_silence_segments.sum() / total_segments),
        "raw_nan_model_values_train": int(x_train.isna().sum().sum()),
        "raw_nan_model_values_validation": int(x_valid.isna().sum().sum()),
        "raw_inf_model_values_train": 0,
        "raw_inf_model_values_validation": 0,
    }
    metrics = {
        "dataset": {"branch": "mission1/eunkyo", "split_sha256": split_hash,
                    "seed": SEED, "train_calls": len(train),
                    "internal_validation_calls": len(valid)},
        "mfcc": {"library": "librosa mel filter bank + scipy DCT-II",
                 "sample_rate": SAMPLE_RATE, "n_mfcc": N_MFCC, "n_mels": N_MELS,
                 "n_fft": N_FFT, "hop_length": HOP_LENGTH, "win_length": WIN_LENGTH,
                 "frequency_range_hz": [FMIN_HZ, FMAX_HZ], "window": WINDOW,
                 "pre_emphasis": PRE_EMPHASIS, "center": False,
                 "mel_scale": "Slaney", "mel_filter_normalization": "slaney area",
                 "dct": "type-II, ortho", "waveform_normalization": None,
                 "cepstral_mean_variance_normalization": None},
        "features": {"dimension": len(MODEL_FEATURES), "names": list(MODEL_FEATURES),
                     "segment_summary": "per-coefficient temporal mean and std",
                     "call_aggregation": "segment-feature mean and std, plus n_segments and total_duration_s"},
        "classifier": {"type": "SVC", "kernel": "rbf", "C": 1.0,
                       "gamma": "scale", "probability": False, "class_weight": None,
                       "cache_size_mb": SVC_CACHE_MB, "random_state": None,
                       "determinism": "SVC fit/predict is deterministic with probability disabled"},
        "preprocessing": {"imputer": "Train-only median", "scaler": "StandardScaler fit on Train only",
                          "scaler_fit_samples": int(pipeline.named_steps["scaler"].n_samples_seen_)},
        "resource_assessment": {"dense_kernel_matrix_gib": len(train) ** 2 * 8 / 1024 ** 3,
                                "benchmark_train_calls": benchmark_n,
                                "benchmark_fit_seconds": benchmark_seconds,
                                "quadratic_projection_seconds": projected_seconds,
                                "full_fit_seconds": fit_seconds,
                                "validation_predict_seconds": predict_seconds,
                                "decision": "Full RBF SVC retained: fixed 54-D matrix is small and measured resource benchmark was acceptable; libsvm cache capped at 512 MB."},
        "quality": quality,
        "overall": overall,
        "gender_accuracy": by_gender,
        "confusion_matrix": {"labels": ["M", "F"], "matrix": cm.tolist()},
        "prediction_counts": {key: int(value) for key, value in Counter(pred.prediction).items()},
        "duration_groups": by_duration,
        "segment_count_groups": by_segments,
        "mfcc_quality_groups": by_quality,
        "comparison": comparison,
    }
    if metrics["preprocessing"]["scaler_fit_samples"] != 22388:
        raise AssertionError("Scaler fit did not use exactly 22,388 Train calls")
    return pred, metrics


def write_report(path: Path, metrics: dict, versions: dict, command: str):
    m, q, c = metrics, metrics["quality"], metrics["comparison"]
    cm = metrics["confusion_matrix"]["matrix"]
    short_difference = c["short_lt_8s_mfcc_accuracy"] - c["short_lt_8s_f0_accuracy"]
    relation = "높다" if c["mfcc_minus_f0_accuracy"] > 0 else "낮다" if c["mfcc_minus_f0_accuracy"] < 0 else "같다"
    practical = ("1 percentage point 미만으로 비슷한 수준" if abs(c["mfcc_minus_f0_accuracy"]) < .01
                 else "이 fixed split의 baseline 비교에서는 의미 있는 차이")
    gender_larger = "Male" if abs(c["mfcc_minus_f0_male_accuracy"]) > abs(c["mfcc_minus_f0_female_accuracy"]) else "Female"
    lines = [
        "# STEP 4-2 — MFCC + SVM baseline", "",
        "## 고정 split", "",
        f"브랜치 `mission1/eunkyo`, seed {SEED}. 기존 split SHA256 `{m['dataset']['split_sha256']}`를 그대로 사용했다. "
        "F0 `call_features.csv`의 전체 call_id/gender/partition과도 일대일 동일성을 검사했다. 새 split이나 공식 Validation은 사용하지 않았다.", "",
        "| Partition | Male | Female | Total |", "|---|---:|---:|---:|",
        "| Train | 10,480 | 11,908 | 22,388 |",
        "| Internal Validation | 2,620 | 2,977 | 5,597 |", "",
        "`speaker=1` 신고자 segment만 사용했다. EDA에서 약 96% 통화에 신고자/상담원 annotation overlap이 있었으므로, 신고자 segment를 추출한다고 해서 상담원 음성이 완전히 제거되는 것은 아니다.", "",
        "## MFCC와 aggregation", "",
        f"8kHz mono 입력에서 n_mfcc={N_MFCC}, n_mels={N_MELS}, n_fft={N_FFT}(32ms), win_length={WIN_LENGTH}(25ms), "
        f"hop_length={HOP_LENGTH}(10ms), {FMIN_HZ:.0f}–{FMAX_HZ:.0f}Hz, Hamming window, pre-emphasis={PRE_EMPHASIS}를 사용했다. "
        "Slaney mel filter area normalization과 orthonormal DCT-II를 사용했고 waveform normalization과 CMVN은 적용하지 않았다. "
        "각 coefficient sequence의 mean/std를 segment당 26차원으로 만든 뒤, 통화 안에서 각 값의 mean/std를 취하고 segment 수와 총 길이를 더해 최종 54차원으로 고정했다.", "",
        "구현은 [librosa MFCC 문서](https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html)의 mel-spectrum→DCT 구성을 명시적으로 사용한다. "
        "프레임/FFT 설정은 8kHz 전화 음성의 시간 해상도와 주파수 해상도를 함께 유지하는 작은 baseline 설정이다.", "",
        "## 특징 품질", "",
        "| 항목 | 개수 | 비율 |", "|---|---:|---:|",
        f"| MFCC 성공 call | {q['mfcc_success_calls']:,} | {q['mfcc_success_calls']/q['calls']:.6%} |",
        f"| MFCC 실패 call | {q['mfcc_failure_calls']:,} | {q['mfcc_failure_calls']/q['calls']:.6%} |",
        f"| 성공 segment | {q['successful_segments']:,} | {q['successful_segments']/q['segments']:.6%} |",
        f"| 실패 segment | {q['failed_segments']:,} | {q['failed_segments']/q['segments']:.6%} |",
        f"| 25ms 미만 segment | {q['short_segments_lt_25ms']:,} | {q['short_segment_rate']:.6%} |",
        f"| silence ratio >=50% segment | {q['high_silence_segments']:,} | {q['high_silence_segment_rate']:.6%} |", "",
        f"모델 특징의 추출 후 NaN/inf는 Train {q['raw_nan_model_values_train']}/{q['raw_inf_model_values_train']}, Validation {q['raw_nan_model_values_validation']}/{q['raw_inf_model_values_validation']}개다. "
        "25ms 미만 구간은 zero-padding한 한 frame으로 유지하고 품질 플래그를 남겼다. 실패 segment를 조용히 제거하지 않으며, 일부 실패면 성공 segment로 집계하고 전부 실패한 call은 Train 중앙값 대체 후 cohort에 유지하도록 구현했다. "
        "무음 대용치는 frame RMS <0.001이며 VAD가 아니므로 저에너지 발화와 잡음을 구분하지 못한다. 무음이 많은 segment의 MFCC는 배경/채널 특성을 더 반영할 수 있다.", "",
        "## RBF SVC 자원 판단", "",
        f"scikit-learn은 SVC fit 시간이 표본 수에 대해 적어도 이차적으로 증가할 수 있다고 경고한다. 22,388² dense kernel은 약 {m['resource_assessment']['dense_kernel_matrix_gib']:.2f}GiB지만 libsvm이 이를 전부 상주시킬 필요는 없고 cache를 {SVC_CACHE_MB}MB로 제한했다. "
        f"Train 앞 {m['resource_assessment']['benchmark_train_calls']:,}개 자원 측정은 {m['resource_assessment']['benchmark_fit_seconds']:.2f}초, 보수적 이차 투영은 {m['resource_assessment']['quadratic_projection_seconds']:.2f}초였다. "
        f"따라서 대안으로 바꾸지 않고 사전 지정 RBF SVC 한 개를 실행했으며 실제 full fit은 {m['resource_assessment']['full_fit_seconds']:.2f}초였다. "
        "설정은 C=1.0, gamma=scale, probability=False, class_weight=None이며 tuning하지 않았다. "
        "probability calibration을 사용하지 않아 SVC fit/predict에는 stochastic random_state가 필요하지 않으며, seed 42는 frozen split과 재현성 표기에 사용했다. "
        "[SVC 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)", "",
        "NaN/inf 대체 중앙값과 StandardScaler는 Train 22,388개에만 fit했고 Validation에는 transform만 적용했다.", "",
        "## 고정 Internal Validation 결과", "",
        "| Model | Validation Unit | Accuracy |", "|---|---|---:|",
        f"| Majority Female | Call | {c['majority_accuracy']:.6%} |",
        f"| F0 + Acoustic + Logistic Regression | Call | {c['f0_overall_accuracy']:.6%} |",
        f"| MFCC + RBF SVM | Call | {m['overall']['accuracy']:.6%} |", "",
        f"MFCC overall은 F0보다 {c['mfcc_minus_f0_accuracy']*100:+.6f}pp, Male은 {c['mfcc_minus_f0_male_accuracy']*100:+.6f}pp, Female은 {c['mfcc_minus_f0_female_accuracy']*100:+.6f}pp다. "
        f"Validation 5,597개 중 두 모델 모두 정답 {c['both_correct']:,}, MFCC만 정답 {c['mfcc_only_correct']:,}, F0만 정답 {c['f0_only_correct']:,}, 둘 다 오답 {c['both_wrong']:,}이다.", "",
        f"Male accuracy {m['gender_accuracy']['M']['accuracy']:.6%}, Female accuracy {m['gender_accuracy']['F']['accuracy']:.6%}. "
        f"예측 수 M {m['prediction_counts'].get('M',0):,}, F {m['prediction_counts'].get('F',0):,}.", "",
        "| | Pred M | Pred F |", "|---|---:|---:|",
        f"| True M | {cm[0][0]:,} | {cm[0][1]:,} |",
        f"| True F | {cm[1][0]:,} | {cm[1][1]:,} |", "",
        "### 총 신고자 발화 시간별", "", "| 구간 | n | Accuracy |", "|---|---:|---:|",
    ]
    for name, value in m["duration_groups"].items():
        accuracy = "N/A" if value["accuracy"] is None else f"{value['accuracy']:.3%}"
        lines.append(f"| {name} | {value['n']:,} | {accuracy} |")
    lines += ["", "### 신고자 segment 수별", "", "| 구간 | n | Accuracy |", "|---|---:|---:|"]
    for name, value in m["segment_count_groups"].items():
        accuracy = "N/A" if value["accuracy"] is None else f"{value['accuracy']:.3%}"
        lines.append(f"| {name} | {value['n']:,} | {accuracy} |")
    lines += ["", "### MFCC 품질별", "", "| 그룹 | n | Accuracy |", "|---|---:|---:|"]
    for name, value in m["mfcc_quality_groups"].items():
        accuracy = "N/A" if value["accuracy"] is None else f"{value['accuracy']:.3%}"
        lines.append(f"| {name} | {value['n']:,} | {accuracy} |")
    lines += [
        "", "## 해석", "",
        f"1. MFCC baseline은 F0 baseline보다 {relation}: 차이는 {c['mfcc_minus_f0_accuracy']*100:+.6f}pp다.",
        f"2. 전체 차이는 {practical}이다. 두 모델이 서로 다르게 맞힌 통화가 {c['mfcc_only_correct'] + c['f0_only_correct']:,}개라 오류가 완전히 같지는 않지만 hybrid/ensemble 실험은 수행하지 않았다.",
        f"3. F0 대비 변화의 절댓값은 {gender_larger} 쪽이 더 크다. 성별별 차이는 Male {c['mfcc_minus_f0_male_accuracy']*100:+.6f}pp, Female {c['mfcc_minus_f0_female_accuracy']*100:+.6f}pp다.",
        f"4. <8초 통화는 n={m['duration_groups']['<8s']['n']}로 작다. MFCC {c['short_lt_8s_mfcc_accuracy']:.3%}, F0 {c['short_lt_8s_f0_accuracy']:.3%}, 차이 {short_difference*100:+.3f}pp이므로 이 표본만으로 더 안정적이라고 일반화할 수 없다.",
        "5. MFCC가 낮거나 특정 그룹에서 약하면 segment 통계화로 시간 순서를 잃는 점, 긴 무음/잡음과 채널 특성, operator overlap, 울음·비명·긴장 발성, 짧은 구간 zero-padding이 가능한 원인이다.",
        "6. Frozen SSL은 MFCC/F0가 놓치는 시간 구조와 음성 표현을 시험할 가치가 있다. 다만 같은 fixed split과 caller-only 구간을 유지하고, 현재 known risks를 해결된 것으로 간주하면 안 된다.", "",
        "동일 사람 반복 등 speaker-level leakage는 현재 메타데이터로 확인 불가하다. 이 결과는 현재 Internal Validation에 한정되며 공식 Validation 성능이나 독립 화자 일반화를 보장하지 않는다.", "",
        "## 재현성", "",
        f"실행 명령: `{command}`", "",
        f"전체 실행 시간은 {m['runtime']['total_seconds']:.2f}초이며 worker {m['runtime']['workers']}개를 사용했다.", "",
        "라이브러리: " + ", ".join(f"{key}={value}" for key, value in versions.items()), "",
        "기본 `results/`가 이미 존재하면 실행을 중단한다. 재실행은 새 `--out-dir`을 지정해야 하며, `--limit`은 feature smoke test만 수행하고 모델을 학습하지 않는다.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    mission = here.parents[1]
    parser.add_argument("--split", type=Path, default=mission / "validation/split_assignments.csv")
    parser.add_argument("--manifest", type=Path, default=mission / "validation/manifests/calls.csv")
    parser.add_argument("--segments", type=Path, default=mission / "eda/eda_outputs/segments.csv")
    parser.add_argument("--f0-metrics", type=Path, default=mission / "baseline/f0_lr/results/metrics.json")
    parser.add_argument("--f0-features", type=Path, default=mission / "baseline/f0_lr/results/call_features.csv")
    parser.add_argument("--f0-predictions", type=Path, default=mission / "baseline/f0_lr/results/val_predictions.csv")
    parser.add_argument("--out-dir", type=Path, default=here / "results")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--limit", type=int, help="Feature smoke test only; no classifier fit/evaluation")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Output directory already exists; use a new --out-dir (no overwrite mode)")
    if args.workers < 1:
        parser.error("--workers must be positive")
    jobs, split_hash = load_jobs(args.split.resolve(), args.manifest.resolve(), args.segments.resolve(),
                                 args.f0_metrics.resolve(), args.f0_features.resolve())
    if args.limit:
        jobs = jobs[:args.limit]
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(aggregate_call, *zip(*jobs), chunksize=4), 1):
            rows.append(row)
            if index % 500 == 0 or index == len(jobs):
                print(f"MFCC extraction {index:,} / {len(jobs):,}", flush=True)
    frame = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True)
    frame.to_csv(args.out_dir / "call_features.csv", index=False, lineterminator="\n")
    if args.limit:
        print("Development subset complete; no model evaluation performed")
        return
    predictions, metrics = fit_and_evaluate(frame, split_hash, args.f0_predictions.resolve())
    predictions.to_csv(args.out_dir / "val_predictions.csv", index=False, lineterminator="\n")
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "librosa": librosa.__version__,
                "scipy": importlib.metadata.version("scipy"), "soundfile": sf.__version__,
                "scikit-learn": importlib.metadata.version("scikit-learn")}
    metrics["runtime"] = {"total_seconds": time.perf_counter() - started,
                          "workers": args.workers, "versions": versions}
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    command = f"python mission-1/baseline/mfcc_svm/run_mfcc_baseline.py --workers {args.workers}"
    write_report(args.out_dir / "REPORT.md", metrics, versions, command)
    print(json.dumps({"accuracy": metrics["overall"]["accuracy"],
                      "male_accuracy": metrics["gender_accuracy"]["M"]["accuracy"],
                      "female_accuracy": metrics["gender_accuracy"]["F"]["accuracy"],
                      "comparison": metrics["comparison"], "quality": metrics["quality"]}, indent=2))


if __name__ == "__main__":
    main()

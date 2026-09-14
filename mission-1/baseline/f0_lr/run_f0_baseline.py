"""F0 + compact acoustic LogisticRegression baseline on the frozen call split."""
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
import sys
import time

import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RAW_F0_MIN_HZ = 50.0
RAW_F0_MAX_HZ = 800.0
SPEECH_F0_MIN_HZ = 60.0
SPEECH_F0_MAX_HZ = 400.0
PITCH_TIME_STEP_S = 0.01
VOICING_THRESHOLD = 0.45
MIN_FILTERED_FRAMES = 3
MIN_VOICED_RATIO = 0.10
LOW_VOICED_RATIO = 0.10
TONE_CONCENTRATION_THRESHOLD = 0.90
AGGREGATED_SEGMENT_FEATURES = (
    "f0_mean_hz", "f0_median_hz", "f0_std_hz", "voiced_ratio",
    "rms", "duration_s", "zcr", "spectral_centroid_hz",
)
MODEL_FEATURES = tuple(
    f"{name}_{stat}" for name in AGGREGATED_SEGMENT_FEATURES for stat in ("mean", "median")
) + ("n_segments", "total_duration_s", "reliable_f0_segment_ratio")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else math.nan


def spectral_features(samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
    if len(samples) < 2:
        return math.nan, math.nan
    x = np.asarray(samples, dtype=np.float64)
    zcr = float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1])))
    windowed = x * np.hanning(len(x))
    power = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    band = (freqs >= 50.0) & (freqs <= min(1000.0, sample_rate / 2.0))
    band_power = power[band]
    total = float(band_power.sum())
    if total <= np.finfo(float).eps:
        return zcr, math.nan
    centroid = float(np.sum(freqs[band] * band_power) / total)
    concentration = float(np.max(band_power) / total)
    return zcr, centroid, concentration


def extract_segment(samples: np.ndarray, sample_rate: int) -> dict[str, float | int | bool]:
    x = np.asarray(samples, dtype=np.float64)
    duration = len(x) / sample_rate
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else math.nan
    spec = spectral_features(x, sample_rate)
    if len(spec) == 2:  # only possible for a sub-two-sample segment
        zcr, centroid = spec
        concentration = math.nan
    else:
        zcr, centroid, concentration = spec
    raw = np.empty(0, dtype=float)
    if len(x) >= round(3.0 / RAW_F0_MIN_HZ * sample_rate) and rms > 0:
        try:
            pitch = parselmouth.Sound(x, sampling_frequency=sample_rate).to_pitch_ac(
                time_step=PITCH_TIME_STEP_S,
                pitch_floor=RAW_F0_MIN_HZ,
                max_number_of_candidates=15,
                very_accurate=False,
                silence_threshold=0.03,
                voicing_threshold=VOICING_THRESHOLD,
                octave_cost=0.01,
                octave_jump_cost=0.35,
                voiced_unvoiced_cost=0.14,
                pitch_ceiling=RAW_F0_MAX_HZ,
            )
            frequencies = np.asarray(pitch.selected_array["frequency"], dtype=float)
            raw = frequencies[np.isfinite(frequencies) & (frequencies > 0)]
            n_frames = len(frequencies)
        except Exception:
            n_frames = 0
    else:
        n_frames = 0
    filtered = raw[(raw >= SPEECH_F0_MIN_HZ) & (raw <= SPEECH_F0_MAX_HZ)]
    voiced_ratio = len(filtered) / n_frames if n_frames else 0.0
    tone_dominated = bool(np.isfinite(concentration) and concentration >= TONE_CONCENTRATION_THRESHOLD)
    reliable = bool(
        len(filtered) >= MIN_FILTERED_FRAMES
        and voiced_ratio >= MIN_VOICED_RATIO
        and not tone_dominated
    )
    return {
        "duration_s": duration,
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid_hz": centroid,
        "f0_mean_hz": float(np.mean(filtered)) if reliable else math.nan,
        "f0_median_hz": float(np.median(filtered)) if reliable else math.nan,
        "f0_std_hz": float(np.std(filtered)) if reliable else math.nan,
        "voiced_ratio": voiced_ratio,
        "raw_f0_success": bool(len(raw)),
        "reliable_f0": reliable,
        "raw_f0_frames": int(len(raw)),
        "raw_low_frames": int(np.sum(raw < SPEECH_F0_MIN_HZ)),
        "raw_high_frames": int(np.sum(raw > SPEECH_F0_MAX_HZ)),
        "pitch_frames": int(n_frames),
        "low_voiced": bool(voiced_ratio < LOW_VOICED_RATIO),
        "tone_dominated": tone_dominated,
        "tone_concentration": concentration,
    }


def aggregate_call(call_id: str, gender: str, partition: str, wav_path: str,
                   intervals: list[tuple[float, float]]) -> dict[str, float | int | str]:
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim != 1 or sample_rate != 8000:
        raise ValueError(f"Unexpected WAV format for {call_id}: {sample_rate} Hz, shape={audio.shape}")
    segment_rows = []
    for start_s, end_s in intervals:
        start = round(start_s * sample_rate)
        end = round(end_s * sample_rate)
        if start < 0 or end <= start or end > len(audio):
            raise ValueError(f"Invalid segment bounds for {call_id}: {start_s}, {end_s}")
        segment_rows.append(extract_segment(audio[start:end], sample_rate))
    result: dict[str, float | int | str] = {
        "call_id": call_id, "gender": gender, "partition": partition,
        "n_segments": len(segment_rows),
        "total_duration_s": float(sum(r["duration_s"] for r in segment_rows)),
    }
    for name in AGGREGATED_SEGMENT_FEATURES:
        values = np.asarray([r[name] for r in segment_rows], dtype=float)
        values = values[np.isfinite(values)]
        result[f"{name}_mean"] = float(np.mean(values)) if len(values) else math.nan
        result[f"{name}_median"] = float(np.median(values)) if len(values) else math.nan
    count = len(segment_rows)
    result.update(
        raw_f0_success_segments=sum(bool(r["raw_f0_success"]) for r in segment_rows),
        reliable_f0_segments=sum(bool(r["reliable_f0"]) for r in segment_rows),
        reliable_f0_segment_ratio=sum(bool(r["reliable_f0"]) for r in segment_rows) / count,
        low_voiced_segments=sum(bool(r["low_voiced"]) for r in segment_rows),
        tone_dominated_segments=sum(bool(r["tone_dominated"]) for r in segment_rows),
        pitch_frames=sum(int(r["pitch_frames"]) for r in segment_rows),
        raw_f0_frames=sum(int(r["raw_f0_frames"]) for r in segment_rows),
        raw_low_frames=sum(int(r["raw_low_frames"]) for r in segment_rows),
        raw_high_frames=sum(int(r["raw_high_frames"]) for r in segment_rows),
    )
    return result


def load_jobs(split_path: Path, manifest_path: Path, segments_path: Path):
    split_rows = list(read_csv(split_path))
    if len(split_rows) != 27985 or len({r["call_id"] for r in split_rows}) != 27985:
        raise ValueError("Frozen split must contain exactly 27,985 unique calls")
    split = {r["call_id"]: (r["gender"], r["partition"]) for r in split_rows}
    manifests = {r["call_id"]: r for r in read_csv(manifest_path)}
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in read_csv(segments_path):
        key = row["file_ref"]
        if row["split"] == "Training" and key in split and row["speaker"] == "1":
            intervals[key].append((float(row["start_s"]), float(row["end_s"])))
    if set(split) != set(manifests) or set(split) != set(intervals):
        raise ValueError("Split, call manifest, and reporter segment index do not cover the same calls")
    jobs = []
    for key in sorted(split):
        gender, partition = split[key]
        manifest = manifests[key]
        if manifest["gender"] != gender or manifest["partition"] != partition:
            raise ValueError(f"Frozen metadata mismatch: {key}")
        jobs.append((key, gender, partition, manifest["wav_path"], intervals[key]))
    return jobs


def metric_group(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "accuracy": None, "male_n": 0, "female_n": 0}
    return {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(frame["gender"], frame["prediction"])),
        "male_n": int((frame["gender"] == "M").sum()),
        "female_n": int((frame["gender"] == "F").sum()),
    }


def evaluate(features: pd.DataFrame, split_hash: str) -> tuple[pd.DataFrame, dict, Pipeline]:
    train = features[features.partition == "train"].copy()
    valid = features[features.partition == "internal_validation"].copy()
    if (len(train), len(valid)) != (22388, 5597):
        raise ValueError(f"Unexpected fixed split size: {len(train)}, {len(valid)}")
    x_train = train.loc[:, MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    x_valid = valid.loc[:, MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    if x_train.isna().all().any():
        raise ValueError("A model feature is entirely missing in Train")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=SEED, max_iter=1000, class_weight=None, solver="lbfgs"
        )),
    ])
    pipeline.fit(x_train, train.gender)
    predictions = pipeline.predict(x_valid)
    probability_f = pipeline.predict_proba(x_valid)[:, list(pipeline.classes_).index("F")]
    pred = valid[["call_id", "gender", "total_duration_s", "n_segments",
                  "reliable_f0_segments", "reliable_f0_segment_ratio",
                  "low_voiced_segments", "tone_dominated_segments"]].copy()
    pred["prediction"] = predictions
    pred["probability_female"] = probability_f
    overall = metric_group(pred)
    by_gender = {g: metric_group(pred[pred.gender == g]) for g in ("M", "F")}
    cm = confusion_matrix(pred.gender, pred.prediction, labels=["M", "F"])
    duration_bins = [0, 8, 16, 32, 64, np.inf]
    duration_labels = ["<8s", "8–16s", "16–32s", "32–64s", ">=64s"]
    pred["duration_group"] = pd.cut(pred.total_duration_s, duration_bins,
                                     labels=duration_labels, right=False)
    by_duration = {label: metric_group(pred[pred.duration_group == label]) for label in duration_labels}
    f0_groups = {
        "no_reliable_f0": pred.reliable_f0_segments == 0,
        "some_reliable_f0": pred.reliable_f0_segments > 0,
        "low_reliable_ratio_lt_0.5": pred.reliable_f0_segment_ratio < 0.5,
        "reliable_ratio_ge_0.5": pred.reliable_f0_segment_ratio >= 0.5,
        "has_low_voiced_segment": pred.low_voiced_segments > 0,
        "no_low_voiced_segment": pred.low_voiced_segments == 0,
        "tone_flag_present": pred.tone_dominated_segments > 0,
        "no_tone_flag": pred.tone_dominated_segments == 0,
    }
    by_f0_quality = {name: metric_group(pred[mask]) for name, mask in f0_groups.items()}
    total_segments = int(features.n_segments.sum())
    raw_success = int(features.raw_f0_success_segments.sum())
    reliable = int(features.reliable_f0_segments.sum())
    pitch_frames = int(features.pitch_frames.sum())
    raw_frames = int(features.raw_f0_frames.sum())
    quality = {
        "segments": total_segments,
        "raw_f0_success_segments": raw_success,
        "raw_f0_failure_segments": total_segments - raw_success,
        "raw_f0_success_rate": raw_success / total_segments,
        "raw_f0_failure_rate": (total_segments - raw_success) / total_segments,
        "baseline_reliable_f0_segments": reliable,
        "baseline_reliable_f0_rate": reliable / total_segments,
        "baseline_unreliable_f0_segments": total_segments - reliable,
        "baseline_unreliable_f0_rate": (total_segments - reliable) / total_segments,
        "low_voiced_segments": int(features.low_voiced_segments.sum()),
        "low_voiced_segment_rate": float(features.low_voiced_segments.sum() / total_segments),
        "tone_dominated_segments": int(features.tone_dominated_segments.sum()),
        "tone_dominated_segment_rate": float(features.tone_dominated_segments.sum() / total_segments),
        "raw_voiced_frames": raw_frames,
        "raw_low_frames": int(features.raw_low_frames.sum()),
        "raw_high_frames": int(features.raw_high_frames.sum()),
        "raw_low_frame_rate": float(features.raw_low_frames.sum() / raw_frames) if raw_frames else None,
        "raw_high_frame_rate": float(features.raw_high_frames.sum() / raw_frames) if raw_frames else None,
        "all_pitch_frames": pitch_frames,
    }
    majority = 2977 / 5597
    metrics = {
        "dataset": {"train_calls": len(train), "internal_validation_calls": len(valid),
                    "split_sha256": split_hash, "seed": SEED},
        "model": {"name": "F0 + acoustic LogisticRegression", "features": list(MODEL_FEATURES),
                  "aggregation": "per-segment feature mean and median, plus n_segments, total duration, reliable-F0 segment ratio",
                  "imputer": "Train-only median", "scaler": "StandardScaler fit on Train only",
                  "classifier": {"solver": "lbfgs", "max_iter": 1000, "random_state": SEED,
                                 "class_weight": None},
                  "n_iter": int(pipeline.named_steps["classifier"].n_iter_[0]),
                  "scaler_fit_samples": int(pipeline.named_steps["scaler"].n_samples_seen_)},
        "overall": overall,
        "gender_accuracy": by_gender,
        "confusion_matrix": {"labels": ["M", "F"], "matrix": cm.tolist()},
        "prediction_counts": {k: int(v) for k, v in Counter(pred.prediction).items()},
        "duration_groups": by_duration,
        "f0_quality_groups": by_f0_quality,
        "f0_quality": quality,
        "majority_baseline_accuracy": majority,
        "absolute_improvement": overall["accuracy"] - majority,
    }
    if metrics["model"]["scaler_fit_samples"] != 22388:
        raise AssertionError("Scaler was not fit on exactly the fixed Train calls")
    return pred, metrics, pipeline


def write_report(path: Path, metrics: dict, versions: dict, command: str):
    m = metrics
    q = m["f0_quality"]
    cm = m["confusion_matrix"]["matrix"]
    lines = [
        "# STEP 4-1 — F0 + Acoustic Logistic Regression", "",
        "## 실행 범위와 고정 split", "",
        f"기존 `split_assignments.csv`(SHA256 `{m['dataset']['split_sha256']}`)를 그대로 사용했다. "
        f"Train {m['dataset']['train_calls']:,} calls, Internal Validation {m['dataset']['internal_validation_calls']:,} calls이며 seed는 {SEED}이다. "
        "새 split, threshold tuning, MFCC/SVM/pretrained 모델 실험은 수행하지 않았다.", "",
        "| Partition | Male | Female | Total |", "|---|---:|---:|---:|",
        "| Train | 10,480 | 11,908 | 22,388 |",
        "| Internal Validation | 2,620 | 2,977 | 5,597 |", "",
        "신고자 역할인 `speaker=1` segment만 잘랐다. 다만 EDA에서 약 96%의 통화에 신고자/상담원 annotation overlap이 있었으므로, "
        "신고자 segment를 자른다고 해서 상담원 음성이 완전히 제거된다고 볼 수 없다.", "",
        "## 특징과 집계", "",
        "각 segment에서 filtered F0 mean/median/std, voiced ratio, RMS, duration, zero-crossing rate, spectral centroid를 계산했다. "
        "통화별로 각 값의 mean과 median을 취하고 segment 수, segment 길이 합, 신뢰 F0 segment 비율을 더했다. 이것이 유일한 aggregation이다.", "",
        f"Pitch extractor는 Praat raw autocorrelation(Parselmouth)이며 time step {PITCH_TIME_STEP_S:.2f}s, 원시 탐색 {RAW_F0_MIN_HZ:.0f}–{RAW_F0_MAX_HZ:.0f}Hz, "
        f"voicing threshold {VOICING_THRESHOLD:.2f}이다. 원시 결과는 품질 감사에만 쓰고, 모델 F0는 {SPEECH_F0_MIN_HZ:.0f}–{SPEECH_F0_MAX_HZ:.0f}Hz 프레임만 남겼다. "
        f"최소 {MIN_FILTERED_FRAMES} 프레임, voiced ratio {MIN_VOICED_RATIO:.2f} 이상이며 50–1000Hz 스펙트럼 단일-bin 전력 집중도가 {TONE_CONCENTRATION_THRESHOLD:.2f} 미만인 segment만 F0 신뢰 구간으로 처리했다. "
        "이 규칙은 validation 결과를 보기 전에 고정했으며 사람 음성을 완벽히 판별하는 VAD나 tone detector가 아니다.", "",
        "660Hz tone이나 비명·울음·잡음은 pitch 후보 또는 octave error를 만들 수 있다. 따라서 filtered F0도 생리적 사람 pitch의 확정값이 아니며, RMS/ZCR/centroid에도 상대 화자와 비음성 신호가 영향을 줄 수 있다.", "",
        "Praat 설정 근거: [raw autocorrelation 설정](https://fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html), "
        "[pitch analysis method 선택](https://uvafon.hum.uva.nl/praat/manual/how_to_choose_a_pitch_analysis_method.html), "
        "[Parselmouth `to_pitch_ac` API](https://parselmouth.readthedocs.io/en/stable/api_reference.html). "
        "Praat의 일반 기본값을 그대로 사람 성별 경계로 사용하지 않고, 전화 음성·고음 감정 발성·660Hz tone 감사를 위해 원시 ceiling을 800Hz로 넓힌 뒤 모델 입력 범위를 별도로 필터링했다.", "",
        "## F0 품질", "",
        "| 항목 | segment 수 | 비율 |", "|---|---:|---:|",
        f"| 원시 F0 추출 성공 | {q['raw_f0_success_segments']:,} | {q['raw_f0_success_rate']:.3%} |",
        f"| 원시 F0 추출 실패 | {q['raw_f0_failure_segments']:,} | {q['raw_f0_failure_rate']:.3%} |",
        f"| baseline 신뢰 F0 | {q['baseline_reliable_f0_segments']:,} | {q['baseline_reliable_f0_rate']:.3%} |",
        f"| baseline F0 제외/불신뢰 | {q['baseline_unreliable_f0_segments']:,} | {q['baseline_unreliable_f0_rate']:.3%} |",
        f"| 낮은 voiced ratio | {q['low_voiced_segments']:,} | {q['low_voiced_segment_rate']:.3%} |",
        f"| tone concentration flag | {q['tone_dominated_segments']:,} | {q['tone_dominated_segment_rate']:.3%} |", "",
        f"원시 voiced frame 중 <{SPEECH_F0_MIN_HZ:.0f}Hz 비율은 {q['raw_low_frame_rate']:.3%}, >{SPEECH_F0_MAX_HZ:.0f}Hz 비율은 {q['raw_high_frame_rate']:.3%}이다.", "",
        "사전 고정한 단일-bin concentration 0.90 기준에서는 tone flag가 검출되지 않았다. 이는 tone이 없다는 증거가 아니라, 긴 segment의 전체 FFT에 적용한 보수적 대용치가 기존 EDA의 tone 사례를 전수 식별하지 못했다는 뜻이다. "
        "고주파 F0 frame 비율과 60–400Hz 필터가 이번 baseline의 주된 오염 방어이며 별도 tone detector로 간주하지 않는다.", "",
        "## 모델과 전처리", "",
        "NaN/inf는 Train에서 계산한 feature별 중앙값으로 대체했다. `StandardScaler`도 Train 22,388개에만 fit했고 Validation에는 transform만 적용했다. "
        f"LogisticRegression은 solver=lbfgs, max_iter=1000, random_state={SEED}, class_weight=None이며 tuning이나 decision threshold 변경은 없다.", "",
        "## 고정 Internal Validation 결과", "",
        "| Model | Validation Unit | Accuracy |", "|---|---|---:|",
        f"| Majority Female | Call | {m['majority_baseline_accuracy']:.6%} |",
        f"| F0 + Acoustic + Logistic Regression | Call | {m['overall']['accuracy']:.6%} |", "",
        f"절대 향상폭은 {m['absolute_improvement']*100:+.6f} percentage points이다. Male accuracy {m['gender_accuracy']['M']['accuracy']:.6%}, "
        f"Female accuracy {m['gender_accuracy']['F']['accuracy']:.6%}. 예측 수는 M {m['prediction_counts'].get('M',0):,}, F {m['prediction_counts'].get('F',0):,}이다.", "",
        "Confusion matrix의 행은 실제 [M, F], 열은 예측 [M, F]이다.", "",
        f"| | Pred M | Pred F |", "|---|---:|---:|",
        f"| True M | {cm[0][0]:,} | {cm[0][1]:,} |",
        f"| True F | {cm[1][0]:,} | {cm[1][1]:,} |", "",
        "### 신고자 segment 총 길이별", "",
        "기존 EDA의 call duration 범위를 보고 모델 평가 전에 `<8, 8–16, 16–32, 32–64, >=64초`로 고정했다.", "",
        "| 구간 | n | Accuracy |", "|---|---:|---:|",
    ]
    for name, value in m["duration_groups"].items():
        acc = "N/A" if value["accuracy"] is None else f"{value['accuracy']:.3%}"
        lines.append(f"| {name} | {value['n']:,} | {acc} |")
    lines += ["", "### F0 품질별", "", "| 그룹 | n | Accuracy |", "|---|---:|---:|"]
    for name, value in m["f0_quality_groups"].items():
        acc = "N/A" if value["accuracy"] is None else f"{value['accuracy']:.3%}"
        lines.append(f"| {name} | {value['n']:,} | {acc} |")
    lines += [
        "", "품질 그룹은 원인 규명용 기술 통계이며 서로 겹칠 수 있다. 낮은 voiced ratio 또는 F0 실패군에서도 duration/RMS/ZCR/centroid 특징과 Train 중앙값 대체가 남아 있어 순수 F0-only 성능으로 해석하면 안 된다.", "",
        "## 재현성과 한계", "",
        f"실행 명령: `{command}`", "",
        "라이브러리: " + ", ".join(f"{k}={v}" for k, v in versions.items()), "",
        "`call_features.csv`는 모델에 사용한 통화 특징과 품질 집계를, `val_predictions.csv`는 고정 Validation 예측을 담는다. "
        "실행기는 기존 결과 파일이 있으면 중단하므로 재실행 시 별도 `--out-dir`을 지정해야 한다.", "",
        "동일 사람의 재신고를 연결하는 식별자가 없어 speaker-level leakage는 확인할 수 없다. annotation overlap, tone/non-speech 혼입, octave error, 감정·긴장·울음에 따른 pitch 변화는 남은 위험이다. "
        "이번 결과는 해석 가능한 첫 acoustic baseline이며 F0만으로 성별이 충분하다는 결론을 뜻하지 않는다.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--split", type=Path, default=here.parents[1] / "validation/split_assignments.csv")
    parser.add_argument("--manifest", type=Path, default=here.parents[1] / "validation/manifests/calls.csv")
    parser.add_argument("--segments", type=Path, default=here.parents[1] / "eda/eda_outputs/segments.csv")
    parser.add_argument("--out-dir", type=Path, default=here / "results")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--limit", type=int, help="Development only; cannot train/evaluate")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Output directory already exists; use a new --out-dir (no overwrite mode)")
    if args.workers < 1:
        parser.error("--workers must be positive")
    jobs = load_jobs(args.split.resolve(), args.manifest.resolve(), args.segments.resolve())
    if args.limit:
        jobs = jobs[:args.limit]
    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(aggregate_call, *zip(*jobs), chunksize=4), 1):
            rows.append(row)
            if index % 500 == 0 or index == len(jobs):
                print(f"Feature extraction {index:,} / {len(jobs):,}", flush=True)
    frame = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True)
    frame.to_csv(args.out_dir / "call_features.csv", index=False, lineterminator="\n")
    if args.limit:
        print("Development subset complete; no model evaluation performed")
        return
    split_hash = sha256_file(args.split.resolve())
    predictions, metrics, _ = evaluate(frame, split_hash)
    predictions.to_csv(args.out_dir / "val_predictions.csv", index=False, lineterminator="\n")
    versions = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "soundfile": sf.__version__, "praat-parselmouth": parselmouth.__version__,
    }
    metrics["runtime"] = {"elapsed_seconds": time.time() - started, "workers": args.workers,
                          "versions": versions}
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    command = f"python {Path(__file__).name} --workers {args.workers}"
    write_report(args.out_dir / "REPORT.md", metrics, versions, command)
    print(json.dumps({"accuracy": metrics["overall"]["accuracy"],
                      "male_accuracy": metrics["gender_accuracy"]["M"]["accuracy"],
                      "female_accuracy": metrics["gender_accuracy"]["F"]["accuracy"],
                      "f0_quality": metrics["f0_quality"]}, indent=2))


if __name__ == "__main__":
    main()

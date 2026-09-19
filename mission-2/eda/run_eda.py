#!/usr/bin/env python3
"""Mission 2 데이터 구조와 발화 라벨을 분석한다.

모델 입력 제한을 지키기 위해 전사 text와 발화 순서는 읽거나 출력하지 않는다.
사용 필드는 speaker, startAt, endAt이며 WAV는 헤더 정보만 읽는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import unicodedata
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DURATION_BINS = [
    ("<0.25s", 0.0, 0.25),
    ("0.25-0.5s", 0.25, 0.5),
    ("0.5-1.0s", 0.5, 1.0),
    ("1.0-1.5s", 1.0, 1.5),
    ("1.5-2.0s", 1.5, 2.0),
    ("2.0-3.0s", 2.0, 3.0),
    ("3.0-5.0s", 3.0, 5.0),
    (">=5.0s", 5.0, math.inf),
]


def normalized_stem(path: Path) -> str:
    return unicodedata.normalize("NFC", path.stem)


def safe_files(root: Path, suffix: str) -> list[Path]:
    files = []
    for path in root.rglob(f"*{suffix}"):
        if not path.is_file():
            continue
        if path.name.startswith("._") or "__MACOSX" in path.parts:
            continue
        files.append(path)
    return sorted(files)


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


def index_by_stem(paths: Iterable[Path]) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        indexed[normalized_stem(path)].append(path)
    return dict(indexed)


def call_ref(stem: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", stem).encode("utf-8")).hexdigest()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p25", "median", "p75", "p95", "max", "mean")}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * p
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)

    return {
        "min": round(ordered[0], 6),
        "p25": round(percentile(0.25), 6),
        "median": round(percentile(0.50), 6),
        "p75": round(percentile(0.75), 6),
        "p95": round(percentile(0.95), 6),
        "max": round(ordered[-1], 6),
        "mean": round(statistics.fmean(ordered), 6),
    }


def read_wav_header(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width_bits = wav.getsampwidth() * 8
        n_frames = wav.getnframes()
    duration_sec = n_frames / sample_rate if sample_rate else 0.0
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width_bits,
        "audio_duration_sec": duration_sec,
    }


def duration_bin(duration_sec: float) -> str:
    for name, lower, upper in DURATION_BINS:
        if lower <= duration_sec < upper:
            return name
    return DURATION_BINS[-1][0]


def analyze_split(
    split: str,
    split_dir: Path,
    limit_calls: int,
    skip_wav_headers: bool,
) -> tuple[dict, list[dict], Counter, Counter, list[dict]]:
    wav_index = index_by_stem(safe_files(split_dir, ".wav"))
    json_index = index_by_stem(safe_files(split_dir, ".json"))
    all_stems = sorted(set(wav_index) | set(json_index))
    if limit_calls:
        all_stems = all_stems[:limit_calls]

    call_rows: list[dict] = []
    issue_rows: list[dict] = []
    speaker_counts: Counter = Counter()
    duration_bins: Counter = Counter()
    durations_by_speaker: dict[int, list[float]] = {0: [], 1: []}
    sample_rates: Counter = Counter()
    channels: Counter = Counter()
    sample_widths: Counter = Counter()
    totals = Counter()

    totals["wav_files"] = sum(len(paths) for paths in wav_index.values())
    totals["json_files"] = sum(len(paths) for paths in json_index.values())
    totals["unique_stems"] = len(set(wav_index) | set(json_index))
    totals["duplicate_wav_stems"] = sum(1 for paths in wav_index.values() if len(paths) > 1)
    totals["duplicate_json_stems"] = sum(1 for paths in json_index.values() if len(paths) > 1)

    for stem in all_stems:
        wav_paths = wav_index.get(stem, [])
        json_paths = json_index.get(stem, [])
        ref = call_ref(stem)
        row = {
            "split": split,
            "call_ref": ref,
            "wav_present": int(len(wav_paths) == 1),
            "json_present": int(len(json_paths) == 1),
            "json_ok": 0,
            "wav_header_ok": 0,
            "sample_rate": "",
            "channels": "",
            "sample_width_bits": "",
            "audio_duration_sec": "",
            "utterances_total": 0,
            "speaker_0_count": 0,
            "speaker_1_count": 0,
            "speaker_0_duration_sec": 0.0,
            "speaker_1_duration_sec": 0.0,
            "invalid_utterances": 0,
            "out_of_audio_bounds": 0,
        }

        if len(wav_paths) == 0:
            totals["wav_missing"] += 1
            issue_rows.append({"split": split, "call_ref": ref, "issue": "wav_missing", "count": 1})
        elif len(wav_paths) > 1:
            totals["wav_duplicate"] += 1
            issue_rows.append({"split": split, "call_ref": ref, "issue": "wav_duplicate", "count": len(wav_paths)})

        if len(json_paths) == 0:
            totals["json_missing"] += 1
            issue_rows.append({"split": split, "call_ref": ref, "issue": "json_missing", "count": 1})
        elif len(json_paths) > 1:
            totals["json_duplicate"] += 1
            issue_rows.append({"split": split, "call_ref": ref, "issue": "json_duplicate", "count": len(json_paths)})

        wav_duration_sec = None
        if len(wav_paths) == 1 and not skip_wav_headers:
            try:
                header = read_wav_header(wav_paths[0])
                row.update(header)
                row["wav_header_ok"] = 1
                wav_duration_sec = float(header["audio_duration_sec"])
                sample_rates[header["sample_rate"]] += 1
                channels[header["channels"]] += 1
                sample_widths[header["sample_width_bits"]] += 1
            except (wave.Error, OSError, EOFError) as exc:
                totals["wav_header_errors"] += 1
                issue_rows.append(
                    {"split": split, "call_ref": ref, "issue": f"wav_header_error:{type(exc).__name__}", "count": 1}
                )

        if len(json_paths) == 1:
            try:
                with json_paths[0].open("r", encoding="utf-8") as handle:
                    label = json.load(handle)
                utterances = label.get("utterances", [])
                if not isinstance(utterances, list):
                    raise ValueError("utterances_not_list")
                row["json_ok"] = 1
                totals["json_ok"] += 1
                for utterance in utterances:
                    row["utterances_total"] += 1
                    totals["utterances_total"] += 1
                    if not isinstance(utterance, dict):
                        row["invalid_utterances"] += 1
                        continue
                    speaker = utterance.get("speaker")
                    start_at = utterance.get("startAt")
                    end_at = utterance.get("endAt")
                    try:
                        speaker = int(speaker)
                        start_at = float(start_at)
                        end_at = float(end_at)
                    except (TypeError, ValueError):
                        row["invalid_utterances"] += 1
                        continue
                    if speaker not in (0, 1) or start_at < 0 or end_at <= start_at:
                        row["invalid_utterances"] += 1
                        continue

                    duration_sec = (end_at - start_at) / 1000.0
                    speaker_counts[(split, speaker)] += 1
                    duration_bins[(split, speaker, duration_bin(duration_sec))] += 1
                    durations_by_speaker[speaker].append(duration_sec)
                    row[f"speaker_{speaker}_count"] += 1
                    row[f"speaker_{speaker}_duration_sec"] += duration_sec
                    totals["utterances_valid"] += 1

                    if wav_duration_sec is not None and end_at / 1000.0 > wav_duration_sec + 0.05:
                        row["out_of_audio_bounds"] += 1

                if row["invalid_utterances"]:
                    totals["invalid_utterances"] += row["invalid_utterances"]
                    issue_rows.append(
                        {
                            "split": split,
                            "call_ref": ref,
                            "issue": "invalid_utterance",
                            "count": row["invalid_utterances"],
                        }
                    )
                if row["out_of_audio_bounds"]:
                    totals["out_of_audio_bounds"] += row["out_of_audio_bounds"]
                    issue_rows.append(
                        {
                            "split": split,
                            "call_ref": ref,
                            "issue": "out_of_audio_bounds",
                            "count": row["out_of_audio_bounds"],
                        }
                    )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                totals["json_errors"] += 1
                issue_rows.append(
                    {"split": split, "call_ref": ref, "issue": f"json_error:{type(exc).__name__}", "count": 1}
                )

        row["speaker_0_duration_sec"] = round(float(row["speaker_0_duration_sec"]), 6)
        row["speaker_1_duration_sec"] = round(float(row["speaker_1_duration_sec"]), 6)
        if isinstance(row["audio_duration_sec"], float):
            row["audio_duration_sec"] = round(float(row["audio_duration_sec"]), 6)
        call_rows.append(row)

    summary = {
        "split": split,
        "calls_scanned": len(all_stems),
        "limited_run": bool(limit_calls),
        "file_counts": dict(totals),
        "sample_rates": {str(key): value for key, value in sorted(sample_rates.items())},
        "channels": {str(key): value for key, value in sorted(channels.items())},
        "sample_width_bits": {str(key): value for key, value in sorted(sample_widths.items())},
        "duration_seconds": {
            "speaker_0": quantiles(durations_by_speaker[0]),
            "speaker_1": quantiles(durations_by_speaker[1]),
        },
    }
    return summary, call_rows, speaker_counts, duration_bins, issue_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_plots(out_dir: Path, speaker_rows: list[dict], bin_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib이 없어 PNG 그래프 생성을 건너뜁니다.")
        return

    splits = sorted({row["split"] for row in speaker_rows})
    x = list(range(len(splits)))
    width = 0.36
    counts0 = [next((row["utterance_count"] for row in speaker_rows if row["split"] == split and row["speaker"] == 0), 0) for split in splits]
    counts1 = [next((row["utterance_count"] for row in speaker_rows if row["split"] == split and row["speaker"] == 1), 0) for split in splits]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([value - width / 2 for value in x], counts0, width, label="119 operator (0)")
    ax.bar([value + width / 2 for value in x], counts1, width, label="Reporter (1)")
    ax.set_xticks(x, splits)
    ax.set_ylabel("Utterances")
    ax.set_title("Mission 2 speaker counts")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "speaker_counts.png", dpi=160)
    plt.close(fig)

    bin_names = [name for name, _, _ in DURATION_BINS]
    fig, axes = plt.subplots(max(len(splits), 1), 1, figsize=(10, 4 * max(len(splits), 1)), squeeze=False)
    for axis, split in zip(axes[:, 0], splits):
        values0 = [next((row["utterance_count"] for row in bin_rows if row["split"] == split and row["speaker"] == 0 and row["duration_bin"] == name), 0) for name in bin_names]
        values1 = [next((row["utterance_count"] for row in bin_rows if row["split"] == split and row["speaker"] == 1 and row["duration_bin"] == name), 0) for name in bin_names]
        pos = list(range(len(bin_names)))
        axis.bar([value - width / 2 for value in pos], values0, width, label="119 operator (0)")
        axis.bar([value + width / 2 for value in pos], values1, width, label="Reporter (1)")
        axis.set_xticks(pos, bin_names, rotation=35, ha="right")
        axis.set_ylabel("Utterances")
        axis.set_title(f"{split} utterance duration")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "duration_distribution.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission 2 EDA")
    parser.add_argument("--data-root", required=True, help="Training/과 Validation/을 포함하는 폴더")
    parser.add_argument("--out-dir", default="mission2_eda_results")
    parser.add_argument("--limit-calls", type=int, default=0, help="split별 앞 N통화만 점검; 0이면 전체")
    parser.add_argument("--skip-wav-headers", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    call_rows: list[dict] = []
    issue_rows: list[dict] = []
    all_speaker_counts: Counter = Counter()
    all_duration_bins: Counter = Counter()

    for split in ("Training", "Validation"):
        split_dir = find_split_dir(data_root, split)
        print(f"[{split}] {split_dir}")
        summary, split_calls, speaker_counts, duration_bins, split_issues = analyze_split(
            split,
            split_dir,
            args.limit_calls,
            args.skip_wav_headers,
        )
        summaries[split] = summary
        call_rows.extend(split_calls)
        issue_rows.extend(split_issues)
        all_speaker_counts.update(speaker_counts)
        all_duration_bins.update(duration_bins)
        print(
            f"  calls={summary['calls_scanned']:,} "
            f"valid_utterances={summary['file_counts'].get('utterances_valid', 0):,}"
        )

    speaker_rows = []
    for split in ("Training", "Validation"):
        for speaker in (0, 1):
            count = all_speaker_counts[(split, speaker)]
            speaker_rows.append(
                {
                    "split": split,
                    "speaker": speaker,
                    "role": "119_operator" if speaker == 0 else "reporter",
                    "utterance_count": count,
                }
            )

    bin_rows = []
    for split in ("Training", "Validation"):
        for speaker in (0, 1):
            total = all_speaker_counts[(split, speaker)]
            for name, _, _ in DURATION_BINS:
                count = all_duration_bins[(split, speaker, name)]
                bin_rows.append(
                    {
                        "split": split,
                        "speaker": speaker,
                        "role": "119_operator" if speaker == 0 else "reporter",
                        "duration_bin": name,
                        "utterance_count": count,
                        "share": round(count / total, 8) if total else 0.0,
                    }
                )

    summary_document = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "mission": "speaker classification",
            "allowed_fields": ["speaker", "startAt", "endAt", "WAV header"],
            "excluded_from_outputs": ["text", "raw file stem", "absolute path", "utterance order feature"],
            "data_root_name": data_root.name,
            "limit_calls_per_split": args.limit_calls,
            "wav_headers_skipped": args.skip_wav_headers,
        },
        "splits": summaries,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_document, handle, ensure_ascii=False, indent=2)

    write_csv(
        out_dir / "speaker_counts.csv",
        speaker_rows,
        ["split", "speaker", "role", "utterance_count"],
    )
    write_csv(
        out_dir / "duration_bins.csv",
        bin_rows,
        ["split", "speaker", "role", "duration_bin", "utterance_count", "share"],
    )
    call_fields = [
        "split",
        "call_ref",
        "wav_present",
        "json_present",
        "json_ok",
        "wav_header_ok",
        "sample_rate",
        "channels",
        "sample_width_bits",
        "audio_duration_sec",
        "utterances_total",
        "speaker_0_count",
        "speaker_1_count",
        "speaker_0_duration_sec",
        "speaker_1_duration_sec",
        "invalid_utterances",
        "out_of_audio_bounds",
    ]
    write_csv(out_dir / "call_summary.csv", call_rows, call_fields)
    write_csv(out_dir / "quality_issues.csv", issue_rows, ["split", "call_ref", "issue", "count"])
    create_plots(out_dir, speaker_rows, bin_rows)
    print(f"결과 저장: {out_dir}")


if __name__ == "__main__":
    main()

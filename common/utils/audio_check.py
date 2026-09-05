"""soundfile로 오디오 파일들의 샘플레이트/채널/길이 일관성을 점검하는 스크립트.

전처리(padding/sliding window, MFCC/Mel 추출) 파이프라인을 설계하기 전에
전체 오디오가 동일한 샘플레이트/채널 수를 쓰는지, 길이 분포가 어느 정도인지 먼저 확인해야
Mission 1/2 팀원(B, C)에게 정확한 전처리 가이드를 줄 수 있다.

soundfile.info()는 파일을 전부 디코딩하지 않고 헤더만 읽으므로 대용량 폴더에도 빠르게 동작한다.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import List, Optional

import soundfile as sf

from common import config
from common.utils.io import collect_split_pairs


class AudioInfo:
    __slots__ = ("stem", "samplerate", "channels", "duration_sec", "error")

    def __init__(self, stem: str, samplerate: Optional[int], channels: Optional[int],
                 duration_sec: Optional[float], error: Optional[str] = None):
        self.stem = stem
        self.samplerate = samplerate
        self.channels = channels
        self.duration_sec = duration_sec
        self.error = error


def inspect_wav(path: Path) -> AudioInfo:
    """단일 wav 파일의 샘플레이트/채널/길이를 읽는다. 손상 파일은 error 필드에 메시지를 남긴다."""
    try:
        info = sf.info(str(path))
        return AudioInfo(
            stem=path.stem,
            samplerate=info.samplerate,
            channels=info.channels,
            duration_sec=info.frames / info.samplerate if info.samplerate else None,
        )
    except Exception as exc:  # soundfile은 다양한 백엔드 예외를 던질 수 있음
        return AudioInfo(stem=path.stem, samplerate=None, channels=None, duration_sec=None, error=str(exc))


def inspect_wavs(paths: List[Path]) -> List[AudioInfo]:
    """여러 wav 파일을 순회하며 inspect_wav를 적용한다."""
    return [inspect_wav(p) for p in paths]


def summarize(infos: List[AudioInfo]) -> dict:
    """샘플레이트/채널 분포와 길이 통계, 에러 개수를 요약한다."""
    ok = [i for i in infos if i.error is None]
    errors = [i for i in infos if i.error is not None]

    sr_counter = Counter(i.samplerate for i in ok)
    ch_counter = Counter(i.channels for i in ok)
    durations = [i.duration_sec for i in ok if i.duration_sec is not None]

    summary = {
        "total": len(infos),
        "ok": len(ok),
        "errors": len(errors),
        "samplerate_distribution": dict(sr_counter),
        "channel_distribution": dict(ch_counter),
    }
    if durations:
        summary["duration_sec_min"] = min(durations)
        summary["duration_sec_max"] = max(durations)
        summary["duration_sec_mean"] = sum(durations) / len(durations)
    return summary


def save_csv(infos: List[AudioInfo], out_path: Path) -> None:
    """개별 파일 단위 점검 결과를 csv로 저장한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "samplerate", "channels", "duration_sec", "error"])
        for info in infos:
            writer.writerow([info.stem, info.samplerate, info.channels, info.duration_sec, info.error])


def main() -> None:
    parser = argparse.ArgumentParser(description="wav 파일 샘플레이트/채널/길이 일관성 체크")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        choices=["Training", "Validation", "both"],
        default="both",
    )
    parser.add_argument("--sample-size", type=int, default=0, help="split당 점검할 파일 수 (0=전체)")
    parser.add_argument("--save-csv-dir", type=str, default=str(config.EDA_OUTPUT_DIR))
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else config.DATA_ROOT
    splits = (
        [config.TRAIN_DIR_NAME, config.VALID_DIR_NAME]
        if args.split == "both"
        else [args.split]
    )

    for split_name in splits:
        pairs = collect_split_pairs(split_name, data_root=data_root)
        wav_paths = [p.wav_path for p in pairs if p.wav_path is not None]
        if args.sample_size:
            wav_paths = wav_paths[: args.sample_size]

        print(f"\n=== {split_name} (wav {len(wav_paths)}개) ===")
        if not wav_paths:
            print("  wav 파일을 찾지 못했습니다.")
            continue

        infos = inspect_wavs(wav_paths)
        summary = summarize(infos)
        for key, value in summary.items():
            print(f"  {key}: {value}")

        if summary["errors"]:
            print("  -- 손상/읽기 실패 파일 예시 --")
            for info in [i for i in infos if i.error is not None][:5]:
                print(f"    {info.stem}: {info.error}")

        if len(summary.get("samplerate_distribution", {})) > 1:
            print("  [경고] 샘플레이트가 통일되어 있지 않습니다. 전처리 시 resampling 필요.")
        if len(summary.get("channel_distribution", {})) > 1:
            print("  [경고] 채널 수가 통일되어 있지 않습니다. 전처리 시 mono 변환 필요.")

        if args.save_csv_dir:
            out_path = Path(args.save_csv_dir) / f"audio_check_{split_name}.csv"
            save_csv(infos, out_path)
            print(f"  -> 상세 결과 저장: {out_path}")


if __name__ == "__main__":
    main()

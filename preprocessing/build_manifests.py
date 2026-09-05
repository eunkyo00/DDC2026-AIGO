"""미션별 학습에 바로 쓸 수 있는 전처리 매니페스트(구간 인덱스)를 생성하는 스크립트.

오디오 원본을 미리 잘라 별도 파일로 저장하지 않는다. wav는 그대로 두고, 각 미션이
학습 시 필요한 (파일 경로 + 시작/종료 시각(ms) + 라벨) 인덱스만 csv로 만든다.
실제 crop/pad/sliding window/MFCC 등은 각 미션의 dataloader에서
preprocessing.preprocessing_utils 함수로 그때그때 수행하면 된다
(발화 단위로 실제 오디오 파일을 미리 잘라두면 통화당 수십 개씩 파일이 늘어나
디스크 용량이 감당되지 않기 때문).

생성되는 매니페스트 3종:
  - mission1_gender_{split}.csv   : 통화 1개 = 행 1개. 신고자(speaker=1) 발화 구간 리스트 + gender 라벨.
  - mission2_speaker_{split}.csv  : 발화(utterance) 1개 = 행 1개. speaker(0/1) 라벨.
  - mission3_symptom_{split}.csv  : 통화 1개 = 행 1개. 발화 텍스트 리스트 + 9개 클래스로 필터링된 symptom 라벨.

대회 규정상 학습은 서울 데이터만 사용해야 하므로, 매니페스트 생성 시 항상
utils.filter_seoul.is_seoul로 한 번 더 필터링한다.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import config
from utils.filter_seoul import is_seoul
from utils.io import collect_split_pairs
from preprocessing.preprocessing_utils import get_speaker_id, get_symptom_labels, get_utterances


def _load_label(json_path: Path) -> Optional[dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def write_mission1_manifest(pairs, split_name: str, out_dir: Path) -> dict:
    """통화 단위: 신고자(speaker=1) 발화 구간 리스트 + gender 라벨."""
    out_path = out_dir / f"mission1_gender_{split_name}.csv"
    n_rows = 0
    n_skipped_no_gender = 0
    n_skipped_no_segment = 0
    gender_counts = {"M": 0, "F": 0}

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "wav_path", "gender", "reporter_segments_ms", "num_segments"])

        for pair in pairs:
            if pair.wav_path is None or pair.json_path is None:
                continue
            label = _load_label(pair.json_path)
            if label is None:
                continue

            gender = label.get("gender")
            if gender not in config.GENDER_LABELS:
                n_skipped_no_gender += 1
                continue

            utterances = get_utterances(label)
            segments = [
                [u.get("startAt"), u.get("endAt")]
                for u in utterances
                if get_speaker_id(u) == 1 and u.get("startAt") is not None and u.get("endAt") is not None
            ]
            if not segments:
                n_skipped_no_segment += 1
                continue

            writer.writerow([pair.stem, str(pair.wav_path), gender, json.dumps(segments), len(segments)])
            gender_counts[gender] += 1
            n_rows += 1

    return {
        "out_path": out_path,
        "n_rows": n_rows,
        "n_skipped_no_gender": n_skipped_no_gender,
        "n_skipped_no_segment": n_skipped_no_segment,
        "gender_counts": gender_counts,
    }


def write_mission2_manifest(pairs, split_name: str, out_dir: Path) -> dict:
    """발화 단위: 발화 1개 = 행 1개. speaker(0/1) 라벨.

    Mission 2는 텍스트/순서 정보를 모델 입력으로 쓸 수 없으므로, 매니페스트에도
    text는 기록하지 않는다 (참고용으로도 남기지 않아 실수로 흘러들어가는 것을 방지).
    """
    out_path = out_dir / f"mission2_speaker_{split_name}.csv"
    n_rows = 0
    speaker_counts = {0: 0, 1: 0}

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "wav_path", "utterance_id", "startAt", "endAt", "speaker"])

        for pair in pairs:
            if pair.wav_path is None or pair.json_path is None:
                continue
            label = _load_label(pair.json_path)
            if label is None:
                continue

            for u in get_utterances(label):
                speaker = get_speaker_id(u)
                start_at, end_at = u.get("startAt"), u.get("endAt")
                if speaker not in (0, 1) or start_at is None or end_at is None:
                    continue
                writer.writerow([pair.stem, str(pair.wav_path), u.get("id", ""), start_at, end_at, speaker])
                speaker_counts[speaker] += 1
                n_rows += 1

    return {"out_path": out_path, "n_rows": n_rows, "speaker_counts": speaker_counts}


def write_mission3_manifest(pairs, split_name: str, out_dir: Path) -> dict:
    """통화 단위: 발화 텍스트 리스트 + 9개 클래스로 필터링된 symptom 라벨."""
    out_path = out_dir / f"mission3_symptom_{split_name}.csv"
    n_rows = 0
    n_empty_symptom = 0
    symptom_counts = {cls: 0 for cls in config.SYMPTOM_CLASSES}

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "json_path", "utterance_texts", "symptom_labels", "num_symptoms"])

        for pair in pairs:
            if pair.json_path is None:
                continue
            label = _load_label(pair.json_path)
            if label is None:
                continue

            utterances = get_utterances(label)
            texts = [u.get("text", "") for u in utterances if u.get("text")]
            symptoms = get_symptom_labels(label)
            if not symptoms:
                n_empty_symptom += 1

            for s in symptoms:
                symptom_counts[s] += 1

            writer.writerow([pair.stem, str(pair.json_path), json.dumps(texts, ensure_ascii=False), ";".join(symptoms), len(symptoms)])
            n_rows += 1

    return {
        "out_path": out_path,
        "n_rows": n_rows,
        "n_empty_symptom": n_empty_symptom,
        "symptom_counts": symptom_counts,
    }


def build_split(split_name: str, data_root: Optional[Path], out_dir: Path, seoul_only: bool) -> None:
    print(f"\n=== {split_name} ===")
    pairs = collect_split_pairs(split_name, data_root=data_root)
    pairs = [p for p in pairs if p.wav_path is not None and p.json_path is not None]
    print(f"  전체 매칭 쌍: {len(pairs)}")

    if seoul_only:
        filtered = []
        for p in pairs:
            label = _load_label(p.json_path)
            if label is not None and is_seoul(label):
                filtered.append(p)
        print(f"  서울 필터 적용 후: {len(filtered)}")
        pairs = filtered

    m1 = write_mission1_manifest(pairs, split_name, out_dir)
    print(f"  [mission1] rows={m1['n_rows']} gender={m1['gender_counts']} "
          f"skip(no_gender)={m1['n_skipped_no_gender']} skip(no_segment)={m1['n_skipped_no_segment']}")
    print(f"    -> {m1['out_path']}")

    m2 = write_mission2_manifest(pairs, split_name, out_dir)
    print(f"  [mission2] rows={m2['n_rows']} speaker_counts={m2['speaker_counts']}")
    print(f"    -> {m2['out_path']}")

    m3 = write_mission3_manifest(pairs, split_name, out_dir)
    top_symptoms = dict(sorted(m3["symptom_counts"].items(), key=lambda kv: -kv[1]))
    print(f"  [mission3] rows={m3['n_rows']} empty_symptom_calls={m3['n_empty_symptom']}")
    print(f"    symptom_counts={top_symptoms}")
    print(f"    -> {m3['out_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="미션 1/2/3 전처리 매니페스트 생성")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["Training", "Validation", "both"], default="both")
    parser.add_argument("--out-dir", type=str, default="manifests")
    parser.add_argument(
        "--no-seoul-filter",
        action="store_true",
        help="서울 필터를 적용하지 않음 (기본은 적용됨; 현재 데이터는 어차피 100% 서울)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else config.DATA_ROOT
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [config.TRAIN_DIR_NAME, config.VALID_DIR_NAME] if args.split == "both" else [args.split]
    for split_name in splits:
        build_split(split_name, data_root, out_dir, seoul_only=not args.no_seoul_filter)


if __name__ == "__main__":
    main()

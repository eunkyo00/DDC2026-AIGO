"""전체 라벨 분포 EDA: gender / symptom(9개 클래스) / urgencyLevel / disasterMedium /
서울-비서울 비율 / speaker 분포를 집계해서 eda_outputs/에 csv와 plot으로 저장한다.

다른 팀원(B/C/D)이 각자 미션 모델링을 시작하기 전에 클래스 불균형, 결측치, 서울 비율 등을
한눈에 볼 수 있게 하는 것이 목적이다.

사용법:
    python -m eda.eda_all
    python -m eda.eda_all --split Training --sample-size 500
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # 서버/헤드리스 환경에서도 저장 가능하도록
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import config

# 라벨 값(gender/urgencyLevel/symptom 등)이 한글이라 기본 폰트(DejaVu Sans)로는 깨져 보인다.
# 팀원 OS(Windows/Mac)에 흔히 설치된 한글 폰트 후보를 순서대로 찾아 적용하고,
# 하나도 없으면 경고만 출력하고 기본 폰트로 계속 진행한다 (그림 생성 자체는 실패하지 않게).
_KOREAN_FONT_CANDIDATES = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_matched_font = next((name for name in _KOREAN_FONT_CANDIDATES if name in _available_fonts), None)
if _matched_font:
    plt.rcParams["font.family"] = _matched_font
    plt.rcParams["axes.unicode_minus"] = False
else:
    print(
        "[경고] 한글 폰트를 찾지 못했습니다. plot의 한글 라벨이 깨져 보일 수 있습니다. "
        "(Windows: 'Malgun Gothic', Mac: 'AppleGothic'/'NanumGothic' 설치 여부 확인)"
    )
from utils.filter_seoul import is_seoul
from utils.io import collect_split_pairs
from preprocessing.preprocessing_utils import get_symptom_labels, get_speaker_id, get_utterances


def _safe_get(d: dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def build_records(split_name: str, data_root: Optional[Path], sample_size: Optional[int]) -> List[dict]:
    """한 split(Training/Validation)의 라벨 json들을 읽어 EDA용 flat record 리스트를 만든다."""
    pairs = collect_split_pairs(split_name, data_root=data_root)
    labeled_pairs = [p for p in pairs if p.json_path is not None]
    if sample_size:
        labeled_pairs = labeled_pairs[:sample_size]

    records = []
    for pair in labeled_pairs:
        try:
            with open(pair.json_path, "r", encoding="utf-8") as f:
                label = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue

        utterances = get_utterances(label)
        speaker_ids = [get_speaker_id(u) for u in utterances]
        start_at = _safe_get(label, "startAt")
        end_at = _safe_get(label, "endAt")
        duration_sec = (end_at - start_at) / 1000.0 if isinstance(start_at, (int, float)) and isinstance(end_at, (int, float)) else None

        records.append(
            {
                "stem": pair.stem,
                "split": split_name,
                "has_wav": pair.wav_path is not None,
                "is_seoul": is_seoul(label),
                "gender": _safe_get(label, "gender"),
                "disasterLarge": _safe_get(label, "disasterLarge"),
                "disasterMedium": _safe_get(label, "disasterMedium"),
                "urgencyLevel": _safe_get(label, "urgencyLevel"),
                "sentiment": _safe_get(label, "sentiment"),
                "triage": _safe_get(label, "triage"),
                "symptom_raw": _safe_get(label, "symptom") or [],
                "symptom_filtered": get_symptom_labels(label),
                "num_utterances": len(utterances),
                "num_speaker_0": speaker_ids.count(0),
                "num_speaker_1": speaker_ids.count(1),
                "duration_sec": duration_sec,
            }
        )
    return records


def save_records_csv(records: List[dict], out_path: Path) -> pd.DataFrame:
    """record 리스트를 csv로 저장하고 DataFrame으로 반환한다 (symptom 리스트는 ';'로 join)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df_to_save = df.copy()
    if not df_to_save.empty:
        df_to_save["symptom_raw"] = df_to_save["symptom_raw"].apply(lambda x: ";".join(x) if x else "")
        df_to_save["symptom_filtered"] = df_to_save["symptom_filtered"].apply(lambda x: ";".join(x) if x else "")
    df_to_save.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def plot_value_counts(series: pd.Series, title: str, out_path: Path, top_n: int = 20) -> None:
    """단일 범주형 컬럼의 분포를 막대그래프로 저장한다."""
    counts = series.dropna().value_counts().head(top_n)
    if counts.empty:
        return
    plt.figure(figsize=(8, max(3, 0.4 * len(counts))))
    counts.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_symptom_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """9개 타겟 symptom 클래스별 등장 횟수(멀티라벨이므로 합산)를 막대그래프로 저장한다."""
    counts = {cls: 0 for cls in config.SYMPTOM_CLASSES}
    for symptoms in df["symptom_filtered"]:
        for s in symptoms:
            if s in counts:
                counts[s] += 1

    series = pd.Series(counts).sort_values()
    plt.figure(figsize=(8, 4))
    series.plot(kind="barh")
    plt.title("Symptom class distribution (target 9 classes)")
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_eda(split_name: str, data_root: Optional[Path], sample_size: Optional[int], out_dir: Path) -> None:
    print(f"\n=== EDA: {split_name} ===")
    records = build_records(split_name, data_root, sample_size)
    if not records:
        print("  라벨 데이터를 찾지 못했습니다. config.DATA_ROOT 설정을 확인하세요.")
        return

    df = save_records_csv(records, out_dir / f"labels_{split_name}.csv")
    print(f"  레코드 수: {len(df)}  -> csv 저장 완료")

    print("  gender 분포:")
    print(df["gender"].value_counts(dropna=False).to_string())
    print("  서울/비서울 분포:")
    print(df["is_seoul"].value_counts(dropna=False).to_string())
    print("  urgencyLevel 분포:")
    print(df["urgencyLevel"].value_counts(dropna=False).to_string())

    plot_value_counts(df["gender"], f"{split_name} - gender", out_dir / f"gender_{split_name}.png")
    plot_value_counts(df["urgencyLevel"], f"{split_name} - urgencyLevel", out_dir / f"urgency_{split_name}.png")
    plot_value_counts(df["disasterMedium"], f"{split_name} - disasterMedium", out_dir / f"disaster_medium_{split_name}.png")
    plot_value_counts(df["sentiment"], f"{split_name} - sentiment", out_dir / f"sentiment_{split_name}.png")
    plot_symptom_distribution(df, out_dir / f"symptom_{split_name}.png")

    print(f"  -> plot 저장 완료: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨 분포 EDA 실행")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        choices=["Training", "Validation", "both"],
        default="both",
    )
    parser.add_argument("--sample-size", type=int, default=0, help="split당 사용할 파일 수 (0=전체)")
    parser.add_argument("--out-dir", type=str, default=str(config.EDA_OUTPUT_DIR))
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else config.DATA_ROOT
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_size = args.sample_size or None

    splits = (
        [config.TRAIN_DIR_NAME, config.VALID_DIR_NAME]
        if args.split == "both"
        else [args.split]
    )
    for split_name in splits:
        run_eda(split_name, data_root, sample_size, out_dir)


if __name__ == "__main__":
    main()

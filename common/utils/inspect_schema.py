"""라벨(json) 스키마 자동 탐색 스크립트.

이 프로젝트의 어떤 코드도 json 키 경로를 하드코딩하지 않는다. 대신 이 스크립트를
**제일 먼저** 실행해서 실제 데이터의 키 구성, 타입, 카디널리티(범주형 필드의 고유값 분포)를
확인하고, 그 결과를 바탕으로 config.py의 SYMPTOM_CLASSES/ADDRESS_KEY_CANDIDATES 등이
실제 데이터와 맞는지 검증한다.

사용법:
    python -m utils.inspect_schema
    python -m utils.inspect_schema --split Training --sample-size 200
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import config
from common.utils.io import collect_split_pairs

# 이 이상의 고유값이 나오면 범주형 통계 출력에서 제외 (자유 텍스트/ID성 필드로 간주)
CATEGORICAL_MAX_UNIQUE = 30
# 각 필드별로 예시 값을 몇 개까지 저장할지
MAX_EXAMPLES = 5


def walk_keys(obj: Any, prefix: str = "") -> List[tuple]:
    """중첩된 dict/list 구조를 재귀적으로 순회하며 (key_path, leaf_value) 쌍을 모두 수집한다.

    리스트는 "path[]" 형태로 표시하고 내부 원소를 계속 재귀 탐색한다.
    (예: utterances 리스트의 각 dict 안의 "speaker" 키는 "utterances[].speaker"로 기록됨)

    Args:
        obj: 순회할 값 (dict, list, 또는 leaf 값).
        prefix: 현재까지의 key path.

    Returns:
        (key_path, leaf_value) 튜플 리스트.
    """
    results: List[tuple] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            results.extend(walk_keys(value, path))
    elif isinstance(obj, list):
        path = f"{prefix}[]"
        if not obj:
            results.append((path, None))
        for item in obj:
            results.extend(walk_keys(item, path))
    else:
        results.append((prefix, obj))
    return results


def collect_schema(json_paths: List[Path], sample_size: Optional[int] = None) -> Dict[str, dict]:
    """여러 json 파일을 순회하며 key_path별 등장 횟수/타입/예시값/고유값 분포를 집계한다.

    Args:
        json_paths: 탐색할 json 파일 경로 리스트.
        sample_size: 앞에서부터 몇 개 파일만 볼지. None이면 전체.

    Returns:
        key_path -> {"count": int, "types": Counter, "examples": list, "value_counts": Counter} 딕셔너리.
    """
    paths = json_paths[:sample_size] if sample_size else json_paths
    schema: Dict[str, dict] = defaultdict(
        lambda: {"count": 0, "types": Counter(), "examples": [], "value_counts": Counter()}
    )

    n_parsed = 0
    n_failed = 0
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            n_failed += 1
            continue
        n_parsed += 1

        for key_path, value in walk_keys(data):
            entry = schema[key_path]
            entry["count"] += 1
            entry["types"][type(value).__name__] += 1
            if isinstance(value, (str, int, float, bool)) or value is None:
                entry["value_counts"][value] += 1
            if len(entry["examples"]) < MAX_EXAMPLES:
                entry["examples"].append(value)

    schema["__meta__"] = {"n_parsed": n_parsed, "n_failed": n_failed, "n_total": len(paths)}
    return schema


def print_schema_report(schema: Dict[str, dict]) -> None:
    """collect_schema 결과를 사람이 읽기 좋은 형태로 출력한다."""
    meta = schema.pop("__meta__", {})
    print(f"\n파일 파싱 결과: 성공 {meta.get('n_parsed', 0)} / 실패 {meta.get('n_failed', 0)} "
          f"(총 {meta.get('n_total', 0)})")

    print("\n--- 키 스키마 ---")
    for key_path in sorted(schema.keys()):
        entry = schema[key_path]
        types_str = ", ".join(f"{t}x{c}" for t, c in entry["types"].most_common())
        print(f"  {key_path:40s} count={entry['count']:<6d} types=[{types_str}]")

        n_unique = len(entry["value_counts"])
        if 0 < n_unique <= CATEGORICAL_MAX_UNIQUE:
            top = ", ".join(f"{v!r}:{c}" for v, c in entry["value_counts"].most_common(10))
            print(f"      -> 고유값({n_unique}개): {top}")
        elif entry["examples"]:
            print(f"      -> 예시: {entry['examples'][:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨 json 스키마 자동 탐색")
    parser.add_argument("--data-root", type=str, default=None, help="데이터셋 최상위 경로")
    parser.add_argument(
        "--split",
        type=str,
        choices=["Training", "Validation", "both"],
        default="both",
        help="어느 split의 라벨을 볼지",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="split당 확인할 파일 수 (0=전체)")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else config.DATA_ROOT
    splits = (
        [config.TRAIN_DIR_NAME, config.VALID_DIR_NAME]
        if args.split == "both"
        else [args.split]
    )
    sample_size = None if args.sample_size == 0 else args.sample_size

    for split_name in splits:
        pairs = collect_split_pairs(split_name, data_root=data_root)
        json_paths = [p.json_path for p in pairs if p.json_path is not None]
        print(f"\n############ split: {split_name} (json 파일 {len(json_paths)}개 발견) ############")
        if not json_paths:
            print("  json 파일을 찾지 못했습니다. config.DATA_ROOT / LABEL_DIR_KEYWORD를 확인하세요.")
            continue
        schema = collect_schema(json_paths, sample_size=sample_size)
        print_schema_report(schema)


if __name__ == "__main__":
    main()

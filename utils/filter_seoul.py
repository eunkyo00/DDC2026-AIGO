"""주소 필드 기준 서울 데이터 필터링.

대회 규정상 학습(Training)에는 서울 지역 데이터만 사용해야 한다. 실제 스키마에서는
최상위 "address" 문자열 키 하나로 확인되지만(예: "서울특별시 영등포구 대림동"),
혹시 팀원 환경/향후 데이터 갱신으로 키 이름이나 위치가 달라질 경우를 대비해
1) 알려진 키 후보를 먼저 찾고, 2) 실패하면 전체 json을 재귀적으로 훑어
"서울"이 포함된 문자열 값을 찾는 fallback을 둔다.
"""

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional

import config
from utils.io import collect_split_pairs


def _find_key_recursive(obj: Any, key_candidates: List[str]) -> Optional[Any]:
    """dict/list 구조를 재귀적으로 훑어 key_candidates 중 하나와 이름이 일치하는
    첫 번째 키의 값을 반환한다. 못 찾으면 None.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in key_candidates:
                return value
        for value in obj.values():
            found = _find_key_recursive(value, key_candidates)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key_candidates)
            if found is not None:
                return found
    return None


def _find_string_containing_recursive(obj: Any, substrings: List[str]) -> Optional[str]:
    """재귀적으로 모든 문자열 leaf 값을 훑어 substrings 중 하나라도 포함하는
    첫 번째 문자열을 반환한다 (address 키를 못 찾았을 때의 최종 fallback).
    """
    if isinstance(obj, dict):
        for value in obj.values():
            found = _find_string_containing_recursive(value, substrings)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_string_containing_recursive(item, substrings)
            if found is not None:
                return found
    elif isinstance(obj, str):
        if any(s in obj for s in substrings):
            return obj
    return None


def get_address(label: dict) -> Optional[str]:
    """라벨 json에서 주소 문자열을 찾는다. 알려진 키 우선, 못 찾으면 전체 재귀 탐색.

    Args:
        label: json.load로 읽은 라벨 딕셔너리.

    Returns:
        주소로 보이는 문자열, 없으면 None.
    """
    address = _find_key_recursive(label, config.ADDRESS_KEY_CANDIDATES)
    if isinstance(address, str) and address.strip():
        return address

    # fallback: 키 이름을 못 찾은 경우, 서울 키워드를 포함하는 문자열 값을 직접 탐색
    return _find_string_containing_recursive(label, config.SEOUL_KEYWORDS)


def is_seoul(label: dict) -> bool:
    """라벨의 주소가 서울 지역인지 판별한다.

    Args:
        label: json.load로 읽은 라벨 딕셔너리.

    Returns:
        주소 문자열에 SEOUL_KEYWORDS 중 하나라도 포함되면 True.
    """
    address = get_address(label)
    if not address:
        return False
    return any(keyword in address for keyword in config.SEOUL_KEYWORDS)


def filter_seoul_pairs(pairs, data_root: Optional[Path] = None):
    """PairedFile 리스트 중 라벨이 서울인 것만 걸러 반환한다.

    json이 없는(wav만 있는) 페어는 판별 불가하므로 제외한다.
    """
    seoul_pairs = []
    for pair in pairs:
        if pair.json_path is None:
            continue
        try:
            with open(pair.json_path, "r", encoding="utf-8") as f:
                label = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if is_seoul(label):
            seoul_pairs.append(pair)
    return seoul_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="서울/비서울 라벨 필터링 리포트")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        choices=["Training", "Validation", "both"],
        default="both",
    )
    parser.add_argument(
        "--save-list",
        type=str,
        default=None,
        help="서울 데이터 stem 목록을 저장할 txt 경로 (예: eda_outputs/seoul_train_stems.txt)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else config.DATA_ROOT
    splits = (
        [config.TRAIN_DIR_NAME, config.VALID_DIR_NAME]
        if args.split == "both"
        else [args.split]
    )

    for split_name in splits:
        pairs = collect_split_pairs(split_name, data_root=data_root)
        labeled_pairs = [p for p in pairs if p.json_path is not None]

        seoul_count = 0
        non_seoul_count = 0
        no_address_count = 0
        seoul_stems = []

        for pair in labeled_pairs:
            try:
                with open(pair.json_path, "r", encoding="utf-8") as f:
                    label = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            address = get_address(label)
            if not address:
                no_address_count += 1
                continue
            if is_seoul(label):
                seoul_count += 1
                seoul_stems.append(pair.stem)
            else:
                non_seoul_count += 1

        print(f"\n=== {split_name} ===")
        print(f"  라벨 있는 파일 수 : {len(labeled_pairs)}")
        print(f"  서울             : {seoul_count}")
        print(f"  서울 아님        : {non_seoul_count}")
        print(f"  주소 필드 없음   : {no_address_count}")

        if args.save_list and split_name == splits[0]:
            out_path = Path(args.save_list)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(seoul_stems), encoding="utf-8")
            print(f"  -> 서울 stem 목록 저장: {out_path} ({len(seoul_stems)}개)")


if __name__ == "__main__":
    main()

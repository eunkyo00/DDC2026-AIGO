"""재귀 파일 스캔 + wav/json 파일을 stem(파일명, 확장자 제외) 기준으로 페어링하는 유틸.

실제 배포 데이터는 폴더 depth가 어떻게 바뀌든(예: TS_서울_구급 하위에 날짜별 서브폴더가
추가되는 경우 등) 파일명 stem만 일치하면 매칭되어야 하므로, 특정 depth를 가정하지 않고
전체 서브트리를 재귀적으로 훑는다.

json 내부의 "audioPath" 필드는 원본 배포 서버 기준 경로라 로컬 파일 위치와 다를 수 있으므로
매칭 근거로 사용하지 않는다. 대신 실제 로컬 파일명(stem)을 기준으로 매칭한다.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional

from common import config


class PairedFile(NamedTuple):
    stem: str
    wav_path: Optional[Path]
    json_path: Optional[Path]


def find_dirs_by_keyword(root: Path, keyword: str) -> List[Path]:
    """root 하위(자신 포함)에서 폴더명에 keyword가 포함된 모든 디렉토리를 재귀적으로 찾는다.

    Args:
        root: 탐색을 시작할 최상위 디렉토리.
        keyword: 폴더명에 포함되어야 하는 부분 문자열 (대소문자 무시).

    Returns:
        keyword를 포함하는 디렉토리 경로 리스트 (중복 없음, 발견 순서 유지).
    """
    root = Path(root)
    if not root.exists():
        return []

    keyword_lower = keyword.lower()
    matches: List[Path] = []
    if keyword_lower in root.name.lower():
        matches.append(root)

    for path in root.rglob("*"):
        if path.is_dir() and keyword_lower in path.name.lower():
            matches.append(path)

    return matches


def scan_files(root: Path, ext: str) -> List[Path]:
    """root 하위를 재귀적으로 스캔해 주어진 확장자를 가진 모든 파일 경로를 반환한다.

    Args:
        root: 탐색을 시작할 디렉토리.
        ext: 확장자 (".wav", ".json" 등, 점 포함 여부 상관없음).

    Returns:
        정렬된 파일 경로 리스트.
    """
    root = Path(root)
    if not root.exists():
        return []
    ext = ext if ext.startswith(".") else f".{ext}"
    return sorted(p for p in root.rglob(f"*{ext}") if p.is_file())


def pair_by_stem(
    wav_files: Iterable[Path], json_files: Iterable[Path]
) -> List[PairedFile]:
    """wav 파일 리스트와 json 파일 리스트를 파일명(stem) 기준으로 페어링한다.

    한쪽에만 존재하는 stem도 PairedFile로 포함시키되 없는 쪽은 None으로 채운다.
    (짝이 안 맞는 파일을 조용히 누락시키지 않고 호출자가 보고/디버깅할 수 있게 하기 위함)

    Args:
        wav_files: wav 파일 경로들.
        json_files: json 파일 경로들.

    Returns:
        stem 오름차순으로 정렬된 PairedFile 리스트.
    """
    wav_by_stem: Dict[str, Path] = {p.stem: p for p in wav_files}
    json_by_stem: Dict[str, Path] = {p.stem: p for p in json_files}

    all_stems = sorted(set(wav_by_stem) | set(json_by_stem))
    return [
        PairedFile(stem=stem, wav_path=wav_by_stem.get(stem), json_path=json_by_stem.get(stem))
        for stem in all_stems
    ]


def collect_split_pairs(split_dir_name: str, data_root: Optional[Path] = None) -> List[PairedFile]:
    """Training 또는 Validation 하위의 wav/json을 찾아 stem 기준으로 페어링한다.

    Args:
        split_dir_name: config.TRAIN_DIR_NAME 또는 config.VALID_DIR_NAME 값.
        data_root: 데이터셋 최상위 경로. None이면 config.DATA_ROOT 사용.

    Returns:
        PairedFile 리스트. split 폴더 자체를 못 찾으면 빈 리스트.
    """
    data_root = Path(data_root) if data_root else config.DATA_ROOT
    split_dirs = find_dirs_by_keyword(data_root, split_dir_name)
    if not split_dirs:
        return []

    wav_files: List[Path] = []
    json_files: List[Path] = []
    for split_dir in split_dirs:
        for source_dir in find_dirs_by_keyword(split_dir, config.SOURCE_DIR_KEYWORD):
            wav_files.extend(scan_files(source_dir, config.AUDIO_EXT))
        for label_dir in find_dirs_by_keyword(split_dir, config.LABEL_DIR_KEYWORD):
            json_files.extend(scan_files(label_dir, config.LABEL_EXT))

    return pair_by_stem(wav_files, json_files)


def summarize_pairs(pairs: List[PairedFile]) -> Dict[str, int]:
    """페어링 결과에 대한 매칭/누락 통계를 계산한다."""
    matched = sum(1 for p in pairs if p.wav_path and p.json_path)
    wav_only = sum(1 for p in pairs if p.wav_path and not p.json_path)
    json_only = sum(1 for p in pairs if p.json_path and not p.wav_path)
    return {
        "total": len(pairs),
        "matched": matched,
        "wav_only": wav_only,
        "json_only": json_only,
    }


def _print_report(split_dir_name: str, pairs: List[PairedFile], show_missing: int) -> None:
    stats = summarize_pairs(pairs)
    print(f"\n=== {split_dir_name} ===")
    print(f"  total stems : {stats['total']}")
    print(f"  matched     : {stats['matched']}")
    print(f"  wav only    : {stats['wav_only']}")
    print(f"  json only   : {stats['json_only']}")

    if show_missing > 0:
        missing = [p for p in pairs if not (p.wav_path and p.json_path)][:show_missing]
        if missing:
            print(f"  -- 짝이 안 맞는 파일 예시 (최대 {show_missing}개) --")
            for p in missing:
                side = "wav만 있음" if p.wav_path else "json만 있음"
                print(f"    {p.stem} ({side})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="wav/json 재귀 스캔 및 stem 기준 페어링 리포트")
    parser.add_argument("--data-root", type=str, default=None, help="데이터셋 최상위 경로 (기본값: config.DATA_ROOT)")
    parser.add_argument("--show-missing", type=int, default=5, help="짝이 안 맞는 파일 예시를 몇 개 보여줄지")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else config.DATA_ROOT
    print(f"DATA_ROOT: {data_root}")

    for split_name in (config.TRAIN_DIR_NAME, config.VALID_DIR_NAME):
        pairs = collect_split_pairs(split_name, data_root=data_root)
        _print_report(split_name, pairs, args.show_missing)

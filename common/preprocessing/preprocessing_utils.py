"""미션 1/2/3 공용 전처리 함수 모음.

팀원들이 각자 미션 코드에서 필요한 함수만 골라 import해서 쓸 수 있도록 함수 단위로
잘게 분리했다. 오디오 관련 함수(crop/pad/sliding window/MFCC/Mel)는 Mission 1·2에서,
텍스트 관련 함수(발화 추출/증상 필터링)는 Mission 3에서 주로 사용한다.

의존성 최소화를 위해 utils 패키지에 의존하지 않고, 라벨 json의 키 이름이 조금 달라도
동작하도록 자체적으로 방어적인 키 탐색 헬퍼를 둔다 (실제 데이터 확인 전 작성되었기 때문).

단위 규칙: 라벨 json의 startAt/endAt은 밀리초(ms) 단위다 (utils/inspect_schema.py로 확인).
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

from common import config

# ---------------------------------------------------------------------------
# 오디오 로딩
# ---------------------------------------------------------------------------


def load_audio(path: str, target_sr: Optional[int] = None, mono: bool = True):
    """wav 파일을 읽어 (waveform, samplerate)를 반환한다.

    Args:
        path: wav 파일 경로.
        target_sr: 지정하면 librosa로 리샘플링해서 반환 (샘플레이트가 통일되어 있지 않을 때 사용).
        mono: True면 다채널 오디오를 평균내 mono로 변환.

    Returns:
        (waveform: np.ndarray shape (n_samples,), samplerate: int)
    """
    waveform, sr = sf.read(path, always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)

    if mono and waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if target_sr is not None and target_sr != sr:
        import librosa  # 지연 import: resampling이 필요할 때만 로드

        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return waveform, sr


# ---------------------------------------------------------------------------
# Crop / Pad / Sliding window
# ---------------------------------------------------------------------------


def crop_segment(waveform: np.ndarray, sr: int, start_ms: float, end_ms: float) -> np.ndarray:
    """startAt/endAt(ms) 구간에 해당하는 오디오 조각을 잘라낸다.

    Args:
        waveform: 전체 오디오 파형.
        sr: 샘플레이트.
        start_ms: 발화 시작 시각 (ms).
        end_ms: 발화 종료 시각 (ms).

    Returns:
        잘라낸 파형. 범위가 오디오 길이를 벗어나면 클리핑한다.
    """
    n_samples = len(waveform)
    start_idx = max(0, int(round(start_ms / 1000.0 * sr)))
    end_idx = min(n_samples, int(round(end_ms / 1000.0 * sr)))
    if end_idx <= start_idx:
        return waveform[0:0]
    return waveform[start_idx:end_idx]


def pad_to_length(waveform: np.ndarray, target_len: int, mode: str = "end") -> np.ndarray:
    """짧은 오디오를 0으로 패딩해 target_len 샘플 길이로 맞춘다.

    Args:
        waveform: 입력 파형 (target_len보다 짧아야 함).
        target_len: 목표 샘플 수.
        mode: "end"면 뒤에 패딩, "start"면 앞에 패딩, "center"면 양쪽에 균등 패딩.

    Returns:
        길이가 target_len인 파형. 이미 그 이상 길이면 그대로 반환한다.
    """
    n_samples = len(waveform)
    if n_samples >= target_len:
        return waveform

    pad_total = target_len - n_samples
    if mode == "start":
        pad_width = (pad_total, 0)
    elif mode == "center":
        pad_left = pad_total // 2
        pad_width = (pad_left, pad_total - pad_left)
    else:  # "end"
        pad_width = (0, pad_total)

    return np.pad(waveform, pad_width, mode="constant")


def trim_to_length(waveform: np.ndarray, target_len: int, mode: str = "end") -> np.ndarray:
    """긴 오디오를 target_len 샘플로 잘라낸다.

    Args:
        waveform: 입력 파형.
        target_len: 목표 샘플 수.
        mode: "end"면 앞부분 target_len만 사용, "start"면 뒷부분, "center"면 중앙.

    Returns:
        길이가 target_len 이하인 파형.
    """
    n_samples = len(waveform)
    if n_samples <= target_len:
        return waveform

    if mode == "start":
        return waveform[n_samples - target_len :]
    if mode == "center":
        start = (n_samples - target_len) // 2
        return waveform[start : start + target_len]
    return waveform[:target_len]  # "end"


def pad_or_trim_to_length(waveform: np.ndarray, target_len: int) -> np.ndarray:
    """짧으면 패딩, 길면 트림해서 정확히 target_len 샘플 길이로 맞춘다 (배치 학습용 고정 길이 변환)."""
    if len(waveform) < target_len:
        return pad_to_length(waveform, target_len)
    return trim_to_length(waveform, target_len)


def sliding_window(
    waveform: np.ndarray,
    sr: int,
    window_ms: float,
    stride_ms: float,
    drop_last: bool = False,
) -> List[np.ndarray]:
    """긴 오디오를 고정 길이 윈도우들로 분할한다 (Mission 1 힌트: 긴 음성은 sliding window).

    Args:
        waveform: 입력 파형.
        sr: 샘플레이트.
        window_ms: 윈도우 길이 (ms).
        stride_ms: 윈도우 간 이동 간격 (ms). window_ms보다 작으면 윈도우가 겹친다.
        drop_last: True면 window_ms보다 짧게 남는 마지막 조각을 버린다. False면 0-패딩해서 포함.

    Returns:
        각 원소가 길이 window_len(샘플)인 파형 리스트.
    """
    window_len = int(round(window_ms / 1000.0 * sr))
    stride_len = int(round(stride_ms / 1000.0 * sr))
    if window_len <= 0 or stride_len <= 0:
        raise ValueError("window_ms/stride_ms는 0보다 커야 합니다.")

    n_samples = len(waveform)
    if n_samples <= window_len:
        return [pad_to_length(waveform, window_len)]

    windows = []
    start = 0
    while start < n_samples:
        end = start + window_len
        chunk = waveform[start:end]
        if len(chunk) < window_len:
            if drop_last:
                break
            chunk = pad_to_length(chunk, window_len)
        windows.append(chunk)
        if end >= n_samples:
            break
        start += stride_len

    return windows


# ---------------------------------------------------------------------------
# Feature extraction: MFCC / Mel-spectrogram
# ---------------------------------------------------------------------------


def extract_mfcc(waveform: np.ndarray, sr: int, n_mfcc: int = 40, **kwargs) -> np.ndarray:
    """MFCC 특징을 추출한다.

    Args:
        waveform: 입력 파형.
        sr: 샘플레이트.
        n_mfcc: MFCC 계수 개수.
        **kwargs: librosa.feature.mfcc에 그대로 전달되는 추가 옵션 (n_fft, hop_length 등).

    Returns:
        shape (n_mfcc, n_frames)의 MFCC 행렬.
    """
    import librosa

    return librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, **kwargs)


def extract_melspectrogram(
    waveform: np.ndarray, sr: int, n_mels: int = 128, to_db: bool = True, **kwargs
) -> np.ndarray:
    """Mel-spectrogram을 추출한다 (ResNet/ViT 등 이미지 분류 모델에 2D 입력으로 사용 가능).

    Args:
        waveform: 입력 파형.
        sr: 샘플레이트.
        n_mels: Mel band 개수.
        to_db: True면 power_to_db로 로그 스케일 변환 (일반적으로 학습에 더 안정적).
        **kwargs: librosa.feature.melspectrogram에 그대로 전달되는 추가 옵션.

    Returns:
        shape (n_mels, n_frames)의 Mel-spectrogram 행렬.
    """
    import librosa

    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, **kwargs)
    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel


# ---------------------------------------------------------------------------
# 라벨 json 방어적 키 탐색 (실제 스키마 확정 전 작성됨)
# ---------------------------------------------------------------------------


def _find_key_recursive(obj: Any, key_candidates: Sequence[str]) -> Optional[Any]:
    """dict/list를 재귀적으로 훑어 key_candidates 중 하나와 이름이 일치하는 값을 반환한다."""
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


def _find_list_of_dicts_with_key(obj: Any, required_key: str) -> Optional[List[dict]]:
    """required_key를 가진 dict들로 구성된 리스트를 재귀적으로 찾는다.

    "utterances" 같은 리스트 키 이름 자체가 다를 경우를 대비한 최종 fallback.
    """
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        if all(required_key in x for x in obj):
            return obj
    if isinstance(obj, dict):
        for value in obj.values():
            found = _find_list_of_dicts_with_key(value, required_key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_list_of_dicts_with_key(item, required_key)
            if found is not None:
                return found
    return None


def get_utterances(label: dict) -> List[dict]:
    """라벨 json에서 발화 리스트를 가져온다. 각 원소는 최소한 "text"를 포함해야 인정한다.

    known key("utterances") 우선 탐색, 실패 시 "text" 필드를 가진 dict 리스트를 재귀 탐색한다.

    Args:
        label: json.load로 읽은 라벨 딕셔너리.

    Returns:
        발화 dict 리스트 (없으면 빈 리스트).
    """
    utterances = _find_key_recursive(label, ["utterances", "Utterances"])
    if isinstance(utterances, list) and utterances:
        return utterances

    fallback = _find_list_of_dicts_with_key(label, "text")
    return fallback or []


def get_speaker_id(utterance: dict) -> Optional[int]:
    """발화 dict에서 speaker id를 가져온다 (키 이름 대소문자 변형 방어)."""
    value = _find_key_recursive(utterance, ["speaker", "Speaker", "speakerId", "speaker_id"])
    return int(value) if value is not None else None


def get_speaker_role(speaker_id: Optional[int]) -> Optional[str]:
    """speaker id(0/1)를 역할 문자열("dispatcher"/"reporter")로 변환한다.

    config.SPEAKER_ID_TO_ROLE 매핑을 사용한다 (대회 규정: 0=119대원, 1=신고자).
    """
    if speaker_id is None:
        return None
    return config.SPEAKER_ID_TO_ROLE.get(speaker_id)


def guess_speaker_role_from_text(text: str) -> Optional[str]:
    """발화 텍스트의 키워드로 화자 역할을 추정한다 (QA/EDA 전용, 절대 모델 입력으로 쓰지 말 것).

    Mission 2는 텍스트/순서 정보를 사용할 수 없으므로, 이 함수는 라벨 일관성 검증이나
    speaker id 필드가 없는 예외적인 데이터를 EDA하는 용도로만 사용한다.

    Args:
        text: 발화 텍스트.

    Returns:
        "dispatcher" 또는 "reporter" 중 더 많은 키워드가 매칭된 쪽, 매칭이 없으면 None.
    """
    if not text:
        return None

    scores = {role: 0 for role in config.SPEAKER_ROLE_KEYWORDS}
    for role, keywords in config.SPEAKER_ROLE_KEYWORDS.items():
        scores[role] = sum(1 for kw in keywords if kw in text)

    best_role = max(scores, key=scores.get)
    return best_role if scores[best_role] > 0 else None


def extract_utterance_text(label: dict, speaker_id: Optional[int] = None) -> List[str]:
    """라벨 json에서 발화 텍스트 리스트를 추출한다 (Mission 3 텍스트 입력용).

    Args:
        label: json.load로 읽은 라벨 딕셔너리.
        speaker_id: 지정하면 해당 화자의 발화만 필터링 (None이면 전체 발화).

    Returns:
        시간순 발화 텍스트 리스트.
    """
    utterances = get_utterances(label)
    if speaker_id is not None:
        utterances = [u for u in utterances if get_speaker_id(u) == speaker_id]
    return [u.get("text", "") for u in utterances if u.get("text")]


# ---------------------------------------------------------------------------
# Mission 3 - 증상 라벨 필터링
# ---------------------------------------------------------------------------


def filter_symptom_labels(
    symptoms: Sequence[str], valid_classes: Sequence[str] = config.SYMPTOM_CLASSES
) -> List[str]:
    """symptom 리스트에서 타겟 9개 클래스에 속하지 않는 항목만 제거한다.

    대회 규정: 9개 클래스 밖의 증상이 섞여 있어도 샘플 전체를 버리지 않고,
    해당 라벨만 제외한다 (예: ['두통', '복통', '찰과상'] -> ['두통', '복통']).

    Args:
        symptoms: 라벨 json의 symptom 리스트.
        valid_classes: 유효한 클래스 목록 (기본값: config.SYMPTOM_CLASSES).

    Returns:
        valid_classes에 속하는 symptom만 남긴 리스트 (순서 유지, 중복 제거하지 않음).
    """
    valid_set = set(valid_classes)
    return [s for s in symptoms if s in valid_set]


def get_symptom_labels(label: dict) -> List[str]:
    """라벨 json에서 symptom 리스트를 가져와 9개 타겟 클래스로 필터링한다."""
    raw = _find_key_recursive(label, ["symptom", "Symptom", "symptoms"])
    if not isinstance(raw, list):
        return []
    return filter_symptom_labels(raw)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="preprocessing_utils 단독 실행: 지정한 wav/json 한 쌍으로 함수 동작을 확인"
    )
    parser.add_argument("--wav", type=str, required=True, help="테스트용 wav 파일 경로")
    parser.add_argument("--json", type=str, required=True, help="테스트용 json 라벨 경로")
    args = parser.parse_args()

    waveform, sr = load_audio(args.wav)
    print(f"loaded waveform: shape={waveform.shape}, sr={sr}")

    with open(args.json, "r", encoding="utf-8") as f:
        label = json.load(f)

    utterances = get_utterances(label)
    print(f"utterances: {len(utterances)}개")
    if utterances:
        first = utterances[0]
        seg = crop_segment(waveform, sr, first.get("startAt", 0), first.get("endAt", 0))
        print(f"  첫 발화 crop shape: {seg.shape}, speaker={get_speaker_id(first)} "
              f"role={get_speaker_role(get_speaker_id(first))}")
        mfcc = extract_mfcc(pad_to_length(seg, sr), sr)
        print(f"  MFCC shape: {mfcc.shape}")

    print(f"symptom labels (필터링 후): {get_symptom_labels(label)}")

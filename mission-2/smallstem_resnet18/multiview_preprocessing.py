"""Mission 2: 기존 1.5초 Mel과 전체 발화 요약 Mel을 나란히 배치한다.

기본 crop/Mel 추출은 github_preprocessing의 원본 함수를 그대로 사용한다.
결과는 [64 Mel bins, 48 frames]로, 앞의 24 frame은 기존 특징과 같다.
"""

from __future__ import annotations

import numpy as np

from github_preprocessing import crop_segment, extract_melspectrogram, pad_or_trim_to_length


TARGET_SECONDS = 1.5
N_MELS = 64
N_FRAMES = 24


def _normalize_mel(mel: np.ndarray) -> np.ndarray:
    return np.clip((mel.astype(np.float32) + 80.0) / 80.0, 0.0, 1.0)


def _pool_time(mel: np.ndarray, frames: int = N_FRAMES) -> np.ndarray:
    """긴 발화의 전체 Mel을 동일한 시간 구간별 평균으로 축약한다."""
    if mel.shape[1] < frames:
        raise ValueError(f"전체 Mel frame이 {frames}개보다 적습니다: {mel.shape}")
    boundaries = np.rint(np.linspace(0, mel.shape[1], frames + 1)).astype(np.int64)
    lengths = np.diff(boundaries)
    if np.any(lengths <= 0):
        raise ValueError(f"빈 시간 구간이 있습니다: {mel.shape}")
    cumulative = np.pad(np.cumsum(mel, axis=1, dtype=np.float32), ((0, 0), (1, 0)))
    return (cumulative[:, boundaries[1:]] - cumulative[:, boundaries[:-1]]) / lengths


def make_multiview_feature(waveform: np.ndarray, sr: int, start_at: float, end_at: float) -> np.ndarray:
    """동일 발화의 앞 1.5초 + 발화 전체를 결합한 float32 [64, 48] 특징."""
    segment = crop_segment(waveform, sr, start_at, end_at)
    first = pad_or_trim_to_length(segment, int(round(TARGET_SECONDS * sr)))
    first_mel = extract_melspectrogram(first, sr, n_mels=N_MELS, to_db=True)
    if first_mel.shape != (N_MELS, N_FRAMES):
        raise ValueError(f"예상하지 못한 Mel 크기: {first_mel.shape}; sr={sr}")
    first_feature = _normalize_mel(first_mel)

    if len(segment) <= len(first):
        whole_feature = first_feature
    else:
        whole_mel = extract_melspectrogram(segment, sr, n_mels=N_MELS, to_db=True)
        whole_feature = _pool_time(_normalize_mel(whole_mel))
    return np.concatenate((first_feature, whole_feature), axis=1)


def quantize_feature(feature: np.ndarray) -> np.ndarray:
    """정규화된 Mel을 uint8로 저장해 2-view를 기존 저장 크기로 유지한다."""
    return np.rint(feature * 255.0).astype(np.uint8)

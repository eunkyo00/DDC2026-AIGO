"""GitHub 공통 전처리에서 Mission 2에 필요한 함수만 옮긴 모듈.

Source:
https://github.com/eunkyo00/DDC2026-AIGO/blob/main/common/preprocessing/preprocessing_utils.py

Verified against repository commit: c83727276141377cfda7e10a39eaf61630c927a3

아래 함수 본문은 저장소의 구현과 동일하다. Mission 2 설정값(1.5초, 64 Mel bands,
log-scale)은 ``dataset.py``에서 함수 인자로 명시한다.
"""

from typing import Optional

import numpy as np
import soundfile as sf


def load_audio(path: str, target_sr: Optional[int] = None, mono: bool = True):
    """wav 파일을 읽어 (waveform, samplerate)를 반환한다."""
    waveform, sr = sf.read(path, always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)

    if mono and waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if target_sr is not None and target_sr != sr:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return waveform, sr


def crop_segment(waveform: np.ndarray, sr: int, start_ms: float, end_ms: float) -> np.ndarray:
    """startAt/endAt(ms) 구간에 해당하는 오디오 조각을 잘라낸다."""
    n_samples = len(waveform)
    start_idx = max(0, int(round(start_ms / 1000.0 * sr)))
    end_idx = min(n_samples, int(round(end_ms / 1000.0 * sr)))
    if end_idx <= start_idx:
        return waveform[0:0]
    return waveform[start_idx:end_idx]


def pad_to_length(waveform: np.ndarray, target_len: int, mode: str = "end") -> np.ndarray:
    """짧은 오디오를 0으로 패딩해 target_len 샘플 길이로 맞춘다."""
    n_samples = len(waveform)
    if n_samples >= target_len:
        return waveform

    pad_total = target_len - n_samples
    if mode == "start":
        pad_width = (pad_total, 0)
    elif mode == "center":
        pad_left = pad_total // 2
        pad_width = (pad_left, pad_total - pad_left)
    else:
        pad_width = (0, pad_total)

    return np.pad(waveform, pad_width, mode="constant")


def trim_to_length(waveform: np.ndarray, target_len: int, mode: str = "end") -> np.ndarray:
    """긴 오디오를 target_len 샘플로 잘라낸다."""
    n_samples = len(waveform)
    if n_samples <= target_len:
        return waveform

    if mode == "start":
        return waveform[n_samples - target_len :]
    if mode == "center":
        start = (n_samples - target_len) // 2
        return waveform[start : start + target_len]
    return waveform[:target_len]


def pad_or_trim_to_length(waveform: np.ndarray, target_len: int) -> np.ndarray:
    """짧으면 패딩, 길면 트림해서 정확히 target_len 샘플 길이로 맞춘다."""
    if len(waveform) < target_len:
        return pad_to_length(waveform, target_len)
    return trim_to_length(waveform, target_len)


def extract_melspectrogram(
    waveform: np.ndarray, sr: int, n_mels: int = 128, to_db: bool = True, **kwargs
) -> np.ndarray:
    """Mel-spectrogram을 추출한다."""
    import librosa

    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, **kwargs)
    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel

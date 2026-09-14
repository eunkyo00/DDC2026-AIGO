# STEP 4-1 — F0 + Acoustic Logistic Regression

## 실행 범위와 고정 split

기존 `split_assignments.csv`(SHA256 `04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06`)를 그대로 사용했다. Train 22,388 calls, Internal Validation 5,597 calls이며 seed는 42이다. 새 split, threshold tuning, MFCC/SVM/pretrained 모델 실험은 수행하지 않았다.

| Partition | Male | Female | Total |
|---|---:|---:|---:|
| Train | 10,480 | 11,908 | 22,388 |
| Internal Validation | 2,620 | 2,977 | 5,597 |

신고자 역할인 `speaker=1` segment만 잘랐다. 다만 EDA에서 약 96%의 통화에 신고자/상담원 annotation overlap이 있었으므로, 신고자 segment를 자른다고 해서 상담원 음성이 완전히 제거된다고 볼 수 없다.

## 특징과 집계

각 segment에서 filtered F0 mean/median/std, voiced ratio, RMS, duration, zero-crossing rate, spectral centroid를 계산했다. 통화별로 각 값의 mean과 median을 취하고 segment 수, segment 길이 합, 신뢰 F0 segment 비율을 더했다. 이것이 유일한 aggregation이다.

Pitch extractor는 Praat raw autocorrelation(Parselmouth)이며 time step 0.01s, 원시 탐색 50–800Hz, voicing threshold 0.45이다. 원시 결과는 품질 감사에만 쓰고, 모델 F0는 60–400Hz 프레임만 남겼다. 최소 3 프레임, voiced ratio 0.10 이상이며 50–1000Hz 스펙트럼 단일-bin 전력 집중도가 0.90 미만인 segment만 F0 신뢰 구간으로 처리했다. 이 규칙은 validation 결과를 보기 전에 고정했으며 사람 음성을 완벽히 판별하는 VAD나 tone detector가 아니다.

660Hz tone이나 비명·울음·잡음은 pitch 후보 또는 octave error를 만들 수 있다. 따라서 filtered F0도 생리적 사람 pitch의 확정값이 아니며, RMS/ZCR/centroid에도 상대 화자와 비음성 신호가 영향을 줄 수 있다.

Praat 설정 근거: [raw autocorrelation 설정](https://fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html), [pitch analysis method 선택](https://uvafon.hum.uva.nl/praat/manual/how_to_choose_a_pitch_analysis_method.html), [Parselmouth `to_pitch_ac` API](https://parselmouth.readthedocs.io/en/stable/api_reference.html). Praat의 일반 기본값을 그대로 사람 성별 경계로 사용하지 않고, 전화 음성·고음 감정 발성·660Hz tone 감사를 위해 원시 ceiling을 800Hz로 넓힌 뒤 모델 입력 범위를 별도로 필터링했다.

## F0 품질

| 항목 | segment 수 | 비율 |
|---|---:|---:|
| 원시 F0 추출 성공 | 442,616 | 99.995% |
| 원시 F0 추출 실패 | 23 | 0.005% |
| baseline 신뢰 F0 | 378,941 | 85.609% |
| baseline F0 제외/불신뢰 | 63,698 | 14.391% |
| 낮은 voiced ratio | 63,685 | 14.388% |
| tone concentration flag | 0 | 0.000% |

원시 voiced frame 중 <60Hz 비율은 0.129%, >400Hz 비율은 20.222%이다.

사전 고정한 단일-bin concentration 0.90 기준에서는 tone flag가 검출되지 않았다. 이는 tone이 없다는 증거가 아니라, 긴 segment의 전체 FFT에 적용한 보수적 대용치가 기존 EDA의 tone 사례를 전수 식별하지 못했다는 뜻이다. 고주파 F0 frame 비율과 60–400Hz 필터가 이번 baseline의 주된 오염 방어이며 별도 tone detector로 간주하지 않는다.

## 모델과 전처리

NaN/inf는 Train에서 계산한 feature별 중앙값으로 대체했다. `StandardScaler`도 Train 22,388개에만 fit했고 Validation에는 transform만 적용했다. LogisticRegression은 solver=lbfgs, max_iter=1000, random_state=42, class_weight=None이며 tuning이나 decision threshold 변경은 없다.

## 고정 Internal Validation 결과

| Model | Validation Unit | Accuracy |
|---|---|---:|
| Majority Female | Call | 53.189209% |
| F0 + Acoustic + Logistic Regression | Call | 91.209577% |

절대 향상폭은 +38.020368 percentage points이다. Male accuracy 89.961832%, Female accuracy 92.307692%. 예측 수는 M 2,586, F 3,011이다.

Confusion matrix의 행은 실제 [M, F], 열은 예측 [M, F]이다.

| | Pred M | Pred F |
|---|---:|---:|
| True M | 2,357 | 263 |
| True F | 229 | 2,748 |

### 신고자 segment 총 길이별

기존 EDA의 call duration 범위를 보고 모델 평가 전에 `<8, 8–16, 16–32, 32–64, >=64초`로 고정했다.

| 구간 | n | Accuracy |
|---|---:|---:|
| <8s | 23 | 91.304% |
| 8–16s | 328 | 90.854% |
| 16–32s | 2,754 | 91.649% |
| 32–64s | 2,231 | 91.080% |
| >=64s | 261 | 88.123% |

### F0 품질별

| 그룹 | n | Accuracy |
|---|---:|---:|
| no_reliable_f0 | 0 | N/A |
| some_reliable_f0 | 5,597 | 91.210% |
| low_reliable_ratio_lt_0.5 | 2 | 100.000% |
| reliable_ratio_ge_0.5 | 5,595 | 91.206% |
| has_low_voiced_segment | 4,687 | 91.274% |
| no_low_voiced_segment | 910 | 90.879% |
| tone_flag_present | 0 | N/A |
| no_tone_flag | 5,597 | 91.210% |

품질 그룹은 원인 규명용 기술 통계이며 서로 겹칠 수 있다. 낮은 voiced ratio 또는 F0 실패군에서도 duration/RMS/ZCR/centroid 특징과 Train 중앙값 대체가 남아 있어 순수 F0-only 성능으로 해석하면 안 된다.

## 재현성과 한계

실행 명령: `python run_f0_baseline.py --workers 6`

라이브러리: python=3.12.2, numpy=2.2.2, pandas=2.3.0, scikit-learn=1.9.1, soundfile=0.13.1, praat-parselmouth=0.4.7

`call_features.csv`는 모델에 사용한 통화 특징과 품질 집계를, `val_predictions.csv`는 고정 Validation 예측을 담는다. 실행기는 기존 결과 파일이 있으면 중단하므로 재실행 시 별도 `--out-dir`을 지정해야 한다.

동일 사람의 재신고를 연결하는 식별자가 없어 speaker-level leakage는 확인할 수 없다. annotation overlap, tone/non-speech 혼입, octave error, 감정·긴장·울음에 따른 pitch 변화는 남은 위험이다. 이번 결과는 해석 가능한 첫 acoustic baseline이며 F0만으로 성별이 충분하다는 결론을 뜻하지 않는다.

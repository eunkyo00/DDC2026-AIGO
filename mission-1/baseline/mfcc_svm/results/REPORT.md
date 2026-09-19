# STEP 4-2 — MFCC + SVM baseline

## 고정 split

브랜치 `mission1/eunkyo`, seed 42. 기존 split SHA256 `04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06`를 그대로 사용했다. F0 `call_features.csv`의 전체 call_id/gender/partition과도 일대일 동일성을 검사했다. 새 split이나 공식 Validation은 사용하지 않았다.

| Partition | Male | Female | Total |
|---|---:|---:|---:|
| Train | 10,480 | 11,908 | 22,388 |
| Internal Validation | 2,620 | 2,977 | 5,597 |

`speaker=1` 신고자 segment만 사용했다. EDA에서 약 96% 통화에 신고자/상담원 annotation overlap이 있었으므로, 신고자 segment를 추출한다고 해서 상담원 음성이 완전히 제거되는 것은 아니다.

## MFCC와 aggregation

8kHz mono 입력에서 n_mfcc=13, n_mels=26, n_fft=256(32ms), win_length=200(25ms), hop_length=80(10ms), 50–3800Hz, Hamming window, pre-emphasis=0.97를 사용했다. Slaney mel filter area normalization과 orthonormal DCT-II를 사용했고 waveform normalization과 CMVN은 적용하지 않았다. 각 coefficient sequence의 mean/std를 segment당 26차원으로 만든 뒤, 통화 안에서 각 값의 mean/std를 취하고 segment 수와 총 길이를 더해 최종 54차원으로 고정했다.

구현은 [librosa MFCC 문서](https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html)의 mel-spectrum→DCT 구성을 명시적으로 사용한다. 프레임/FFT 설정은 8kHz 전화 음성의 시간 해상도와 주파수 해상도를 함께 유지하는 작은 baseline 설정이다.

## 특징 품질

| 항목 | 개수 | 비율 |
|---|---:|---:|
| MFCC 성공 call | 27,985 | 100.000000% |
| MFCC 실패 call | 0 | 0.000000% |
| 성공 segment | 442,639 | 100.000000% |
| 실패 segment | 0 | 0.000000% |
| 25ms 미만 segment | 1 | 0.000226% |
| silence ratio >=50% segment | 742 | 0.167631% |

모델 특징의 추출 후 NaN/inf는 Train 0/0, Validation 0/0개다. 25ms 미만 구간은 zero-padding한 한 frame으로 유지하고 품질 플래그를 남겼다. 실패 segment를 조용히 제거하지 않으며, 일부 실패면 성공 segment로 집계하고 전부 실패한 call은 Train 중앙값 대체 후 cohort에 유지하도록 구현했다. 무음 대용치는 frame RMS <0.001이며 VAD가 아니므로 저에너지 발화와 잡음을 구분하지 못한다. 무음이 많은 segment의 MFCC는 배경/채널 특성을 더 반영할 수 있다.

## RBF SVC 자원 판단

scikit-learn은 SVC fit 시간이 표본 수에 대해 적어도 이차적으로 증가할 수 있다고 경고한다. 22,388² dense kernel은 약 3.73GiB지만 libsvm이 이를 전부 상주시킬 필요는 없고 cache를 512MB로 제한했다. Train 앞 5,000개 자원 측정은 0.64초, 보수적 이차 투영은 12.83초였다. 따라서 대안으로 바꾸지 않고 사전 지정 RBF SVC 한 개를 실행했으며 실제 full fit은 7.97초였다. 설정은 C=1.0, gamma=scale, probability=False, class_weight=None이며 tuning하지 않았다. probability calibration을 사용하지 않아 SVC fit/predict에는 stochastic random_state가 필요하지 않으며, seed 42는 frozen split과 재현성 표기에 사용했다. [SVC 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

NaN/inf 대체 중앙값과 StandardScaler는 Train 22,388개에만 fit했고 Validation에는 transform만 적용했다.

## 고정 Internal Validation 결과

| Model | Validation Unit | Accuracy |
|---|---|---:|
| Majority Female | Call | 53.189209% |
| F0 + Acoustic + Logistic Regression | Call | 91.209577% |
| MFCC + RBF SVM | Call | 95.068787% |

MFCC overall은 F0보다 +3.859210pp, Male은 +4.847328pp, Female은 +2.989587pp다. Validation 5,597개 중 두 모델 모두 정답 4,967, MFCC만 정답 354, F0만 정답 138, 둘 다 오답 138이다.

Male accuracy 94.809160%, Female accuracy 95.297279%. 예측 수 M 2,624, F 2,973.

| | Pred M | Pred F |
|---|---:|---:|
| True M | 2,484 | 136 |
| True F | 140 | 2,837 |

### 총 신고자 발화 시간별

| 구간 | n | Accuracy |
|---|---:|---:|
| <8s | 23 | 82.609% |
| 8–16s | 328 | 93.598% |
| 16–32s | 2,754 | 95.171% |
| 32–64s | 2,231 | 95.473% |
| >=64s | 261 | 93.487% |

### 신고자 segment 수별

| 구간 | n | Accuracy |
|---|---:|---:|
| 1–5 | 41 | 85.366% |
| 6–10 | 969 | 94.427% |
| 11–20 | 3,558 | 95.728% |
| 21–40 | 1,018 | 93.811% |
| >=41 | 11 | 90.909% |

### MFCC 품질별

| 그룹 | n | Accuracy |
|---|---:|---:|
| mfcc_success | 5,597 | 95.069% |
| mfcc_failure | 0 | N/A |
| has_failed_segment | 0 | N/A |
| all_segments_succeeded | 5,597 | 95.069% |
| has_short_segment_lt_25ms | 0 | N/A |
| no_short_segment_lt_25ms | 5,597 | 95.069% |
| has_high_silence_segment | 122 | 95.902% |
| no_high_silence_segment | 5,475 | 95.050% |

## 해석

1. MFCC baseline은 F0 baseline보다 높다: 차이는 +3.859210pp다.
2. 전체 +3.859pp는 이 fixed split의 baseline 비교에서는 의미 있는 차이다. 두 모델이 서로 다르게 맞힌 통화가 492개라 오류가 완전히 같지는 않지만 hybrid/ensemble 실험은 수행하지 않았다. 공식 Validation이나 독립 화자 기준에서도 같은 차이가 유지되는지는 확인하지 않았다.
3. F0 대비 변화의 절댓값은 Male 쪽이 더 크다. 성별별 차이는 Male +4.847328pp, Female +2.989587pp다.
4. <8초 통화는 n=23로 작다. MFCC 82.609%, F0 91.304%, 차이 -8.696pp이므로 이 표본만으로 더 안정적이라고 일반화할 수 없다.
5. MFCC가 낮거나 특정 그룹에서 약하면 segment 통계화로 시간 순서를 잃는 점, 긴 무음/잡음과 채널 특성, operator overlap, 울음·비명·긴장 발성, 짧은 구간 zero-padding이 가능한 원인이다.
6. Frozen SSL은 MFCC/F0가 놓치는 시간 구조와 음성 표현을 시험할 가치가 있다. 다만 같은 fixed split과 caller-only 구간을 유지하고, 현재 known risks를 해결된 것으로 간주하면 안 된다.

동일 사람 반복 등 speaker-level leakage는 현재 메타데이터로 확인 불가하다. 이 결과는 현재 Internal Validation에 한정되며 공식 Validation 성능이나 독립 화자 일반화를 보장하지 않는다.

## 재현성

실행 명령: `python mission-1/baseline/mfcc_svm/run_mfcc_baseline.py --workers 6`

전체 실행 시간은 1994.82초이며 worker 6개를 사용했다.

라이브러리: python=3.12.2, numpy=2.2.2, pandas=2.3.0, librosa=0.11.0, scipy=1.18.1, soundfile=0.13.1, scikit-learn=1.9.1

기본 `results/`가 이미 존재하면 실행을 중단한다. 재실행은 새 `--out-dir`을 지정해야 하며, `--limit`은 feature smoke test만 수행하고 모델을 학습하지 않는다.

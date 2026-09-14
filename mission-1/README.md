# Mission 1 — 신고자 음성 성별 분류

## 1. Mission 1 한눈에 보기

**목표는 신고자 음성을 `male` / `female`로 분류하는 것**이다. 입력은 WAV + JSON, 평가는 **Accuracy(통화 중 정답을 맞힌 비율)**, 최종 출력은 **Call-level 성별 예측 CSV**다. call은 통화 1건, segment는 통화 안의 발화 구간이다.

```text
WAV + JSON → 신고자 발화 추출 → 음성 특징 / 음성 모델
    → 발화 예측·정보 통합 또는 Call 단위 예측 → Call-level male / female → CSV
```

- **WAV**: 실제 통화 음성.
- **JSON**: 신고자가 언제 말했는지 찾는 정보. `speaker=1`은 신고자, `speaker=0`은 상담원이다.
- **`gender`**: 학습·평가용 정답(`M`/`F`)이며 모델 입력으로 사용하지 않는다.

## 2. 데이터와 전처리

### 데이터 구조

| 항목 | Training | 공식 Validation |
|---|---:|---:|
| WAV | 27,987 | 3,640 |
| JSON | 29,200 | 3,640 |
| 정상 JSON + WAV 매칭 Calls | **27,985** | **3,640** |
| Male (정상 매칭 기준) | 13,100 (46.81%) | 1,681 (46.18%) |
| Female (정상 매칭 기준) | 14,885 (53.19%) | 1,959 (53.82%) |

확인한 WAV는 모두 **8kHz(초당 8,000샘플), mono(1채널), PCM 16-bit**다. Training의 WAV 누락 **1,213건**, JSON 읽기/구조 실패 **2건**은 제외했으며 공식 Validation에서는 두 항목 모두 0건이다. JSON 수를 학습 가능한 통화 수로 사용하지 않는다.

### 신고자 음성 추출

WAV와 JSON을 파일명으로 연결하고, JSON의 `utterances[]`에서 신고자 구간을 골라 음성을 자른다. 발화별 `startAt`·`endAt`은 밀리초 단위이므로 초로 변환한다.

```text
전체 WAV + JSON의 startAt / endAt / speaker
    → speaker=1인 신고자 발화 구간 추출
```

### EDA에서 확인한 것

EDA(데이터 구조·품질 탐색)는 성별 분포, 발화 수·길이, sample rate/channel, 기본 음향 품질, 신고자·상담원 구간 겹침, F0 추정 이상을 살폈다. 파일·라벨·시간 경계·WAV 헤더는 전수 확인하고 음향은 **80통화 표본**을 분석했다. 세부 결과와 검사 한계는 [eda/REPORT.md](eda/REPORT.md)에 있다.

## 3. 전체 실험 전략

```text
데이터 구조 확인 / EDA → Call-level Validation 고정
        ↓
Majority
        ↓
F0 + Acoustic → Logistic Regression
        ↓
MFCC → RBF SVM
        ↓
Wav2Vec2 / HuBERT
        ↓
Error Analysis → 여러 신고자 발화 Aggregation
        ↓
필요한 성능 개선 → Final Inference Pipeline
```

**왜 이 순서인가?** 처음부터 복잡한 모델을 쓰기보다 단순한 방법으로 어디까지 가능한지 확인한다. 더 많은 음성 정보를 사용하는 방법을 차례로 비교해 **어떤 정보와 모델이 성능 향상에 기여하는지 같은 평가 조건에서 근거를 남긴다.** 오류 분석은 baseline부터 시작해 모델 비교 후에도 반복하며, 아래의 비교 설계와 실제 측정 결과는 구분한다.

### Call-level Validation

한 통화의 여러 발화를 양쪽에 나누면 같은 목소리·녹음 환경을 학습과 평가에서 함께 접하는 **leakage(데이터 누수)**가 생길 수 있다. 이를 막기 위해 통화 전체를 한쪽에 배정했다.

```text
X  Call A segment 1 → Train
   Call A segment 2 → Validation

O  Call A 전체 → Train
   Call B 전체 → Validation
```

- **정상 Training 27,985 calls → Train 22,388 / Internal Validation 5,597**.
- **seed 42**, **gender-stratified**: 남녀 각각 80:20으로 나눠 성별 비율을 유지한다.
- **모든 모델은 동일한 [고정 split](validation/split_assignments.csv)을 사용**한다. 전처리에 필요한 값도 Train에서만 학습한다.

공식 Validation **3,640건**은 Training에서 나눈 Internal Validation과 별개이며, 이 split 생성에 사용하지 않았다. 배정·검증의 세부 내용은 [validation/REPORT.md](validation/REPORT.md)를 따른다.

## 4. 모델별 실험 방법과 결과

| 방법 | 음성에서 사용하는 정보 | 분류 방법 | 목적 |
|---|---|---|---|
| Majority | 없음 | Female 고정 | 최소 기준 성능 |
| F0 + Acoustic | 높낮이 + 기본 음향 특징 | Logistic Regression | 단순한 음향 정보의 효과 확인 |
| MFCC | 음색 + 주파수 구조 | RBF SVM | 더 넓은 전통적 음성 특징의 추가 효과 확인 |
| Wav2Vec2 / HuBERT | 사전학습된 음성 표현 | Classifier(분류기) | 사전학습 표현의 추가 효과 확인 |

**확인된 실험 결과**는 먼저 아래 표에서 비교할 수 있다. 모두 **같은 Internal Validation 5,597통화의 call 단위 Accuracy**이며, 검증된 수치가 있는 방법만 기록한다.

| Method | Classifier | Accuracy |
|---|---|---:|
| Majority | Female 고정 | 53.189209% |
| F0 + Acoustic | Logistic Regression | 91.209577% |
| MFCC | RBF SVM | **95.068787%** |

동일한 고정 Internal Validation에서 MFCC + RBF SVM이 F0 + Acoustic + Logistic Regression보다 **3.859210%p 높은 Accuracy**를 기록했다. 비교한 baseline 중 가장 높은 결과이며, 사전학습 모델과 후속 비교를 위한 기준으로 삼는다.

### Majority

Train에서 더 많은 성별인 Female로 모든 call을 예측한다. 음성이나 모델 학습 없이 얻는 값으로, 이후 모델의 최소 비교 기준이다.

### F0 + Acoustic + Logistic Regression

```text
신고자 음성 → F0 + Acoustic Features → Call-level 집계
    → Logistic Regression → male / female
```

- **F0**: 목소리 높낮이와 관련된 기본 주파수.
- **Acoustic Features**: RMS(신호 크기), voiced ratio(유성음 판정 비율), zero-crossing rate(0 교차 빈도), spectral centroid(주파수 에너지 중심), duration(발화 길이).
- **Logistic Regression**: 숫자 특징으로 남/여를 분류하는 간단한 모델.

**단순한 음향 정보만으로 얼마나 구분할 수 있는지 확인하는 baseline(비교 기준)**이다. 발화 특징을 통화별 평균·중앙값으로 합치고 발화 수·길이 합 등도 사용했다.

| Metric | Result |
|---|---:|
| Male Accuracy | 89.961832% |
| Female Accuracy | 92.307692% |
| Majority 대비 | +38.020368%p |

Male/Female Accuracy는 각 실제 성별 통화 중 정답 비율이며, %p는 퍼센트포인트다. **이 결과는 F0만이 아니라 F0 + Acoustic Features + Call-level aggregation의 결과**다.

### MFCC + RBF SVM

**MFCC**는 음색과 주파수 에너지 구조를 숫자로 표현하는 전통적인 음성 특징이다. **RBF SVM**은 특징 간 비선형적인 경계도 학습하는 분류기다. F0·기본 음향 특징보다 더 넓은 음색·주파수 정보가 추가 개선을 주는지 확인하기 위해 사용했다.

```text
신고자 음성 → MFCC 추출 → Segment-level mean/std 요약
    → Call-level 집계 → RBF SVM → male / female
```

각 segment에서 MFCC 계수별 mean/std(평균·표준편차)를 계산한 뒤, 통화 안에서 다시 mean/std로 집계하고 segment 수와 총 발화시간을 더했다.

| 핵심 설정 | 값 |
|---|---|
| MFCC | 13개 계수, 26개 mel bands, 50–3800Hz |
| 분석 구간 | 25ms Hamming window, 10ms hop, FFT 256 |
| 전처리·특징 | pre-emphasis 0.97, 최종 call-level **54차원** |
| RBF SVM | `C=1`, `gamma="scale"`, `class_weight=None`, `probability=False` |

동일 fixed split을 사용했으며 StandardScaler는 Train 22,388통화에만 fit했다. 특징 추출은 27,985통화·442,639 segment 모두 성공했고 NaN/inf는 0건이었다.

| 성별별 결과 | Accuracy | F0 대비 |
|---|---:|---:|
| Male | 94.809160% | +4.847328%p |
| Female | 95.297279% | +2.989587%p |

기본 음향 특징을 쓴 F0 baseline도 높은 정확도를 보였고, MFCC + RBF SVM은 이를 더 높였다. 이는 **음색과 더 넓은 주파수 구조가 추가 정보를 제공할 가능성**을 보여주며 남성 쪽 개선폭이 더 컸다. 다만 **특징과 분류기를 동시에 변경했으므로 향상 전체가 MFCC 자체의 효과라고 단정할 수 없다.** 혼동행렬·길이별 결과·재현 설정은 [MFCC REPORT](baseline/mfcc_svm/results/REPORT.md)에 있다.

### Wav2Vec2 / HuBERT

```text
Audio → Frozen Pretrained Encoder → Speech Embedding → Classifier → male / female
```

**pretrained**는 대규모 음성으로 미리 학습했다는 뜻이며, **frozen**은 가중치 고정, **embedding**은 모델이 음성을 표현한 숫자 벡터다. Frozen 비교의 질문은 **“대규모 음성에서 배운 표현이 이미 강한 MFCC baseline을 넘어서는 추가 정보를 제공하는가?”**다. 모델의 복잡성보다 실제 추가 가치를 검증하고, 뚜렷한 이점이 있을 때 fine-tuning(모델 추가 학습)을 검토한다.

## 5. 결과 이후 분석과 개선

### Error Analysis

Accuracy만 보고 끝내지 않고 어떤 통화에서 틀리는지 살핀다. 확인 대상은 **짧은 발화, silence(무음) / clipping(진폭 한계 왜곡), 불안정한 F0, 신고자·상담원 겹침, 남녀별 오류 차이와 대표 오분류 음성**이다. 목적은 **모델의 한계인지, 데이터 품질인지, 발화 활용 방식의 문제인지 구분**하는 것이다.

### 여러 발화 활용

한 통화에는 신고자 발화가 여러 개 있다. 한 개의 짧은 발화에 의존하기보다 여러 정보를 합쳐 안정적인 통화별 결과를 만드는 방법을 비교한다.

```text
segment 1 ─┐
segment 2 ─┼→ Call-level aggregation(통합) → 최종 성별
segment 3 ─┘
```

F0·MFCC 실험의 **특징 집계**와 별도로, 발화별 **예측 결과 통합**을 비교한다. 후보는 probability mean(확률 평균), majority voting(다수결), length weighted(길이 가중), confidence weighted(확신도 가중)다.

### 필요한 경우 성능 개선

후보는 partial fine-tuning(일부 층 추가 학습), F0/acoustic + SSL embedding hybrid(자기지도학습 음성 표현 결합), 약한 augmentation(음성 변형), threshold tuning(판정 경계 조정), ensemble(모델 결과 결합)이다. **모두 적용하지 않고 오류 분석과 모델 비교 결과에 근거해 선택한다.** 강한 높낮이 변형은 성별 단서를 바꿀 수 있어 조심한다.

## 6. 최종 Inference Pipeline

```text
WAV + JSON
    ↓
JSON parsing → 신고자 segment 추출
    ↓
audio preprocessing → 모델 추론
    ↓
여러 발화 결과 통합
    ↓
Call-level male / female → CSV
```

최종 목표는 `inference.py` 같은 **하나의 실행 진입점에서 입력부터 CSV 생성까지 자동화**하는 것이다. 파일명은 설계 예시이며, 특징을 먼저 집계하는 모델은 해당 순서를 모델 구조에 맞춘다. JSON 읽기, 음성 전처리, 예측은 학습 때 정한 방식과 일치시킨다.

## 7. 주요 데이터 리스크 및 상세 문서

다음은 **성능 해석에 중요한 known risks(알려진 위험)**다. MFCC의 Accuracy가 높아져도 이 위험들이 해결된 것은 아니다.

| 리스크 | 확인한 사실과 의미 |
|---|---|
| Caller / Operator overlap | 정상 파싱 통화의 **96.41%**에서 신고자·상담원 라벨 시간 구간이 겹쳤다. 신고자 segment에 상담원 음성이 섞일 수 있지만, 라벨 겹침만으로 실제 동시 발성을 확정하지 않는다. |
| F0 / Tone | **약 660Hz 톤이 F0 약 329Hz로 추정**된 진단 사례가 있다. 추정 F0를 항상 사람 목소리의 실제 높낮이로 볼 수 없으며, 이 사례를 전체 톤 비율로 일반화하지 않는다. |
| Speaker-level leakage | 동일 신고자를 연결할 person ID / phone hash가 없어 같은 사람이 Train과 Validation에 포함됐는지 확인할 수 없다. 통화 단위 분리가 사람 단위 분리까지 보장하지 않는다. |

세부 설정, 재현 명령과 실험 로그는 아래 문서에서 확인한다.

| 내용 | 문서 |
|---|---|
| 데이터 / EDA | [README](eda/README.md), [REPORT](eda/REPORT.md) |
| Validation | [README](validation/README.md), [REPORT](validation/REPORT.md) |
| Validation sanity | [REPORT](validation/sanity/results/REPORT.md) |
| F0 baseline | [README](baseline/f0_lr/README.md), [REPORT](baseline/f0_lr/results/REPORT.md) |
| MFCC baseline | [README](baseline/mfcc_svm/README.md), [REPORT](baseline/mfcc_svm/results/REPORT.md) |

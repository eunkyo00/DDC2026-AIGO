# Mission 1 · 신고자 음성 성별 분류

통화의 신고자 음성을 사용해 **통화별 Male / Female을 예측**한다.
입력은 WAV와 발화 구간이 기록된 JSON이며, 평가는 통화 단위 Accuracy로 비교한다.

> **현재 상태 · 2026-09-20**
>
> Frozen Wav2Vec2 baseline과 기본 오류 비교까지 완료했다.
>
> 최고 Accuracy는 **97.248526%**다. **99% 목표 추가 실험은 계획 단계**이며 아직 실행하지 않았다.

[실험 결과](#실험-결과) · [진행 상황](#진행-상황) · [데이터와-평가-기준](#데이터와-평가-기준) · [폴더와-실행-안내](#폴더와-실행-안내)

**다음 실험:** 기존 embedding으로 LR 조정·MFCC 특징 결합·RBF SVM을 비교하고,
Train 교차검증 오답 분석 후 필요하면 새 모델을 검토한다.
→ [99% 목표 실험 계획: 이유·설정·선택 기준·진행 순서](EXPERIMENT_PLAN_99.md)

## 실험 결과

모든 결과는 **같은 고정 Internal Validation 5,597통화**에서 측정했다.
Male / Female Accuracy는 각 실제 성별에 속한 통화의 정답 비율이다.

| 모델 | Overall | Male | Female |
|---|---:|---:|---:|
| Majority · Female 고정 | 53.189209% | 0.000000% | 100.000000% |
| F0 + Acoustic → Logistic Regression | 91.209577% | 89.961832% | 92.307692% |
| MFCC → RBF SVM | 95.068787% | 94.809160% | 95.297279% |
| **Frozen Wav2Vec2 → Logistic Regression** | **97.248526%** | **96.755725%** | **97.682230%** |

Wav2Vec2는 MFCC보다 **2.179739%p** 높았으며, 오답은 **276개 → 154개**로 줄었다.
전체 L4 embedding 추출·검사는 **11.19시간**, 로컬 LogisticRegression 학습은 약 **3초** 걸렸다.
추출 시간은 Drive 읽기·저장·검사를 포함하며 새 통화 한 건의 추론 지연시간을 뜻하지 않는다.

### MFCC와 Wav2Vec2의 오류 비교

| 결과 | 통화 수 |
|---|---:|
| 둘 다 맞음 | 5,261 |
| Wav2Vec2만 맞음 | 182 |
| MFCC만 맞음 | 60 |
| 둘 다 틀림 | 94 |

성별별 평가, confusion matrix와 call별 오류 겹침을 확인했다.
개별 오답을 듣고 원인을 분류하는 분석은 아직 진행하지 않았다.
→ [최종 REPORT](ssl/wav2vec2_frozen/results/REPORT.md) · [Metrics](ssl/wav2vec2_frozen/results/full_l4/metrics.json) · [Validation 예측](ssl/wav2vec2_frozen/results/full_l4/val_predictions.csv)

## 진행 상황

단순한 음향 특징부터 시작해 사전학습 표현이 추가로 도움이 되는지 비교했다.

```text
데이터 구조 확인 / EDA → Call-level Validation 고정
        ↓
Majority → F0 + Acoustic + LR → MFCC + RBF SVM
        ↓
Frozen Wav2Vec2 + LR → 성능·성별별 평가·오류 겹침 분석
        ↓
기존 특징 활용 개선 + Train OOF 오답 분석
        ↓
필요 시 새 모델·집계 개선 → 최종 평가 → 통합 추론
```

| 상태 | 범위 |
|---|---|
| 완료 | EDA, fixed split, 네 baseline 평가, 성별별 평가·confusion matrix·오류 겹침 |
| 다음 단계 · 계획 | 기존 embedding 기반 LR 조정·MFCC 결합·RBF SVM, Train OOF 오답 분석 |
| 조건부 후속 후보 | Frozen ECAPA, 부분 fine-tuning 또는 aggregation 개선 |
| 미구현 | 새 WAV/JSON 입력부터 최종 예측 CSV까지 연결하는 통합 추론 진입점 |

여러 발화의 **특징 집계(aggregation)는 각 baseline에 이미 적용**했다.
다른 집계 방식의 비교, HuBERT, fine-tuning, augmentation, ensemble, hybrid,
threshold tuning, hyperparameter search는 **완료된 Wav2Vec2 baseline**에 포함하지 않았다.
특징 결합과 제한된 파라미터 비교는 별도 후속 실험으로 계획했으며 기존 결과와 구분한다.
별도 경량 모델 작업도 위 완료 결과에 포함하지 않는다.

## 데이터와 평가 기준

### 사용 데이터

| 항목 | Training 원본 | 공식 Validation 원본 |
|---|---:|---:|
| WAV | 27,987 | 3,640 |
| JSON | 29,200 | 3,640 |
| 정상 JSON + WAV 매칭 통화 | **27,985** | **3,640** |
| Male / Female | 13,100 / 14,885 | 1,681 / 1,959 |

확인한 WAV는 **8kHz · mono · PCM 16-bit**다. Training에서 WAV 누락 1,213건과
JSON 읽기·구조 실패 2건을 제외했다. 공식 Validation에는 이 두 오류가 없었다.

WAV와 JSON을 연결한 뒤 `utterances[]`의 `speaker=1` 구간을 사용한다.
`speaker=0`은 상담원이며 `startAt`·`endAt`은 밀리초에서 초로 변환한다.
정답 `gender`는 학습·평가에만 사용한다.

### 고정 평가 분할

```text
정상 Training 27,985통화 · caller 발화 442,639개
        ├─ Train                 22,388통화
        └─ Internal Validation    5,597통화
```

- seed 42, 남녀 비율을 유지하는 80:20 분할
- 같은 통화의 모든 발화는 같은 partition에 배정
- 모든 모델이 동일한 [split_assignments.csv](validation/split_assignments.csv) 사용
- StandardScaler 등 학습이 필요한 전처리는 Train에만 fit

**공식 Validation 3,640통화는 위 Internal Validation과 별개**다.
이 README의 성능표는 공식 Validation 평가 결과가 아니다.
→ [분할 검증 REPORT](validation/REPORT.md)

## 모델별 방법

| 모델 | 음성 표현과 통화별 집계 | 선택 이유 |
|---|---|---|
| Majority | Train의 다수 성별인 Female로 고정 | 음성 분석 없이 얻는 최소 기준 |
| F0 + Acoustic | 높낮이·음량·유성음 비율·길이 등의 평균·중앙값 → LR | 기본 음향 정보의 분류 효과 확인 |
| MFCC | segment별 계수 mean/std → call별 mean/std와 발화 수·길이, 54차원 → RBF SVM | 음색과 주파수 구조의 추가 효과 확인 |
| Frozen Wav2Vec2 | 마지막 hidden state 시간 평균 → segment 동일 가중 call 평균, 768차원 → LR | 사전학습 음성 표현의 추가 효과 확인 |

MFCC는 **13계수 · 26 mel bands · 50–3800Hz**, RBF SVM은 `C=1`, `gamma="scale"`을 사용했다.
Wav2Vec2는 고정 revision의 `facebook/wav2vec2-base`를 **FP32·Frozen**으로 사용하고
8→16kHz로 resampling했다. 전체 embedding은 동일한 L4 CUDA 환경에서 생성했으며,
검증된 embedding으로 로컬에서 Train-only StandardScaler와 기존 설정의 LR을 학습했다.

모델 간에는 표현뿐 아니라 분류기도 달라지므로 정확도 향상 전체를 특정 특징 하나의 효과로
단정하지 않는다. 파라미터·전처리·재현 명령은 아래 각 모델 문서에 기록했다.

## 해석할 때 주의할 점

| 항목 | 확인한 내용 |
|---|---|
| 사람 단위 분리 | person ID / phone hash가 없어 동일 신고자가 Train과 Validation에 있는지 확인할 수 없다. 통화 단위 분리가 사람 단위 분리를 보장하지 않는다. |
| 신고자·상담원 겹침 | 정상 파싱 통화의 96.41%에서 라벨 시간 구간이 겹쳤다. 이 사실만으로 실제 동시 발성을 확정할 수 없다. |
| F0 이상 | 약 660Hz 톤이 F0 약 329Hz로 추정된 사례가 있다. 추정 F0를 항상 사람의 실제 높낮이로 볼 수 없다. |
| EDA 범위 | 파일·라벨·시간 경계·WAV 헤더는 전수 확인했고, 음향 품질 분석은 80통화 표본을 사용했다. |

다음 상세 오류 분석에서는 Train OOF 예측을 이용해 짧은 발화, 무음·clipping,
라벨 문제와 발화 겹침을 확인한다. 새 모델·집계 개선의 방향은 이 결과에 근거해 결정한다.

## 폴더와 실행 안내

| 경로 | 역할 | 문서 |
|---|---|---|
| `EXPERIMENT_PLAN_99.md` | 99% 목표의 추가 실험 설계 · 아직 미실행 | [실험 계획](EXPERIMENT_PLAN_99.md) |
| `eda/` | 데이터 구조·품질 확인 | [README](eda/README.md) · [REPORT](eda/REPORT.md) |
| `validation/` | fixed split과 분할 검증 | [README](validation/README.md) · [REPORT](validation/REPORT.md) |
| `validation/sanity/` | 추가 분할 점검 | [REPORT](validation/sanity/results/REPORT.md) |
| `baseline/f0_lr/` | F0 + Acoustic + LR | [README](baseline/f0_lr/README.md) · [REPORT](baseline/f0_lr/results/REPORT.md) |
| `baseline/mfcc_svm/` | MFCC + RBF SVM | [README](baseline/mfcc_svm/README.md) · [REPORT](baseline/mfcc_svm/results/REPORT.md) |
| `ssl/wav2vec2_frozen/` | Frozen Wav2Vec2 구현·로컬 평가 | [README](ssl/wav2vec2_frozen/README.md) · [REPORT](ssl/wav2vec2_frozen/results/REPORT.md) |
| `ssl/wav2vec2_frozen/colab/` | CUDA 추출 notebook·runner | [Colab 실행 안내](ssl/wav2vec2_frozen/colab/README.md) |
| `ssl/wav2vec2_frozen/results/full_l4/` | 최종 L4 추출 요약·metrics·Validation 예측 | [Metrics](ssl/wav2vec2_frozen/results/full_l4/metrics.json) |

Wav2Vec2 `results/` 바로 아래의 smoke·benchmark 파일은 **초기 Mac 진단 기록**이며,
최종 결과는 **`results/full_l4/`**를 참고한다. 원본 WAV, 모델 weight, embedding cache는
GitHub에 포함하지 않는다.

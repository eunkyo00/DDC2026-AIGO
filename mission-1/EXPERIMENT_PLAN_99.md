# Mission 1 · 99% 목표 추가 실험 계획

> 업데이트: 2026-09-22 · 상태: **Train CV·오답 분석·새 모델 200통화 비교 완료, ECAPA Validation 평가 예정**
>
> 기존 최고 결과는 Internal Validation **97.248526%**다.
> 99%는 연구 목표이며, 아래 실험의 예상 성능이나 달성 보장이 아니다.

[현재 결과](README.md#실험-결과) · [기존 Wav2Vec2 REPORT](ssl/wav2vec2_frozen/results/REPORT.md)

실행 현황: [전체 비교·오답 분석](experiments/embedding_classifiers/ERROR_ANALYSIS.md) ·
[로컬 실행 안내](experiments/embedding_classifiers/README.md).
최고 CV 평균은 MFCC 결합 LR의 97.453991% (`C=0.1`)이며, 새로운 holdout 결과는 아직 없다.

## 추가 실험이 필요한 이유

현재 Frozen Wav2Vec2 + LR은 5,597통화 중 5,443개를 맞히고 154개를 틀렸다.

| 기준 | 필요한 정답 | 허용 오답 | 현재 대비 필요한 오답 순감소 |
|---|---:|---:|---:|
| 현재 97.248526% | 5,443 | 154 | — |
| 98% 이상 | 5,486 이상 | 111 이하 | 43 이상 |
| **99% 이상** | **5,542 이상** | **55 이하** | **99 이상** |

한 통화는 약 **0.017867%p**다. 정확도 소수점과 함께 정답 수, 새로 맞힌 통화와
새로 틀린 통화를 기록한다. 작은 개선도 보존하되 99%까지 남은 격차를 별도로 판단한다.

MFCC만 맞힌 통화가 60개라서 두 표현의 보완 가능성이 있다. 그러나 두 모델이
모두 틀린 통화도 94개다. 기존 두 모델 중 정답인 예측을 매번 선택하는 가상의
oracle도 약 **98.320529%**다. 이는 실제 ensemble 결과가 아니며, 특징 결합으로
새로 학습한 분류기의 상한도 아니다. 99%에는 공통 오답까지 개선할 필요가 있다.

## 진행 순서

```text
기존 결과·데이터 검증 유지
        ↓
A. 기존 embedding 활용: LR 기준 비교 → MFCC 결합 → RBF SVM
        ↓
B. Train 교차검증 오답 분석: 어떤 오류가 남는가?
        ↓
C. 성별 사전학습 ECAPA / WavLM의 Train 200통화 비교 완료
        ↓
ECAPA 설정 고정 → 고정 Validation 5,597통화 평가 → 통합 추론

D. 추가 개선이 필요할 때만 Train에서 부분 fine-tuning·집계 개선 검토
```

실행 결과 A의 24회 비교와 B의 정량 분석을 완료했다. B는 A에서 생성한
out-of-fold(OOF) 예측을 이용하며, 청취 검토는 아직 남아 있다.
C는 ECAPA 198/200, WavLM 190/200의 사용자 전달 결과를 확인했다. 추가 2,000통화 비교를 생략하고
ECAPA 설정을 고정해 Internal Validation 5,597통화를 평가하기로 했다. D는 현재 보류한다.
[예비 결과·고정 설정·다음 평가](experiments/gender_model_comparison/SCREEN_200_REPORT.md).

## 평가 규칙

| 항목 | 고정할 기준 |
|---|---|
| 외부 분할 | 기존 Train 22,388 / Internal Validation 5,597 유지 |
| Train 내부 비교 | `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` |
| 비교 공정성 | 모든 후보에 동일한 call 단위 fold 적용, fold 배정 파일 저장 |
| 전처리 | StandardScaler를 Pipeline에 넣어 각 fold의 학습 부분에만 fit |
| 선택 지표 | 3-fold 평균 Overall Accuracy; fold별·성별별 결과도 함께 공개 |
| 동률 처리 | 최저 fold Accuracy가 높은 후보, 이후 더 단순한 후보 우선 |
| 최종 학습 | 선택한 설정을 Train 22,388개 전체에 fit |
| 최종 평가 | 설정을 잠근 뒤 기존 Validation 5,597개 평가 |

각 단계의 후보군과 설정을 먼저 기록한다. Validation 점수를 보고 `C`, threshold,
특징 조합을 계속 수정하지 않는다. 단계별 Validation을 다시 평가한다면 그 횟수와
결정 경위를 기록하고, 이를 완전히 독립적인 최종 검증이라고 주장하지 않는다.
이미 관찰한 Validation 오류는 설명 자료로 사용하고, 개선 규칙은 Train OOF에서 개발한다.

공식 Validation 3,640개는 별도 데이터다. 사용 이력과 대회 규칙을 확인해 최종 확인
용도를 결정한다. Internal Validation 99%를 공식 평가 또는 새로운 통화에서의 99%와
동일시하지 않는다. 동일 신고자의 분할 간 중복 여부는 현재 식별자로 확인할 수 없다.

근거: [scikit-learn 교차검증 및 전처리 누수 방지 안내](https://scikit-learn.org/stable/modules/cross_validation.html).

## A. 기존 embedding으로 먼저 할 실험

GPU 재추출 없이 로컬에서 실행한다. LR은 기존 `lbfgs`, `max_iter=1000`,
`class_weight=None`, `random_state=42`, `tol=1e-4`를 유지한다.
수렴 경고가 있으면 해당 결과를 완료 결과로 채택하지 않고 원인을 해결한다.

| ID | 실험 | 정확한 설정 | 질문 |
|---|---|---|---|
| A1 | Wav2Vec2 + LR | 768차원 → Scaler → LR, `C={0.1, 1, 10}` | 규제 강도를 바꾸면 개선되는가? `C=1`이 기준선 |
| **A2** | **Wav2Vec2 + MFCC + LR** | **768+54=822차원** → Scaler → LR, `C={0.1, 1, 10}` | MFCC의 보완 정보를 활용할 수 있는가? |
| A3 | Wav2Vec2 + RBF SVM | 768차원 → Scaler → SVC, `C={1, 10}`, `gamma="scale"`, `class_weight=None`, `probability=False` | 비선형 결정 경계가 도움이 되는가? |

총 **8개 설정 × 3-fold = 24회 학습**이다. A1 기준선을 확보하고 A2를 우선 비교한 뒤
A3를 진행한다. 각 fold에서 정해진 학습 통화를 전부 사용하며, 작은 표본만 학습한
결과를 최종 비교에 사용하지 않는다.

실행 전 검증:

- 기존 embedding의 hash, `(27985, 768)`, float32, NaN/inf 없음 확인.
- MFCC 54개 feature의 이름·순서·추출 설정과 전체 call coverage 확인.
- `call_id`로 1:1 결합하고 중복·누락·label/partition 불일치가 있으면 중단.
- gender, partition, ID 등 메타데이터가 입력 feature에 들어가지 않도록 확인.
- 기존 call cache는 유지하고 새 feature 결합·학습 결과는 별도 경로에 저장.

**시간:** LR 후보들은 분 단위 작업을 예상하지만 입력 검증·저장 시간은 별도다.
RBF SVM은 훨씬 길어질 수 있으므로 첫 fold의 시간과 RAM을 측정한 뒤 전체 예상치를
갱신한다. 기존 LR 1회 약 3초를 SVM 실행 시간으로 일반화하지 않는다.

**다음 단계 판단:** 기존 LR 대비 fold별 개선과 OOF 오답 감소를 확인한다.
일부 fold에서만 오른 결과는 안정적인 개선으로 단정하지 않는다. 작은 이득도 기록하되,
OOF에서 99%와 격차가 크거나 개선이 반복되지 않으면 제한된 A 후보군 비교를 마치고
B·C로 넘어간다. CV 99% 역시 최종 Validation 99%의 보장은 아니다.

## B. 공통 오답의 원인 분석

Train OOF 예측에서 기준선과 새 후보가 모두 틀린 통화, 새로 맞힌 통화, 새로 틀린
통화를 분리한다. 전체 오류의 발화 길이·segment 수·성별 분포를 집계하고, 청취 분석은
선정 기준과 표본 수를 기록한다. 정답 사례도 비교군에 포함한다.

| 확인할 원인 | 결과에 따른 후속 방향 |
|---|---|
| 짧은 발화·무음·clipping·잡음 | 입력 품질과 짧은 segment 처리 점검 |
| 여러 화자·상담원 혼입 가능성 | caller crop 및 라벨 시간 경계 점검 |
| 발화마다 상반된 예측 | segment 집계 방식 비교 검토 |
| 데이터·라벨 불일치 | 근거와 영향 범위를 기록; 기존 평가 정답을 임의 수정하지 않음 |
| 정상 입력에서도 반복되는 오류 | 다른 표현 또는 부분 fine-tuning 검토 |

길이 가중 집계 등의 실험에는 segment별 embedding이 필요하다. 현재 call 평균
embedding만으로 복원할 수 있다고 가정하지 않고 저장 자료부터 확인한다.

## C. 성별 분류용 사전학습 모델 비교 → ECAPA 평가

초기 가설이던 화자 인식용 `speechbrain/spkrec-ecapa-voxceleb`의 embedding + LR 대신,
성별 분류용으로 학습된 `JaesungHuh/voice-gender-classifier`와 `tiantiaf/wavlm-large-age-sex`를 비교했다.
두 모델 모두 추가 학습 없이 frozen 상태에서 성별 확률을 출력했다.

동일 Train 200통화에서 ECAPA 99%, WavLM 95%, 기존 LR/fusion OOF 97.5%였다.
200개 차이만으로 최종 성능이나 통계적 우월성을 확정하지 않는다.
실행 비용을 줄이기 위해 2,000개 확대 비교를 생략하고, ECAPA의 checkpoint·전처리·집계를 고정한 뒤
기존 Internal Validation 5,597개를 평가한다. 학습이 없으므로 Train 22,388개에 별도 fit하지 않는다.
위 CV·Scaler 규칙은 A의 학습형 분류기에 적용되며 이 frozen 직접 분류기에는 적용하지 않는다.

caller 전체 연결 → 최대 15초 균등 창 → 실제 길이 가중 확률 평균으로 기존 Wav2Vec2와 집계가 다르므로
전체 파이프라인을 비교한다. 정확한 revision·조건·저장 방식과 판정 기준은
[200통화 결과 및 평가 계획](experiments/gender_model_comparison/SCREEN_200_REPORT.md),
실행 인계는 [다음 창 프롬프트](experiments/gender_model_comparison/ECAPA_VALIDATION_HANDOFF.md)에 기록했다.

## D. 표현 개선이 더 필요한 경우

Frozen 후보들의 Train OOF 오류가 많이 남으면 다음 두 방향 중 원인에 맞는 것을 선택한다.

- **부분 fine-tuning:** 기존 Wav2Vec2의 convolutional encoder와 하위 층을 고정하고
  상위 Transformer 일부와 분류 head를 학습하는 후보. 해제 층 수, learning rate,
  call 단위 loss·집계, epoch·early stopping을 실행 전에 별도 설계로 확정한다.
- **발화 집계 개선:** 오류 분석에서 근거가 있을 때 동일 가중 평균과 길이 가중 평균
  등을 비교한다. 재추출 필요 여부와 비용을 먼저 확인한다.

이 단계의 세부 설정은 아직 확정하지 않았다. 여러 변경을 한 번에 적용하지 않고 각각의
효과를 비교한다. 경량 CNN은 속도 목표의 별도 후보이며 99% 정확도를 보장하는 대안으로
취급하지 않는다. HuBERT·대규모 탐색·앙상블을 자동으로 추가하지 않는다.

## 결과 기록과 완료 기준

실험별로 설정·코드/입력 hash·fold 배정·OOF 예측·성별별 지표·학습/추론 시간·메모리를
남긴다. 최종 후보는 다음 표로 기존 baseline과 비교한다.

| 실험 | CV 평균/범위 | Validation Overall / Male / Female | 오답 수·순감소 | 시간 | 상태 |
|---|---|---|---|---|---|
| 기존 Wav2Vec2 + LR | 추가 비교 때 측정 | 97.248526 / 96.755725 / 97.682230% | 154 / 기준 | 추출 11.19h, LR 약 3s | 완료 |
| A1 · C=0.1 | 97.297651% / 97.118735–97.467506% | 미측정 | 미측정 | A1 9회 fit 총 18.30s | Train CV 완료 |
| A2 · C=0.1 | 97.453991% / 97.413562–97.521104% | 미측정 | 미측정 | 선택 후보 3-fold fit 총 4.65s | Train CV 완료 |
| A3 · C=1 | 96.663390% / 96.502747–96.837733% | 미측정 | 미측정 | 선택 후보 3-fold fit 총 60.37s | Train CV 완료 |
| C / D | 미측정 | 미측정 | 미측정 | 미측정 | 조건부 후보 |

Validation에서는 confusion matrix, 기존 모델과의 오류 겹침, 새 정답/새 오답을 함께
기록한다. 작은 차이는 같은 통화의 paired 비교(예: exact McNemar test)와 불확실성을
함께 해석하며, 유의확률 하나로 반복 선택의 영향을 없앴다고 주장하지 않는다.

99% 달성은 **최소 5,542/5,597 정답**으로 확인한다. 미달이면 실제 수치와 남은 오류를
기록한다. 최종 모델 선택 후 WAV/JSON → feature → 예측 CSV 통합 추론을 검증한다.
원본 WAV·모델 weight·대용량 embedding은 GitHub에 올리지 않는다.

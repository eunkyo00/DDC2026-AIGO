# 고정 0.5 분류 head 비교 — 2026-09-22 결과 확인

사용자가 Colab에서 실행한 결과 백업 `m3_head_results`를 정리했다. 여기서 새로 GPU 학습한 결과가 아니다.
실행 날짜는 로그에 별도로 기록되지 않아 제목은 결과 확인일이다.

## 목적과 비교 조건

임계값 튜닝 없이 분류 head의 효과를 비교한다. KLUE-RoBERTa Base, First-512,
BCE, seed 42, 4 epoch를 공통으로 사용하고 마지막 epoch를 평가했다.
내부 학습 23,408건 / 내부 검증 5,792건이며 공식 Validation 평가가 아니다.
학습률 2e-5, batch size 8, gradient accumulation 1, warmup 1,000 steps,
weight decay 0.01, FP16이다. 각 실험의 상세 환경과 모델 revision은 `run_config.json`에 있다.

CLS는 첫 토큰, Mean은 본문 토큰 평균, Label attention은 증상별 가중합을 분류기에 전달한다.
Mean/Attention은 패딩과 CLS/SEP를 제외한다. Attention은 공유 Dense/Tanh와 증상별 출력 가중치를 사용한다.
이 실험은 간단한 attention 구현 비교이며 특정 논문의 전체 재현이 아니다.

## 결과

| Head | Macro F1 (0.5) | CLS 대비 | 오심 precision | 오심 recall | 오심 F1 |
|---|---:|---:|---:|---:|---:|
| CLS | 0.610240 | — | 0.4400 | 0.0988 | 0.1614 |
| Mean | 0.611556 | +0.001316 | 0.5039 | 0.0973 | 0.1631 |
| Label attention | **0.614364** | **+0.004124** | 0.5000 | 0.1123 | 0.1834 |

Label attention이 이번 비교에서 가장 높았으나, 단일 seed 결과로 통계적 우위를 확정하지 않는다.
과거 E0 0.613300, E3 0.615701, E0+E3 0.617433은 별도 실행의 참고값이다.
이번 attention 모델은 기존 최고 앙상블을 넘지 못했다. 새 CLS와 과거 E0의 수치는
초기화·구현 차이가 있어 정확한 재현값으로 취급하지 않는다.

## 오심 오류

실제 오심 668건 중 attention은 75건을 맞히고 593건을 놓쳤다(TP 75 / FP 75 / FN 593).
CLS는 TP 66 / FP 84 / FN 602, Mean은 TP 65 / FP 64 / FN 603이다.
head 변경으로 소폭 개선됐지만 낮은 recall이 남았다. 이 수치만으로 라벨 오류나
텍스트 근거 부족이 원인이라고 단정할 수 없다. 원문 오류 분석 후 attention 구조를
고정한 BCE/가중 BCE 비교가 다음 후보이며, 아직 실시 결과는 없다.

## 검증과 보관 범위

- 세 실험 모두 로그에 epoch 4, step 11,704가 기록됐다.
- 로컬 예측 배열은 각각 (5792, 9), 유한한 0~1 확률이다.
- 기존 고정 0.5 E0 백업과 검증 파일 ID 순서 및 정답 배열이 정확히 일치했다.
- 저장 예측에서 고정 0.5 Macro F1을 재계산해 metrics.json과 일치함을 확인했다.
- run_config의 원본 manifest SHA256은 `36f917c70fe5d9bc782c609bbe1f2b68f2cfa3499e0ba53c6070a90e7ae4d0ec`이다.
  과거 문서의 재저장 manifest 해시와 바이트 표현은 다를 수 있다. 위 검증은 해시만이 아닌
  실제 검증 ID·정답 비교에 근거한다.
- 세 모델의 집계 metrics, 증상별 집계 CSV, 실행 설정, 학습 로그와 출력 없는 노트북을 Git에 보관한다.
- 원본 대화, 개별 파일 ID/예측, split_manifest, 모델/체크포인트는 Git에 올리지 않는다.
- 받은 결과 백업에는 모델 가중치가 없다. 새 데이터 추론에는 별도 모델 백업이 필요하다.

## 실행

[Colab 실험 노트북](../../models/roberta/M3_fixed05_head_experiments.ipynb)을 새 Colab GPU 런타임에서 연다.
Training ZIP은 Drive에서 읽고 기존 split_manifest.csv 및 m3_fixed_results.tar.gz를 /content에 올린다.
섹션별로 오류 분석 → CLS → Mean → Label attention → 결과 백업 순으로 실행한다.
모델 복원에는 노트북의 SymptomModel 클래스와 별도 모델 디렉터리가 필요하다.
전체 실행하면 세 모델이 순차 학습된다. 결과 압축과 모델 백업은 별개다.

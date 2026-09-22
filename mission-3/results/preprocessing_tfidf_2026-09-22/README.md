# KLUE-RoBERTa + TF-IDF 전처리 비교 결과

사용자가 Colab에서 실행한 결과 백업 `m3_roberta_tfidf_preprocess_results.tar.gz`를
Git 공유용으로 정리한 기록이다. 원본 대화, 개별 예측 배열, split manifest, 모델 가중치는
저장소에 넣지 않았다.

## 고정 조건

- 평가: 내부 검증 5,792건, 학습 23,408건
- 임계값: 모든 증상에 0.5 고정
- RoBERTa: `klue/roberta-base` + 증상별 Attention head, BCE, 4 epoch, seed 42
- TF-IDF: 문자 2~5 gram + One-vs-Rest Logistic Regression
- 앙상블: RoBERTa 확률 0.5 + TF-IDF 확률 0.5
- TF-IDF는 모든 조건에서 공백으로 연결한 원문을 사용하고, 한 번만 학습했다.
- 공식 Validation 성능이 아니며, 내부 검증에서의 탐색 결과다.

## 결과

| 실험 | 고정 0.5 Macro F1 | 오심 F1 |
|---|---:|---:|
| TF-IDF 단독 | 0.4706 | 0.0474 |
| RoBERTa raw | 0.6075 | 0.1650 |
| RoBERTa 발화 경계 보존 | **0.6114** | 0.1654 |
| RoBERTa raw + TF-IDF 0.5/0.5 | 0.5869 | 0.0830 |
| RoBERTa 발화 경계 + TF-IDF 0.5/0.5 | 0.5830 | 0.0782 |

발화 사이에 구분 토큰을 넣자 RoBERTa 단독 Macro F1이 0.6075에서 0.6114로 올랐다.
TF-IDF와의 단순 확률 평균은 두 조건 모두 단독 RoBERTa보다 낮았다. 따라서 이번 고정
0.5 조건에서는 TF-IDF를 단순히 0.5로 섞는 것이 개선으로 이어지지 않았다. 과거의
threshold 조정 앙상블 점수와는 평가 규칙이 다르므로 직접 비교하지 않는다.

최소 정규화 조건은 Unicode NFC·연속 공백 정리 후 tokenizer 입력이 Training에서
변하지 않아 학습을 생략했다. 길이 통계는 `length_stats.json`에서 확인한다. 구분 토큰
조건은 512토큰 초과 비율이 raw보다 증가하므로, 개선 폭을 과대해석하지 않는다.

## 파일 안내

- `summary.csv`: 단독·앙상블 Macro F1과 오심별 지표
- `length_stats.json`: 전처리별 token 길이와 512 초과 비율
- 각 실험 폴더의 `metrics.json`, `per_label.csv`: 저장된 성능과 증상별 지표
- [재실행 노트북](../../models/roberta/M3_KLUE_RoBERTa_TFIDF_fixed05_preprocessing.ipynb)

예측 배열은 로컬 백업 압축파일에만 보관한다. 모델을 복원하거나 새 데이터에 추론하려면
별도의 모델 가중치 백업이 필요하다.

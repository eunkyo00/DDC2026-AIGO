# 고정 임계값 0.5 재실험 기록

**실행일:** 2026-09-20  
**평가 지표:** 9개 증상의 Macro F1  
**평가 데이터:** Training에서 분리한 내부 검증 세트 5,792건  
**학습 데이터:** 동일 manifest의 내부 학습 세트 23,408건  
**임계값 규칙:** 모든 증상에 `0.5`를 고정했다. 증상별 threshold 탐색은 수행하지 않았다.

## 목적

멘토링 지침에 따라, 기존의 증상별 threshold 조정 결과와 분리해 **0.5 고정 조건**에서
모델을 다시 비교했다. 따라서 이 문서의 점수는 이전의 threshold 조정 점수와 직접 섞어
비교하지 않는다.

## 결과

| 실험 | 설정 | 고정 0.5 Macro F1 |
|---|---|---:|
| TF-IDF | 문자 2~5 gram + One-vs-Rest Logistic Regression | 0.4706 |
| E0 | KLUE-RoBERTa Base, First-512 | 0.6133 |
| E1 | KLUE-RoBERTa Base, Head-Tail | 0.6102 |
| E3 | KLUE-RoBERTa Large, Head-Tail | 0.6157 |
| E0 + E3 | 확률 동일 가중치 0.5 / 0.5 | **0.6174** |

E3 단독은 E0보다 0.0024 높았고, E0+E3 동일 가중치 앙상블은 E3 단독보다 0.0017
높았다. 개선 폭이 작으므로 단일 E3와의 우열을 확정하는 근거로는 부족하다.

## 실행 조건

- 입력에는 `utterances[].text`만 원래 순서대로 연결해 사용했다.
- 입력에 WAV, speaker, recordId, audioPath, 주소, 정답 라벨을 넣지 않았다.
- 클래스 순서는 `고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란`이다.
- 내부 분할은 E0 원본 manifest를 복원해 사용했고, SHA256은
  `69817218fd5677067294a6d65ec44a3d0b4d7d5d4690f71f6fa4eaea1cfcf64e`이다.
- Colab 실행 결과의 예측값·지표·설정은 로컬 백업 `m3_fixed_results.tar.gz`에 별도 보관한다.
  원본 데이터, 개별 예측, 모델 가중치는 GitHub에 올리지 않는다.

## 해석과 주의점

- 모든 수치는 **내부 검증** 결과이며, 공식 Validation 성능이 아니다.
- E0+E3 조합은 단독 모델 결과를 확인한 뒤 추가로 실행한 탐색적 비교다. threshold와
  가중치는 각각 0.5로 고정했지만, 이 조합 선택 자체도 내부 검증 결과에 영향을 받았다.
- 따라서 현 시점의 단일 모델 기준 후보는 E3이며, E0+E3은 탐색적 내부 검증 최고치로
  기록한다.
- 최종 모델과 설정을 정한 뒤에는 전체 Training으로 재학습하고, 공식 Validation에는
  학습과 선택에 사용하지 않은 동일한 전처리·고정 threshold를 적용해야 한다.

## 재실행 명령

```bash
python mission-3/baseline/tfidf/run_tfidf.py --out-dir /content/m3_fixed/tfidf --fixed-threshold
python mission-3/models/roberta/run_roberta.py --experiment e0 --out-dir /content/m3_fixed/e0 --fixed-threshold
python mission-3/models/roberta/run_roberta.py --experiment e1 --out-dir /content/m3_fixed/e1 --fixed-threshold
python mission-3/models/roberta/run_roberta.py --experiment e3 --out-dir /content/m3_fixed/e3 --fixed-threshold
python mission-3/ensemble/run_ensemble.py --a /content/m3_fixed/e0/dev_predictions.npz --b /content/m3_fixed/e3/dev_predictions.npz --out-dir /content/m3_fixed/e0_e3_equal --fixed-threshold --weight-a 0.5
```

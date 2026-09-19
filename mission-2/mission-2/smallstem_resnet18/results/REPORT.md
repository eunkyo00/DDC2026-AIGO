# Small-stem ResNet18 결과 기록

출처: 사용자 제공 `mission02_experiment_smallstem-20260919T042641Z-1-001.zip` 안의 `best_model.pt` 및 `history.json`. 두 파일을 비교하고 체크포인트의 모델을 strict loading해 `[1,1,64,24]` 입력으로 forward 연산이 가능한지 확인했습니다. 원본 Validation 음원으로 독립 재실행한 것은 아닙니다.

| 항목 | 값 |
|---|---:|
| Epoch | 16 |
| 학습 발화 | 889,358 |
| 공식 Validation 발화 | 111,947 |
| 모델 | `resnet18_smallstem` (11,169,858 parameters) |
| 입력 | 앞 1.5초 `first_1p5`, 64-band log-Mel, `[64,24]` |
| Validation loss | 0.2737853731 |
| Accuracy (threshold 0.5) | 90.139977% |
| Accuracy (Validation에서 선택한 threshold 0.535) | **90.248957%** |
| Confusion matrix (`[[TN, FP], [FN, TP]]`) | `[[46396, 7449], [3467, 54635]]` |

이 수치는 발화 단위 공식 Validation 결과이며, threshold 0.535와 최종 모델을 같은 Validation에서 선택했습니다. 테스트 데이터 성능, 학습 정확도, 추론 시간 또는 제출 승인 결과로 해석하면 안 됩니다. 전체 epoch 로그는 [history.json](history.json)을 참고하세요.

검증에 사용한 파일 SHA-256:

```text
best_model.pt  5a5dfa14227bb7c0961ff13e8f845c6d58bb66124b5875d316cf1e73ce9e84f5
history.json   cdbfbacbb463603d475dc57acfd89f7e7906ebbd5d1196bf08525f51edfea3e5
```

가중치와 데이터는 저장소에 미포함입니다. 공개 저장소의 기록만으로 이 수치를 독립 재검증할 수 없다는 한계가 있습니다.

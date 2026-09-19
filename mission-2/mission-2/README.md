# Mission 2 — 119대원 / 신고자 발화 분류

서울 지역 통화의 발화별 음성만으로 `speaker=0`(119대원)과 `speaker=1`(신고자)을 구별합니다. [smallstem_resnet18](smallstem_resnet18/README.md)에 실행 코드, 재현 방법 및 검증 결과를 정리했습니다.

| 모델 | 공식 Validation 발화 수 | 정확도 | 비고 |
|---|---:|---:|---|
| Small-stem ResNet18 | 111,947 | **90.248957%** | epoch 16, Validation에서 선택한 threshold 0.535 |

위 값은 Validation 정확도이며, 비공개 테스트 정확도나 독립 데이터에서의 성능을 보장하지 않습니다. 모델 가중치, 원본 음원 및 대용량 전처리 특징은 저장소에 올리지 않습니다.

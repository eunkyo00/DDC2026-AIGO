# Mission 2 — 신고자 / 119대원 발화 분류

## 1. Mission 2 한눈에 보기

목표는 통화 속 **한 발화 조각**이 `speaker=0`(119대원)인지 `speaker=1`(신고자)인지 분류하는 것이다. 입력은 WAV와 발화의 `startAt`·`endAt`이 담긴 JSON, 평가지표는 **발화 단위 Accuracy**, 최종 출력은 발화별 예측 CSV다. Mission 1의 통화 단위 성별 분류와 평가 단위가 다르다.

```text
WAV + JSON의 startAt/endAt → 해당 발화 음성 자르기
    → 1.5초 log-Mel 특징 → Small-stem ResNet18
    → 발화별 speaker 0/1 → CSV
```

- **WAV**: 모델이 실제로 듣는 음성.
- **JSON의 `startAt`·`endAt`**: 발화 구간을 자르는 데만 사용한다. 단위는 밀리초(ms)다.
- **JSON의 `speaker`**: 학습·평가 정답이며 모델 입력은 아니다.
- **`text`, 발화 순서, 파일명, 주소 등**: 화자 예측 입력으로 사용하지 않는다.

## 2. 데이터와 전처리

### 데이터 구조

서울 지역 데이터에 공통 검사·매니페스트 생성 파이프라인을 실행해 확인한 수치다. Mission 1 문서의 별도 데이터 감사 수치를 가져온 것이 아니다.

| 항목 | Training | 공식 Validation |
|---|---:|---:|
| WAV / JSON 매칭 통화 | 29,200 / 29,200 | 3,640 / 3,640 |
| 유효한 발화 | 889,358 | 111,947 |
| `speaker=0` 119대원 | 427,161 | 53,845 |
| `speaker=1` 신고자 | 462,197 | 58,102 |

검사한 WAV는 모두 **8kHz, mono, 16-bit PCM**이었다. 이 Mission 2 파이프라인에서는 WAV/JSON 누락·손상 0건을 기록했다. 단, 이 수치는 사용한 데이터 복사본에 대한 검사 결과다. 공통 파이프라인의 실행 순서는 저장소 루트 [README](../README.md)에 있다.

### 발화 구간과 특징

공통 `build_manifests`가 생성한 `mission2_speaker_Training.csv`와 `mission2_speaker_Validation.csv`에서 한 행이 발화 하나다. 매니페스트에는 `stem,wav_path,utterance_id,startAt,endAt,speaker`가 있으나 **`text` 컬럼은 없다**. `stem`·경로·ID는 파일 찾기/매칭에만 쓰며 분류기 특징에 넣지 않는다.

```text
WAV + startAt/endAt(ms)
    → crop_segment
    → 1.5초로 끝부분 zero-pad 또는 앞 1.5초만 유지
    → 64-band log-Mel spectrogram
    → [0,1] 정규화 → [64,24] float16 특징 저장
    → 학습 시 float32로 읽기
```

음성 읽기·구간 자르기·길이 맞춤·Mel 추출은 공통 전처리와 같은 함수 구현을 사용한다. 특징을 미리 계산한 이유는 약 36GB의 원본 WAV를 Colab에 올리지 않고 학습하기 위해서다. **학습에 사용한 특징은 앞 1.5초(`first_1p5`)**이며, 코드에 보이는 `first_plus_whole`은 이 결과와 무관한 후속 실험용 분기다.

### EDA에서 확인한 것

발화 수·역할 분포와 WAV 헤더를 확인했다. Validation에서 119대원 발화의 중앙 길이는 **1.399초**, 신고자 발화는 **1.485초**였다. 따라서 1.5초 고정 입력은 일부 긴 발화의 뒷부분을 버린다. 검사 방법과 집계는 [EDA 보고서](eda/REPORT.md)에 있다. 저장소에는 원본 데이터·개별 발화 매니페스트·대용량 특징을 포함하지 않았다.

## 3. 전체 실험 전략

```text
데이터 스키마·매칭·서울 필터·오디오 검사
    → 공식 Training/Validation 매니페스트 생성
    → 발화 음성 특징 사전 계산
    → ResNet18 기준 모델
    → Small-stem ResNet18 학습·Validation 비교
    → 최적 checkpoint/threshold 선택
    → 단일 명령 추론 CSV 생성
```

이 문제에서는 짧은 `[64,24]` Mel 입력의 시간·주파수 해상도를 초기에 과하게 줄이지 않는지 확인하기 위해 Small-stem 구조를 시험했다. 공식 Training과 Validation 폴더를 분리해 사용하며, Validation은 가중치 학습이 아니라 **epoch와 판정 threshold 선택**에 사용한다. 따라서 보고된 수치는 독립 Test 성능이 아니다.

발화별 입력과 정답은 통화 단위로 묶인 원본 split에서 나온다. 같은 통화의 발화를 임의로 Training/Validation에 나눠 넣는 실험은 하지 않았다. 다만 동일 인물이 서로 다른 통화로 두 split에 존재하는지는 식별자가 없어 검증하지 못했다.

## 4. 모델별 실험 방법과 결과

| 방법 | 사용하는 정보 | 목적 |
|---|---|---|
| 다수 클래스 고정 예측 | 음성 없음 | 최소 비교 기준 |
| 기본 ResNet18 | 앞 1.5초 log-Mel | 학습 기준 모델 |
| **Small-stem ResNet18** | 같은 앞 1.5초 log-Mel | 작은 Mel 입력의 해상도 보존 효과 확인 |
| ResNet18 + TDNN hybrid | 같은 특징의 이미지/시간축 표현 | 후속 비교 실험 |

| 방법 | Validation 발화 Accuracy | 근거 |
|---|---:|---|
| `speaker=1` 고정 | 51.901346% | 공식 Validation의 58,102 / 111,947에서 계산 |
| 기본 ResNet18 | 88.594603% | 앞서 공유된 학습 로그; 이 폴더에 체크포인트/로그 미포함 |
| **Small-stem ResNet18** | **90.248957%** | 제공된 `best_model.pt`와 [16 epoch 로그](smallstem_resnet18/results/history.json) 대조 |
| Hybrid | 약 90.00% | 앞서 공유된 요약값; 이 폴더에 체크포인트/로그 미포함 |

수치의 근거 수준이 다르므로 위 네 줄을 모두 동일하게 재검증된 실험으로 보아서는 안 된다. 이 저장소에서 결과 기록까지 포함한 모델은 **Small-stem**이다. 기본 ResNet18 대비 약 **1.65%p** 높다는 비교도 앞서 공유된 기본 모델 로그가 정확하다는 전제에서만 성립한다.

### Small-stem ResNet18

일반 ResNet18의 큰 `7×7, stride=2` 입력 convolution과 max-pooling 대신 `3×3, stride=1` convolution과 pooling 생략을 사용한다. ImageNet 사전학습 가중치를 초기값으로 사용하고, 학습 중 SpecAugment·Mixup·label smoothing·AdamW·cosine scheduler를 적용했다. 훈련은 **16 epoch**까지 진행됐으며 가장 높은 Validation Accuracy도 epoch 16에서 나왔다.

| 핵심 설정 | 값 |
|---|---|
| 입력 | `first_1p5`, `[1,64,24]` log-Mel |
| 모델 크기 | 11,169,858 parameters |
| Batch / 학습률 | 512 / `2e-4` |
| Dropout / label smoothing / Mixup alpha | 0.25 / 0.03 / 0.10 |
| 선택된 threshold | 0.535 (Validation에서 탐색) |
| Validation loss | 0.2737853731 |
| threshold 0.5 Accuracy | 90.139977% |
| threshold 0.535 Accuracy | **90.248957%** |

검증 혼동행렬은 실제 행·예측 열 순서로 `[[46396, 7449], [3467, 54635]]`이다. 즉 119대원 발화의 정답률은 **86.17%**, 신고자 발화의 정답률은 **94.03%**다. 전체 Accuracy만으로 두 역할의 오류 차이를 놓치지 않도록 함께 기록한다. 자세한 근거와 파일 해시는 [결과 보고서](smallstem_resnet18/results/REPORT.md)에 있다.

## 5. 결과 해석과 다음 분석

### 현재 확인된 한계

- **Validation 선택 편향**: 같은 Validation에서 최고 epoch와 threshold 0.535를 선택했다. 비공개 Test에서도 90.25%가 나온다는 의미는 아니다.
- **긴 발화의 손실**: 앞 1.5초만 사용하므로 긴 발화의 뒷부분에 있는 음향 정보는 모델에 들어가지 않는다.
- **역할별 성능 차이**: 검증에서 `speaker=0` 정답률이 `speaker=1`보다 낮다. 원인은 아직 발화별 오류 검토로 확정하지 않았다.
- **체크포인트 비공개**: 원본 음성·대용량 특징·모델 가중치를 저장소에 넣지 않아 README와 로그만으로 점수를 독립 재실행할 수 없다.

### 추가로 확인할 것

짧은 발화, 무음·겹침, 긴 발화에서 오답이 몰리는지 살피고 전체 발화 정보를 쓰는 별도 특징 실험과 비교할 수 있다. 이런 분석과 후속 모델의 성능은 **이 90.25% 결과에 포함되지 않는다**. 소스의 `analyze_validation.py`는 길이별 오류 진단용이며, 해당 진단 결과를 아직 이 README의 확정 수치로 제시하지 않는다.

## 6. 최종 Inference Pipeline

```text
테스트 WAV + JSON의 발화 startAt/endAt
    → 학습과 동일한 1.5초 log-Mel 전처리
    → 저장된 Small-stem 모델과 threshold 0.535
    → [audio file name, startAt, endAt, speaker] CSV
```

`inference.py`는 한 번의 명령으로 테스트 폴더에서 CSV를 만든다. `speaker`와 `text`를 JSON에서 모델 입력으로 읽지 않는다. 다만 **제출에는 학습 완료된 `best_model.pt`와 추론 코드가 함께 필요**하다. 가중치는 GitHub에 없으므로 별도로 보관한 파일을 `--ckpt_path`에 지정해야 한다.

```bash
cd mission-2/smallstem_resnet18
python inference.py \
  --audio_dir "/path/to/test/1.원천데이터" \
  --label_dir "/path/to/test/2.라벨링데이터" \
  --ckpt_path "/path/to/mission02_experiment_smallstem/best_model.pt" \
  --output "/path/to/outputs/mission2.csv"
```

## 7. 재현 방법과 상세 문서

공통 데이터 검사·매니페스트 생성은 저장소 루트 [README](../README.md)의 0~5단계를 따른다. 이미 사전 생성 특징 `mission02_features/`가 있다면 원본 WAV 없이 학습할 수 있다. Small-stem의 정확한 전처리·학습 명령, Colab 실행 조건과 재개 방법은 [실험 README](smallstem_resnet18/README.md)에 정리했다.

| 내용 | 문서/파일 |
|---|---|
| Mission 2 학습·전처리·추론 코드 | [smallstem_resnet18/](smallstem_resnet18/) |
| 데이터 구조·EDA | [EDA README](eda/README.md), [EDA 보고서](eda/REPORT.md) |
| Small-stem 실행 방법 | [실험 README](smallstem_resnet18/README.md) |
| 결과 해석·체크포인트 검증 | [결과 보고서](smallstem_resnet18/results/REPORT.md) |
| 매 epoch Validation 기록·집계 | [history.json](smallstem_resnet18/results/history.json), [metrics.json](smallstem_resnet18/results/metrics.json) |
| GitHub에 올릴 것/제외할 것 | [UPLOAD_GUIDE.md](UPLOAD_GUIDE.md) |

원본 데이터, 전처리 특징, `best_model.pt`, `last_checkpoint.pt`는 저장소에 포함하지 않는다. 이 결과는 **발화 단위 공식 Validation 기록**이지 최종 비공개 Test 결과나 제출 완료 증명이 아니다.

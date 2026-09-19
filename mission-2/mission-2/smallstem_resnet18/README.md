# Mission 2 — Small-stem ResNet18 (Validation 90.25%)

서울 지역 Training 889,358개 발화로 학습하고 공식 Validation 111,947개 발화로 평가한 실험입니다. 음향 외의 전사 텍스트·화자 정답·발화 순서·통화 ID는 모델 입력으로 사용하지 않습니다. `speaker=0`은 119대원, `speaker=1`은 신고자입니다.

## 재현된 결과

첨부된 `mission02_experiment_smallstem` 실험 폴더의 `best_model.pt`와 [history.json](results/history.json)을 대조했습니다. 가장 높은 Validation 정확도는 epoch 16에서 **90.248957%**(101,031 / 111,947)이며, 선택된 threshold는 **0.535**입니다. 기본 threshold 0.5의 정확도는 **90.139977%**입니다. 정확한 수치와 confusion matrix는 [결과 보고서](results/REPORT.md)를 참고하세요. 이는 같은 Validation 데이터에서 threshold와 최고 epoch를 선택한 수치이며 독립 테스트 성능이 아닙니다.

가중치 `best_model.pt`(약 45MB), 중단 재개용 `last_checkpoint.pt`(약 134MB), 원본 WAV/JSON, 매니페스트 및 사전 생성 특징(약 3GB)은 저장소에 포함하지 않았습니다. 현재 작성자의 체크포인트는 Google Drive의 `MyDrive/DCC/mission02_experiment_smallstem/best_model.pt`에 있습니다. 저장소만 내려받으면 **기록된 90.25%를 바로 재평가하거나 예측할 수 없습니다**. 체크포인트와 입력 특징/테스트 음원에 대한 접근이 별도로 필요합니다.

## 전처리와 데이터

저장소 루트에서 `python -m common.utils.inspect_schema` → `python -m common.utils.io` → `python -m common.utils.filter_seoul` → `python -m common.utils.audio_check` → `python -m common.eda.eda_all` → `python -m common.preprocessing.build_manifests` 순으로 실행해 서울 데이터 매니페스트를 생성합니다. 학습/검증 매니페스트는 각각 `manifests/mission2_speaker_Training.csv`, `manifests/mission2_speaker_Validation.csv`입니다. 그 안의 WAV 절대경로는 로컬 환경에만 유효하므로 커밋하지 않습니다.

`github_preprocessing.py`는 공통 전처리 함수 네 개(`load_audio`, `crop_segment`, `pad_or_trim_to_length`, `extract_melspectrogram`)의 구현을 저장소 commit `c837272`와 대조한 복사본입니다. `precompute_features.py`는 각 발화의 `startAt`/`endAt`(ms)으로 WAV를 잘라 앞 1.5초로 패딩/트림하고, 64-band log-Mel을 `[0,1]`로 변환해 `[64,24]` float16 shard로 저장합니다. 이 체크포인트는 `first_1p5` 특징을 사용했습니다. 소스에 포함된 `first_plus_whole` 분기는 **별도 후속 실험**이며 이 90.25% 측정에 사용하지 않았습니다.

원본 데이터에 접근할 수 있다면 저장소 루트의 공통 스크립트 실행 후 이 디렉터리에서:

```bash
python -m pip install -r requirements_precompute.txt
python precompute_features.py \
  --data-root "/path/to/대학부 데이터" \
  --train-manifest ../../manifests/mission2_speaker_Training.csv \
  --valid-manifest ../../manifests/mission2_speaker_Validation.csv \
  --output-dir "/path/to/mission02_features" \
  --workers 6 --shard-size 50000
python verify_features.py "/path/to/mission02_features"
```

원본 데이터 없이 **이미 만들어진** `mission02_features/`가 있다면 위 생성 단계는 생략할 수 있습니다. `metadata.json`, `Training/`, `Validation/`이 필요합니다.

## Small-stem 학습

CUDA 환경에서 이 디렉터리의 `train.py`를 실행합니다. 출력 경로를 Google Drive처럼 지속되는 위치로 지정해야 런타임이 끊겨도 epoch 단위로 재개할 수 있습니다.

```bash
python -m pip install -r requirements.txt
python train.py \
  --feature-root "/path/to/mission02_features" \
  --output-dir "/path/to/persistent/mission02_experiment_smallstem" \
  --model-name resnet18_smallstem \
  --epochs 16 --batch-size 512 --num-workers 2 \
  --learning-rate 2e-4 --dropout 0.25 \
  --label-smoothing 0.03 --mixup-alpha 0.10 \
  --patience 4 --resume
```

이 명령은 제공된 `DCC_mission2_improve.ipynb`의 Small-stem 셀과 동일합니다. 처음 실행 시 ImageNet ResNet18 초기 가중치를 다운로드합니다. 매 epoch `history.json`과 `last_checkpoint.pt`를 저장하고, 최고 Validation 모델만 `best_model.pt`로 갱신합니다. Colab 노트북은 Drive에 `DCC/mission02.zip`, `DCC/mission02_features/` 및 기존 baseline 체크포인트가 있다는 전제에서 **baseline/small-stem/hybrid 세 모델을 비교**하므로, small-stem만 새로 실행하려면 위 명령을 사용하세요. 기록된 정확도의 재현은 환경과 학습 실행에 따라 달라질 수 있습니다.

## 최종 형식으로 추론

테스트 WAV/JSON 및 별도로 보관한 `best_model.pt`가 준비되면:

```bash
python inference.py \
  --audio_dir "/path/to/test/1.원천데이터" \
  --label_dir "/path/to/test/2.라벨링데이터" \
  --ckpt_path "/path/to/mission02_experiment_smallstem/best_model.pt" \
  --output "/path/to/outputs/mission2.csv"
```

CSV 컬럼은 `audio file name,startAt,endAt,speaker`입니다. JSON에서는 발화 구간의 `startAt`/`endAt`만 읽으며 추론 때 `speaker`나 `text`는 읽지 않습니다. `inference.py`는 저장된 `model_name`과 threshold를 체크포인트에서 읽습니다.

# Colab에서 실행하기

새 노트북 이름 예: `M3_E4_NewModel_Experiment.ipynb`.
기존 E0~E3는 아래 CLI로 재구성할 수 있다. E4 모델은 아직 선정/구현하지 않았다.
각 코드 블록을 별도 셀로 실행한다. GPU는 모델 학습 셀 전에 연결한다.

## 1. 브랜치와 환경

```python
!git clone --branch mission/sumin https://github.com/eunkyo00/DDC2026-AIGO.git /content/DDC2026-AIGO
%cd /content/DDC2026-AIGO
!pip -q install -r mission-3/requirements.txt
```

이미 clone한 폴더가 있으면 clone 셀을 반복 실행하지 않는다.
학교 Colab Pro와 Drive 저장공간은 별개다. 이 문서의 모델 출력은 모두 `/content`에 저장된다.

## 2. 데이터 준비

```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/DDC_M3/data/mission3_train_json.zip mission-3/data/raw/
!cp /content/drive/MyDrive/DDC_M3/data/mission3_validation_json.zip mission-3/data/raw/
!python mission-3/data/prepare.py --train-zip mission-3/data/raw/mission3_train_json.zip --validation-zip mission-3/data/raw/mission3_validation_json.zip
```

Mac의 E0 백업에서 작은 `split_manifest.csv`만 추출해서 Colab 파일 패널로 `/content`에 업로드한다.
400MB 모델을 올려야만 분할을 복구할 수 있는 것은 아니다.

```python
!python mission-3/validation/create_split.py --manifest /content/split_manifest.csv
```

## 3. 모델 실행

```python
!python mission-3/baseline/tfidf/run_tfidf.py --out-dir /content/m3_runs/tfidf
!python mission-3/models/roberta/run_roberta.py --experiment e1 --out-dir /content/m3_runs/e1
!python mission-3/models/roberta/run_roberta.py --experiment e3 --out-dir /content/m3_runs/e3
```

threshold를 0.5로 고정하는 실험이라면 모든 단독 모델 명령 끝에 `--fixed-threshold`를 붙인다.
그때는 증상별 threshold 탐색을 수행하지 않는다.

이 셀은 모델을 재학습한다. 기존에 온전한 가중치가 있다면 불필요하게 반복하지 않는다.
GPU 메모리 부족이면 batch를 줄이고 accumulation을 늘려 유효 batch=8을 유지한다.

## 4. 앙상블

```python
!python mission-3/ensemble/run_ensemble.py --a /content/m3_runs/e1/dev_predictions.npz --b /content/m3_runs/tfidf/dev_predictions.npz --out-dir /content/m3_runs/e2
!python mission-3/ensemble/run_ensemble.py --a /content/m3_runs/e3/dev_predictions.npz --b /content/m3_runs/e2/dev_predictions.npz --out-dir /content/m3_runs/e3_e2
```

고정 0.5 평가에서는 가중치 탐색도 하지 않는다. 사전에 정한 동일 가중치 앙상블은 다음과 같이 실행한다.

```bash
!python mission-3/ensemble/run_ensemble.py --a /content/m3_fixed/e0/dev_predictions.npz --b /content/m3_fixed/e3/dev_predictions.npz --out-dir /content/m3_fixed/e0_e3_equal --fixed-threshold --weight-a 0.5
```

## 5. 백업 및 재개

런타임 삭제 시 `/content`가 사라진다. notebook(.ipynb), 원본 split, metrics,
run_config, 예측 NPZ를 먼저 백업하고, 재학습을 피하려면 모델 폴더도 저장한다.

```python
!tar --exclude='*/checkpoints' -czf /content/m3_runs.tar.gz -C /content m3_runs
!gzip -t /content/m3_runs.tar.gz
!sha256sum /content/m3_runs.tar.gz
from google.colab import files
files.download('/content/m3_runs.tar.gz')
```

오류 없이 압축 검사가 끝난 뒤 다운로드한다. Mac에서도 `gzip -t 파일경로`와
`shasum -a 256 파일경로`로 무결성과 동일 해시를 확인한다.
단순히 파일명이 보인다는 것만으로 업로드/다운로드가 완료됐다고 판단하지 않는다.

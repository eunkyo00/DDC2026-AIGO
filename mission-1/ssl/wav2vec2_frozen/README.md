# Frozen Wav2Vec2 + Logistic Regression baseline

## 최종 결과: Colab L4 CUDA 추출 + 로컬 평가

전체 27,985통화·442,639 caller segment의 FP32 embedding을 NVIDIA L4에서 추출하고
완료 call, call_id, shape `(27985, 768)`, finite 값, 실패 0건을 검증했다.
실제 full extraction은 11.187423시간 걸렸다. 기존 Mac MPS 실행 안내는 초기 자원
평가의 역사적 기록이며 최종 embedding에는 섞이지 않았다.

Train-only StandardScaler와 기존 LogisticRegression 설정으로 로컬 평가한 결과,
동일 fixed Validation 5,597통화의 Accuracy는 **97.248526%**였다. Male은
**96.755725%**, Female은 **97.682230%**다. 상세 비교와 cache provenance는
[최종 REPORT](results/REPORT.md), 단계별 Colab 실행은 [Colab README](colab/README.md),
로컬 평가 진입점은 [evaluate_colab_export.py](evaluate_colab_export.py)를 참고한다.
원본 WAV, 77.53MiB 통합 embedding, 통화별 NPZ cache, 모델 weight는 GitHub에 올리지
않는다. 검증된 export SHA256은
`64cd8d608c5e7e18a0db86193f7b4b26fa08d689aa22e4dab6807b7360bd71c4`이다.

## 현재 사용하는 파일과 재현 방법

| 경로 | 역할 |
|---|---|
| `colab/` | 실제 CUDA 추출 notebook·runner·session 및 Drive resume |
| `evaluate_colab_export.py` | 실제 최종 평가에 사용한 로컬 export 검증·Scaler/LR 학습 |
| `evaluation_requirements.txt` | GPU 없이 로컬 평가에 필요한 고정 패키지 |
| `results/full_l4/` | 최종 metrics, Validation 예측, L4 benchmark와 추출 요약 |
| `results/REPORT.md` | 최종 결과와 오류 겹침·실측 시간 |
| `results/`의 smoke/benchmark JSON·CSV | 초기 Mac MPS 기록 |
| `checkpoint_smoke.py` | 초기 생성 파형 진단 |
| `verify_results.py` | 초기 Mac 결과용 검증기; 최종 `full_l4` 형식과 다름 |

현재 Colab 실험의 재현은 [Colab README](colab/README.md)를 따른다. 다운로드한
통합 embedding의 로컬 평가는 다음 명령을 사용한다. 아래의 Mac `--mode evaluate`에
Colab NPZ를 직접 넣지 않는다.

```bash
python3 -m venv /path/to/local-eval-venv
/path/to/local-eval-venv/bin/python -m pip install \
  -r mission-1/ssl/wav2vec2_frozen/evaluation_requirements.txt
/path/to/local-eval-venv/bin/python \
  mission-1/ssl/wav2vec2_frozen/evaluate_colab_export.py \
  --embeddings /path/to/wav2vec2_l4_full_embeddings.npz \
  --output /path/to/new-evaluation-results
```

이 명령은 기존에 검증된 export SHA256을 확인한다. 새로운 추출 결과는 별도 무결성
검토가 필요하다. 개별 오답 원인 분석과 새 WAV/JSON을 받는 최종 통합 추론 파이프라인은
아직 완료하지 않았다.

## 초기 Mac MPS 구현 기록 — 아래 명령은 최종 Colab 결과용이 아님

고정된 Mission 1 call-level split에서 `speaker=1` 신고자 구간만 사용한다. 8kHz 파형을
`scipy.signal.resample_poly(up=2, down=1, Kaiser beta=5.0)`로 실제 16kHz 파형으로
변환하고, frozen `facebook/wav2vec2-base`의 마지막 hidden state를 시간축 평균한다.
통화 embedding은 segment embedding의 동일 가중 평균이다.

## 고정 설정

- checkpoint: `facebook/wav2vec2-base`
- revision: `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8`
- split SHA256: `04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06`
- seed: 42
- hidden dimension: 768
- classifier: Train-only StandardScaler + LogisticRegression(`C=1`, `solver=lbfgs`,
  `max_iter=1000`, `class_weight=None`, `random_state=42`)

모델은 16kHz 입력을 요구한다. 이 checkpoint는 LibriSpeech 960시간의 음성으로
self-supervised pretraining된 base 모델이며 ASR fine-tuned checkpoint를 사용하지 않는다.

## 실행 순서

저장소 루트에서 다음 순서를 지킨다.

```bash
python mission-1/ssl/wav2vec2_frozen/test_wav2vec2_frozen.py
python mission-1/ssl/wav2vec2_frozen/run_wav2vec2_frozen.py --mode smoke --device mps
python mission-1/ssl/wav2vec2_frozen/run_wav2vec2_frozen.py --mode benchmark --device mps
```

원본 볼륨을 사용할 수 없을 때 `checkpoint_smoke.py`는 생성 파형으로 checkpoint 로딩,
MPS forward, 짧은 입력 padding과 15초 chunk 경로만 진단한다. 실제 WAV/caller crop smoke를
대체하지 않으므로 결과에도 별도로 표시한다.

장시간 실행은 embedding 추출과 classifier/Validation 평가를 별도 단계로 실행한다.
`extract`는 call 하나가 끝날 때마다 embedding, quality row, 완료 상태 순서로 각각
flush/fsync하고 마지막에 status=1을 commit marker로 기록한다. 재실행 전 완료 row의
shape/dtype, finite 값, quality 존재 여부를 검사한 뒤 status=1인 call만 건너뛴다.
`run.lock`은 같은 cache를 사용하는 중복 실행을 막는다.

```bash
python mission-1/ssl/wav2vec2_frozen/run_wav2vec2_frozen.py --mode extract --device mps
python mission-1/ssl/wav2vec2_frozen/run_wav2vec2_frozen.py --mode status
python mission-1/ssl/wav2vec2_frozen/run_wav2vec2_frozen.py --mode evaluate
python mission-1/ssl/wav2vec2_frozen/verify_results.py
```

진행 로그는 기본 10 call마다 `completed/total`, 현재 실패, pending/retry, 누적 시간과
ETA를 출력한다. cache의 `progress.json`에도 call마다 atomic replace로 저장한다. WAV
open/I/O를 포함한 어떤 call 오류도 `failures.jsonl`에 fsync한 뒤 status=2로 표시하고
즉시 전체 추출을 중단한다. 외장하드를 복구한 뒤 같은 `extract` 명령을 재실행하면 해당
call을 다시 시도한다. `status` 명령은 model이나 WAV를 열지 않고 저장된 진행률만 읽는다.

모델 파일은 기본적으로 `/private/tmp/ddc-hf-cache`에 저장한다. `--model-path`로 로컬
snapshot을 지정할 수도 있다. 원본 볼륨 경로가 바뀐 경우 `--data-root`에
`Training/`을 포함하는 새 데이터 루트를 주면 manifest 파일을 수정하지 않고 경로를
재구성한다. 통화 embedding cache는 `cache/`의 NumPy memmap이며
완료 상태를 함께 기록해 중단 후 재개할 수 있다. 약 82MiB이고 segment embedding이나
waveform/hidden state는 저장하지 않는다. `mission-1/.gitignore`의 `cache/` 규칙으로
Git에서 제외된다.

15초를 넘는 segment는 M1 8GB의 peak memory를 제한하기 위해 겹치지 않는 chunk로
forward하고 hidden frame 수로 가중 평균한다. 전체 442,639개 중 해당 segment는 49개다.
chunk 경계에서는 encoder 문맥이 끊기므로 full run 결과가 생길 경우 알려진 근사 오차로
해석해야 한다. 16kHz에서 400 sample보다 짧은 입력은 forward가 가능하도록 zero-pad한다.

최종 CUDA 결과는 [results/REPORT.md](results/REPORT.md)에 기록한다.

## Mac 터미널에서 독립 실행

아래 detached 명령은 `nohup`으로 터미널/Codex 연결과 분리하고 `caffeinate -ims`로
프로세스가 끝날 때까지 idle system sleep과 disk idle을 막는다. stdout/stderr는
`cache/logs/full_extraction.log`에 함께 append된다. 같은 명령을 다시 실행하면 cache
무결성을 검사하고 완료 call은 건너뛴다.

```bash
cd /Users/joeunkyo/Documents/ChatGPT/AIGO
mkdir -p mission-1/ssl/wav2vec2_frozen/cache/logs
nohup caffeinate -ims /private/tmp/ddc-m1-eda-venv/bin/python -u \
  mission-1/ssl/wav2vec2_frozen/run_wav2vec2_frozen.py \
  --mode extract \
  --device mps \
  --model-path /private/tmp/ddc-hf-cache/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8 \
  --data-root "/Volumes/STORYLiNK/대학부 데이터" \
  --split-path mission-1/validation/split_assignments.csv \
  --manifest-path mission-1/validation/manifests/calls.csv \
  --segments-path mission-1/eda/eda_outputs/segments.csv \
  --cache-dir mission-1/ssl/wav2vec2_frozen/cache \
  --results-dir mission-1/ssl/wav2vec2_frozen/results \
  --progress-every 10 \
  >> mission-1/ssl/wav2vec2_frozen/cache/logs/full_extraction.log 2>&1 < /dev/null &
echo $! > mission-1/ssl/wav2vec2_frozen/cache/extraction.pid
```

진행 상태와 로그:

```bash
/private/tmp/ddc-m1-eda-venv/bin/python \
  mission-1/ssl/wav2vec2_frozen/run_wav2vec2_frozen.py --mode status
tail -F mission-1/ssl/wav2vec2_frozen/cache/logs/full_extraction.log
```

`completed_calls=27985`, `failed_calls=0`을 확인한 후 classifier와 Validation 평가를
별도로 실행한다.

```bash
/private/tmp/ddc-m1-eda-venv/bin/python -u \
  mission-1/ssl/wav2vec2_frozen/run_wav2vec2_frozen.py \
  --mode evaluate \
  --split-path mission-1/validation/split_assignments.csv \
  --manifest-path mission-1/validation/manifests/calls.csv \
  --segments-path mission-1/eda/eda_outputs/segments.csv \
  --cache-dir mission-1/ssl/wav2vec2_frozen/cache \
  --results-dir mission-1/ssl/wav2vec2_frozen/results \
  --mfcc-predictions mission-1/baseline/mfcc_svm/results/val_predictions.csv
/private/tmp/ddc-m1-eda-venv/bin/python \
  mission-1/ssl/wav2vec2_frozen/verify_results.py
```

# Frozen Wav2Vec2: Colab extraction → local evaluation

## 완료된 L4 실행 (2026-09-19)

27,985통화·442,639 caller segment를 동일한 NVIDIA L4 CUDA 환경에서 모두 추출했다.
실패 call/segment, NaN/inf는 0건이며 cache 감사가 통과했다. 실제 추출 및 감사 시간은
11.187423시간이었다. 20-call benchmark의 약 1.54시간 추정은 Drive에 수만 개 파일과
진행 상태를 반복 기록하는 실제 비용을 과소평가했다. 이후 로컬 Train-only
StandardScaler + LogisticRegression 평가에서 Validation 5,597통화 Accuracy는
97.248526%였다. 상세 수치와 재현 절차는 [결과 보고서](../results/REPORT.md)에 있다.

실행 당시 runner SHA256은 `b39a174e03f042bfd6ba57a7168654c5e8ad674156ed02c55d417f7fd9d2f757`이다.
모델 로딩 진단의 JSON 변환 줄이 중복되어 있지만 변환은 멱등이며 추론 연산은 바뀌지
않는다. 이미 생성된 Drive cache의 identity를 보존하기 위해 실행 당시 파일을 유지했다.
아래 절차는 새 Colab 세션에서 실험을 재현할 때의 단계별 절차다.

브랜치: `mission1/eunkyo`. 기존 baseline 코드는 수정하지 않고 import한다.
Colab은 CUDA 확인, Drive 접근, 최소 preflight, 실제 6-call smoke, 20-call benchmark,
사용자 승인 후 full 27,985-call embedding extraction만 담당한다.
전체 embedding 검증, Train-only StandardScaler + LogisticRegression, Validation 5,597 calls 평가,
Majority/F0/MFCC 비교와 오류 겹침 분석, REPORT/README/GitHub 정리는 로컬/Codex에서 진행한다.

## 효율 개선 절차 (2026-09-17)

이미 통과한 clone/업로드/Drive mount/dependency 설치는 반복하지 않는다. 아래 각 단계는
사용자가 결과를 전달하고 확인한 뒤에만 진행한다. `ColabSession`은 full extraction 메서드가 없다.

1. 기존 config로 `session = ColabSession(CONFIG_PATH)`를 만든다. CSV 전체 검증/해석은 한 번 수행.
2. `preflight_report = session.preflight()` — 실제 smoke+benchmark 합집합 **22 WAVs**만 검사.
   예상 1~5분. 진행 중 WAV 번호를 직접 출력한다. 20개라는 이전 안내는 잘못된 수치다.
3. `smoke_report = session.smoke()` — 기존 6 calls. 첫 모델 다운로드 포함 예상 2~10분.
4. `benchmark_report = session.benchmark()` — 같은 frozen 모델 재사용, 기존 20 calls를 모두 재추출.
   smoke embedding을 재사용해 benchmark 시간을 줄이지 않는다. 예상 1~5분, 실제 값은 측정 필요.
5. Benchmark 검토 후 STOP. 사용자 승인 전 full extraction을 실행하지 않는다.

매 단계 CSV/code 내용 hash는 다시 확인하지만 100만 CSV 행을 매번 해석하지 않는다.
Python 직접 호출로 Colab subprocess 출력 누락 문제를 피한다. 설치 전 NumPy가 이미 import되어
버전이 다르면 Python 세션 재시작을 요구한다. 런타임 파일/config를 삭제하거나 새로 업로드하지 않는다.

공개 모델 cache만 `/content/wav2vec2_hf_cache`에 저장한다. 세션이 삭제되면 다시 다운로드할 수 있다.
embedding/cache/progress/failures는 기존 Drive OUTPUT_DIR에 계속 저장한다.
Benchmark의 기존 extraction 시간은 유지하고 추가로 WAV open/read 시간, 나머지 시간,
20 calls의 실제 Drive NPZ commit+progress 저장 시간을 따로 기록한다. `benchmark_storage_probe/`는
진단 전용이며 full cache 완료 상태로 가져오지 않는다. `audio_plus_cache_hours`는 Drive 저장비용을
더한 계획용 추정이다. 전체 cache 최종 감사/세션 중단 비용은 포함하지 않는다.
preflight가 일부 파일을 먼저 읽으므로 Drive cache에 의한 낙관적 추정 가능성도 고려한다.

RAM peak는 재사용 Python 세션의 누적 peak이므로 기존 subprocess RAM과 엄밀히 같은 측정 범위가 아니다.
GPU peak는 각 subset 시작 시 reset한다. `non_read_seconds`는 GPU 전용 시간이 아니라
crop/resampling/CPU-GPU 전송/forward/pooling/출력을 모두 포함한다.

Drive I/O가 느리면 로컬 디스크 WAV staging/사전 읽기를, 추출 자체가 느리면 더 빠른 GPU를
검토하되 측정 결과를 본 뒤 결정한다. 지금은 전체 WAV 복사/새 실험/연산 변경을 추가하지 않는다.
추후 별도 full subprocess를 시작하기 전 `session.release_model()`로 notebook GPU 모델을 해제한다.

## Notebook 실행 범위와 전체 실험 상태

`Frozen_Wav2Vec2_Colab.ipynb`를 Colab에 업로드하고 GPU 런타임에서 단계별로 실행한다.
Notebook 자체의 범위는 CUDA 확인 → Drive 접근 → 최소 preflight → smoke → benchmark까지다.
각 단계 결과를 전달하고 정상임을 확인한 뒤 다음 셀을 실행한다. 전체 실행을 누르지 않는다.
Notebook에는 full extraction 실행 셀이 없으며, 모두 실행하더라도 benchmark에서 멈춘다.
Full extraction은 benchmark 검토와 사용자의 명시적 승인 후 사용자가 직접 시작한다.
로컬/Codex가 장시간 extraction을 실행하거나 세션을 유지할 필요는 없다.
실제로는 별도 full CLI 실행과 로컬 평가까지 완료했으며 최종 결과는 `../results/full_l4/`에 있다.

## 코드 전달과 경로

GitHub clone 후 `mission1/eunkyo` checkout. 코드가 저장소에 반영되기 전 기존 Colab
실행에서는 `artifacts/colab_code_bundle.zip`을 repo 구조 위에 풀었다. 새 clone에서는
저장소에 반영된 코드를 직접 사용할 수 있다. 단, 이미 생성된 full cache를 재개할 때는
runner의 SHA256과 모든 identity가 기존 cache와 정확히 같아야 한다.
`python colab/build_bundle.py`로 ZIP을 재생성한다. ZIP에는 코드/기존 subset fixture/fixed split만
포함한다. 원본 WAV, model weights, 대용량 embedding과 CSV는 포함하지 않는다.

Dependencies는 Colab의 CUDA PyTorch를 유지하고 나머지를 기존 버전으로 고정한다.
설치 오류 시 임의 버전 변경 없이 오류를 전달한다. 실제 실행 버전은 결과에 기록한다.

Drive mount 후 실제 폴더를 탐색하고 아래 경로를 입력한다. 공유 폴더도 실제 mount 경로를
확인한다. DATA_ROOT에는 manifest의 wav_relative_path를 붙여 WAV가 되는 경로를 지정한다.

| 설정 | 내용 |
|---|---|
| DATA_ROOT | Drive 원본 WAV 루트; 그 아래 기존 Training/... 구조 유지 |
| SEGMENTS_CSV | 기존 mission-1/eda/eda_outputs/segments.csv 사본 |
| MANIFEST_CSV | 기존 mission-1/validation/manifests/calls.csv 사본 |
| SPLIT_CSV | 기존 mission-1/validation/split_assignments.csv; 기본 clone 경로 |
| OUTPUT_DIR | 본인 Drive의 새로운 CUDA 전용 결과 폴더 |

Mac의 `/Volumes/STORYLiNK` 경로는 사용하지 않는다. Mac embedding/cache 또는 다른 CUDA 환경의
embedding을 OUTPUT_DIR에 복사하거나 합치지 않는다. 전체 WAV를 RAM에 모으지 않는다.

## 실험 정의 유지

`../run_wav2vec2_frozen.py`의 load_jobs, load_encoder, encode_call 및 내부 crop/resampling/
encode_segment/aggregation 함수를 그대로 사용한다.

- facebook/wav2vec2-base revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
- Frozen encoder, FP32, trainable parameters=0, 768 dimensions
- 8→16kHz scipy resample_poly(up=2, down=1, Kaiser beta=5)
- caller만 사용, 마지막 hidden state temporal mean, segment 동일 가중 call mean
- 기존 15초 초과 구간 분할/frame-weighted pooling과 400 samples 미만 padding 유지
- batch size=1, TF32 off, padded batching/autocast/FP16/BF16 사용 안 함
- split SHA256: 04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06
- Train/Validation=22,388/5,597, 27,985 calls, 442,639 caller segments
- 로컬 평가 시 Train-only StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=1000,
  class_weight=None, random_state=42); 실제 평가 진입점은 `../evaluate_colab_export.py`

CUDA와 MPS 간 bitwise 동일성을 주장하지 않는다. 전체 결과는 모두 동일한 CUDA 환경에서 생성한다.
GPU 이름/compute capability/CUDA/cuDNN/driver/라이브러리 버전/입력 CSV/코드 hash를 cache identity에
포함한다. 재개 시 하나라도 다르면 차단한다. GPU의 물리적 UUID와 순간 여유 메모리는 제외하여
같은 종류와 소프트웨어 환경의 GPU로 세션을 재개할 수 있게 한다. 환경이 달라지면 임의로 metadata를
편집하지 말고 결과를 전달한다. 기존 환경을 복원하거나 새 전체 cache로 시작해야 한다.

## 최소 preflight와 smoke/benchmark

기존 loader의 전체 CSV checksum/coverage/label/segment count 검증을 재사용한다.
WAV open/read 검사는 기존 smoke/benchmark 대상 합집합에 한정한다. 결과에 selected_wav_calls,
checked_calls, checked_call_ids와 검사 범위를 기록한다. 8kHz mono/crop bounds/finite samples/
missing WAV/I/O error를 검사한다. 이 단계는 전체 WAV의 무오류를 보장하지 않으며 나머지는
full extraction 중 기존 encode_call이 읽고 검사한다.

CUDA unavailable이면 중단한다. model load 후 실제 CUDA device/FP32/frozen을 검사한다.
Smoke/benchmark는 기존 저장된 call ID와 label/partition/구간 수/길이를 대조한다.
원래 encode_call의 embedding만 추출하며 Colab에서 classifier를 fit하거나 accuracy를 만들지 않는다.
실패하면 call ID와 0-based segment index/구간/I/O error 여부를 failures/에 저장하고 중단한다.
WAV open 실패는 segment index=-1이며 미시도 segment를 실패로 세지 않는다.

단계 성공 기록은 동일 identity에서만 재사용한다. 코드/환경/CSV가 바뀌면 preflight부터 다시 실행한다.
동일 경로의 WAV 내용을 바꾼 경우에도 다시 preflight부터 실행한다.

Benchmark는 기존 20 calls / 336 segments / 655.775초 audio와 비교한다.
Mac runtime=83.455510625초, 4.172775531초/call, 0.248379496초/segment, RTF=0.127262416.
약 33.13시간은 전체 caller audio × RTF 추정이다. Colab도 audio/call/segment 기반 예상 시간을
따로 출력한다. runtime은 WAV 읽기+embedding 추출이며 모델 로드는 제외한다.
RAM은 실행 프로세스의 peak RSS이며 session 재사용 시 누적 peak다. GPU는 모델 포함 allocated/reserved peak이다.
Drive cache 쓰기 지연, session 제한 등으로 실제 전체 시간은 추정과 달라질 수 있다.
기존 resource_assessment.json은 실제 smoke 이전 기록이며 이후 real-data smoke/benchmark를 기준으로 한다.

## 승인 이후 full extraction의 저장/복구

별도 CLI의 `--stage extract`와 `--confirm-full-extraction`을 모두 명시해야 실행할 수 있다.
같은 identity의 preflight/smoke/benchmark 성공도 필요하다. 완료된 실험도 이 별도 CLI로 실행했다.

- OUTPUT_DIR/call_cache 아래 call ID 앞 두 자리로 폴더를 나누어 NPZ를 저장한다.
- NPZ마다 call_id, FP32 (768,) embedding, quality, identity hash를 저장한다.
  임시 파일 close/fsync → rename → 재읽기 검사 뒤 완료로 인정한다.
- 재개할 때 완료 NPZ의 identity/call ID/shape/dtype/finite/segment 수를 검증하고 건너뛴다.
  커밋 전 남은 .tmp 파일은 완료로 취급하지 않으며 해당 call을 재시도한다.
- progress.json을 매 call 갱신한다. 완료 상태의 원본은 검증된 NPZ이므로 progress 저장 직전에
  종료되어도 완료 call을 재추출하지 않는다. 오류는 기록 후 중단한다.
- 같은 OUTPUT_DIR에 여러 Colab 세션을 동시에 연결하지 않는다. 강제 종료로 ACTIVE_SESSION.lock이
  남으면 이전 세션 종료를 확인한 후 lock만 삭제하고 같은 환경/config로 재개한다.
- Drive FUSE의 원격 동기화는 로컬 트랜잭션과 다르므로 정상 종료 후 Drive 저장 반영을 확인한다.
- 종료 직전 모든 완료 NPZ를 다시 읽어 정확한 coverage/shape/dtype/finite/segment 수를 검증한다.
  embedding_index.csv에 call_id/row_index/gender/partition/cache 상대경로를 저장한다.
  extraction_summary.json에는 완료/미완료/실패 call 수, 성공/실패 segment 수,
  [27985,768] shape, NaN/inf 개수, identity, 과거 실패·복구 시도 통계를 저장한다.
- 실패 로그는 성공적으로 재개한 뒤에도 남긴다. 최종 실패 0과 과거 실패 시도 수를 구분한다.

## 전체 추출 완료 후 로컬 전달

실제 실행에서는 Drive의 call별 NPZ를 검증하면서 통합한
`wav2vec2_l4_full_embeddings.npz`를 로컬로 전달했다. 여기에는 embeddings, call_ids,
gender, partition, n_segments, identity_json, summary_json이 포함된다. 별도의 작은 audit
ZIP으로 index·summary·진행 기록·smoke/benchmark·코드를 함께 검토했다. 원본 call_cache는
Drive에 보관하고 HF model weights와 WAV는 다운로드하지 않았다.

로컬에서는 `../evaluate_colab_export.py`가 export SHA256, call 순서·성별·fixed split,
FP32 shape·finite 값을 확인한 뒤 기존 설정의 Scaler/LR을 학습·평가했다. 명령과 환경은
[Wav2Vec2 README](../README.md)의 로컬 재현 절차를 따른다. Colab cache는 초기 Mac
memmap/evaluate CLI에 직접 넣지 않는다.

최종 비교 기준: Majority 53.189209%, F0+Acoustic LR 91.209577%, MFCC RBF SVM 95.068787%.
같은 Validation에서 Overall/Male/Female accuracy, confusion matrix, MFCC 오류 겹침을 비교한다.
모든 검증 이후에만 GitHub 정리하며 raw WAV/weights/embedding cache는 제외한다.

## 로컬 검증

`python colab/test_colab_runner.py`로 GPU 다운로드/실제 데이터 추출 없이 CUDA 차단, gate,
기존 대상 선택, 최소 WAV 검사 범위, classifier 미실행, CUDA 환경 identity, 중단/재개,
완료 cache 검증을 테스트한다. Notebook 코드 셀은 문법 검사한다. 테스트에는 가짜 embedding만
사용한다. 실제 Colab GPU/Drive 동작과 성능은 사용자가 실행한 결과로 확인한다.

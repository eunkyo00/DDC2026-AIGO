# Mission 1 — Frozen Wav2Vec2 최종 결과

## 결론

동일한 고정 Internal Validation 5,597통화에서 Frozen Wav2Vec2 + Logistic Regression은
**97.248526% (5,443/5,597)**를 기록했다. 기존 최고 MFCC + RBF SVM의
95.068787%보다 **2.179739%p** 높다. 전체 27,985통화의 embedding은 모두 동일한
Colab NVIDIA L4 CUDA 환경에서 생성했으며 Mac MPS embedding은 섞지 않았다.

| 방법 | Overall | Male | Female |
|---|---:|---:|---:|
| Majority (Female 고정) | 53.189209% | 0.000000% | 100.000000% |
| F0 + Acoustic + Logistic Regression | 91.209577% | 89.961832% | 92.307692% |
| MFCC + RBF SVM | 95.068787% | 94.809160% | 95.297279% |
| Frozen Wav2Vec2 + Logistic Regression | **97.248526%** | **96.755725%** | **97.682230%** |

## 고정 데이터와 모델 정의

- fixed split SHA256: `04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06`
- Train / Validation: **22,388 / 5,597** calls; 전체 caller segment **442,639**
- manifest SHA256: `dc57d0f7bea6e06017ac3e27444a0ff5bb299e401ea787400e84cbdf7d072e23`
- segments SHA256: `8bf8c83c0190232e4500b4067bc28d2611b09dd227a7ce4dc9c3741e6dcce584`
- checkpoint: `facebook/wav2vec2-base`, revision `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8`
- 8kHz mono caller crop → 기존 Kaiser-5 `resample_poly` 16kHz → frozen FP32 encoder
- 마지막 hidden state의 시간축 평균 → segment 동일 가중 call 평균, **768차원**
- batch size 1, autocast·TF32 꺼짐, encoder trainable parameter 0
- Train-only `StandardScaler` → `LogisticRegression(C=1, solver=lbfgs,
  max_iter=1000, class_weight=None, random_state=42)`

15초보다 긴 segment의 기존 chunk 처리와 400 sample 미만 padding도 원래 구현을
사용했다. 다른 모델, 증강, ensemble, threshold tuning, hyperparameter search를
추가하지 않았다.

## Colab 추출 검증과 시간

L4 preflight는 smoke·benchmark 합집합 22개 WAV의 8kHz, mono, 접근·읽기를
검사했다. 이는 전체 WAV 내용 감사가 아니다. 6-call smoke와 20-call benchmark는
실제 WAV/caller crop/resampling/CUDA forward 및 finite embedding을 검증했다.

| 단계 | 통화 / segment | 결과 및 소요시간 |
|---|---:|---:|
| 6-call smoke | 6 / 142 | 1.373392초, `(6, 768)`, 실패 0 |
| 20-call L4 benchmark | 20 / 336 | 3.451937초, 0.172597초/call, RTF 0.005264 |
| benchmark 기반 full 예상 | 27,985 / 442,639 | 약 1.540206시간 |
| **실제 full extraction + cache 감사** | **27,985 / 442,639** | **11.187423시간** |

Benchmark의 20개 WAV는 사전에 읽은 소수 파일이며 Drive read/cache, 통화별 NPZ
fsync·재읽기, 매 통화 progress 원자적 저장 및 마지막 27,985개 파일 전수 감사 비용을
대표하지 못했다. 따라서 초기 1.54시간 추정은 실제 실행 시간을 크게 과소평가했다.
향후 작업의 시간 계획에는 **11.19시간 실측값**을 사용해야 한다.

`extraction_summary.json`의 결과는 완료 **27,985/27,985**, 실패 call/segment/I/O
**0/0/0**, embedding shape **(27985, 768)**, dtype float32, NaN/inf **0/0**,
call ID 완전성 통과다. 원본 call별 cache와 통합 export는 Google Drive에 보존한다.
통합 파일 `wav2vec2_l4_full_embeddings.npz`는 **77.53MiB**, SHA256
`64cd8d608c5e7e18a0db86193f7b4b26fa08d689aa22e4dab6807b7360bd71c4`다.
로컬에서 export, fixed split, call 순서, 성별, partition, segment 합계, finite 값을
다시 전수 검증했다.

추출 identity의 adapter SHA256은
`b39a174e03f042bfd6ba57a7168654c5e8ad674156ed02c55d417f7fd9d2f757`다.
Colab 실행 중 JSON 로딩 진단을 직렬화하는 동일한 변환 줄이 중복 삽입되었다.
이 변환은 멱등이며 모델 가중치·전처리·추론 함수는 바뀌지 않았다. 저장소의 runner는
cache 재개 시 identity를 유지하도록 실행 당시 SHA256과 일치한다.

## 로컬 분류와 오류 분석

Train 22,388통화로만 scaler와 LR을 fit했다. Python 3.12.2, NumPy 2.2.2,
SciPy 1.18.1, scikit-learn 1.9.1에서 LR이 **251회 반복**, fit **2.996초**에
수렴했다. 저장한 모델을 다시 로드하여 모든 validation 예측을 재현했다.

| 실제 성별 | 통화 | 맞음 | Accuracy |
|---|---:|---:|---:|
| Male | 2,620 | 2,535 | 96.755725% |
| Female | 2,977 | 2,908 | 97.682230% |

Confusion matrix (행=실제, 열=예측):

|  | Male | Female |
|---|---:|---:|
| Male | 2,535 | 85 |
| Female | 69 | 2,908 |

동일 call ID의 MFCC validation 예측과 직접 대조했다.

| 분류 결과 | 통화 |
|---|---:|
| 둘 다 맞음 | 5,261 |
| Wav2Vec2만 맞음 | 182 |
| MFCC만 맞음 | 60 |
| 둘 다 틀림 | 94 |

Wav2Vec2 오답은 **154개**, MFCC 오답은 **276개**다. 통화 단위 성별 분류에서
사전학습 표현의 이점이 관찰되었지만, 이 결과만으로 사람 단위 일반화나 다른 지역·
녹음 환경에서 같은 정확도를 보장하지 않는다. person ID/phone hash가 없어 같은
신고자가 두 partition에 포함됐는지 확인할 수 없다. caller/operator 라벨 시간이
겹친 통화의 실제 동시 발성 여부 역시 이 결과로 판정할 수 없다.

## 재현과 저장소 산출물

Colab 환경·Drive 설정과 stage 순서는 [Colab README](../colab/README.md)에 있다.
전체 추출은 runner의 `--stage extract --confirm-full-extraction`으로 별도 시작한다.
같은 cache 재개에는 실행 identity의 code/CSV/GPU 정보가 정확히 일치해야 한다.
통합 export를 로컬에 내려받은 뒤 다음 명령으로 CPU 분류를 재현한다.

```bash
python3 -m venv /path/to/local-eval-venv
/path/to/local-eval-venv/bin/python -m pip install \
  -r mission-1/ssl/wav2vec2_frozen/evaluation_requirements.txt
/path/to/local-eval-venv/bin/python \
  mission-1/ssl/wav2vec2_frozen/evaluate_colab_export.py \
  --embeddings /path/to/wav2vec2_l4_full_embeddings.npz
```

이 스크립트는 export SHA256과 fixed split 일치 여부를 확인하고
`results/full_l4/metrics.json`, `val_predictions.csv`, `export_validation.json`을
작성한다. 재실행할 때는 기존 결과를 덮지 않도록 `--output`에 새 디렉터리를 지정한다.
학습 모델 joblib, raw WAV, checkpoint weights, 통화별/통합 embedding cache는
GitHub에서 제외한다. Validation 예측과 metrics만 공개 가능한 크기로 보존한다.

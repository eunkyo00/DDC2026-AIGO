# Mission 1 고정 내부 Validation v1

Training 정상 파싱·WAV 매칭 통화 **27,985개**만 사용한다. 공식 Validation은 배정에 사용하지 않는다.
Python 3.12 표준 라이브러리만 필요하다. 학습·음향 특징 추출 의존성이 없다.

## 실행

저장소 루트에서 최초 생성:

```bash
python3 mission-1/validation/create_split.py
```

기본 입력은 `mission-1/eda/eda_outputs/`, WAV 루트는 그 안의 `run_metadata.json`에서 읽는다.
`--eda-dir`, `--data-root`, `--out-dir`로 다른 환경을 지정할 수 있다.
기존 split 산출물이 있으면 덮어쓰지 않고 실패한다. 재실행할 때는 다음을 사용한다.

```bash
python3 mission-1/validation/create_split.py --verify
python3 mission-1/validation/test_split.py
```

seed **42**, gender별 **80:20**, SHA256 기반 고정 의사무작위 순서다. 정확한 문자열 조합은 코드와 REPORT에 있다.
현재 cohort의 각 gender 수가 5의 배수이므로 정수 quota가 정확하다. 다른 cohort를 자동 수용하지 않는다.
파일 순서를 뒤집어도 동일 배정인지 실제 생성/검증마다 검사한다.

## 파일

| 파일 | 역할 |
|---|---|
| `create_split.py` | 입력 검증, 고정 배정, 분포·누수 확인, majority 산술 계산 |
| `report.py` | summary/checks에서 REPORT 생성 |
| `test_split.py` | 합성 데이터 기반 실패 조건·재현성 검증 |
| `split_assignments.csv` | 공유 가능한 고정 call_id/gender/partition 배정표 |
| `manifests/calls.csv` | 27,985개 call-level 표; 로컬 경로 포함 |
| `manifests/train_calls.csv` | 내부 Train 통화 목록 |
| `manifests/val_calls.csv` | Internal Validation 통화 목록 |
| `split_summary.json` | 설정·통계·Baseline 0·분포 비교 |
| `validation_checks.json` | 누수·정합성·재현성 검사 결과와 한계 |
| `split_metadata.json` | 입력/출력 SHA256·코드 hash·버전·생성 환경 |
| `REPORT.md` | 결과와 후속 실험 사용 계약 |

`manifests/`는 기존 루트 `.gitignore`에서 제외하는 로컬 재생성 데이터다.
공유할 고정 배정은 `split_assignments.csv`에 있으며, 원본 ID 대신 EDA의 가명 file_ref를 사용한다.
다른 컴퓨터에서는 별도 `--out-dir`에 생성한 뒤 공유 배정표의 SHA256과 일치하는지 확인한다.
절대 경로가 다른 통화 목록의 SHA256은 달라질 수 있지만 배정표는 같아야 한다.

## 기존 인덱스 재사용

- `call_id = calls.csv.file_ref = SHA256(NFC(WAV stem))`
- `wav_path`: Training 파일 목록의 stem을 같은 방법으로 해시해 연결한다. 오디오/JSON 내용은 읽지 않는다.
- `n_segments`, `total_duration`: **신고자 segment 수와 단순 길이 합(초)**. `segments.csv`를 스트리밍해 재검증한다.
- `reporter_union_duration`: 기존 EDA 합집합 시간. 음성 결합이나 새로운 전처리를 수행하지 않는다.
- 상담원 포함 구간은 `n_all_segments`, `all_segment_duration`으로 구분한다.
- 별도 utterance index를 만들지 않는다. 후속 loader는 EDA `segments.csv.file_ref`와 고정 배정표 `call_id`를
  many-to-one join한 뒤 `speaker=1`을 선택한다. 배정표에 없는 call은 이 버전의 cohort 밖이다.
- `_id/recordId/audioPath`의 원문은 EDA에 없으므로 EDA의 전역 고유성 검사를 근거로 재사용한다.
  source SHA256은 이를 고정하기 위한 것이며 원본 WAV의 현재 무결성을 새로 검사했다는 뜻은 아니다.

후속 모델은 기존 baseline 코드의 기본 공식 Validation 설정을 그대로 실행하지 말고,
이 고정 배정표를 명시적으로 연결해야 한다. 이번 단계에서는 기존 baseline을 수정하지 않았다.

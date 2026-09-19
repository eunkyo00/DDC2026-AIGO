# Mission 2 데이터 구조 및 EDA

## 목적과 범위

서울 지역 Training/Validation의 WAV·JSON 매칭, WAV 헤더, 발화 역할 분포와 발화 길이를 점검한다. 모델 학습이나 정확도 평가는 하지 않는다. `speaker`는 집계·라벨 검사에만 사용하며, `text`·발화 순서·파일명을 모델 특징으로 만들지 않는다.

## 재실행

저장소 루트에서 `Training/`과 `Validation/`을 직접 포함하는 원본 데이터 폴더를 지정한다.

```bash
python mission-2/eda/test_eda.py
python mission-2/eda/run_eda.py \
  --data-root "/path/to/대학부 데이터" \
  --out-dir "/path/to/local/mission2_eda_results"
```

`run_eda.py`는 Python 표준 라이브러리로 CSV·JSON을 생성한다. PNG 그래프를 만들려면 `matplotlib`이 추가로 필요하다. 기본 `--out-dir`은 현재 작업 디렉터리의 `mission2_eda_results`이며, 이미 있는 같은 이름의 결과 파일을 갱신하므로 별도 로컬 경로를 권장한다. `--limit-calls`는 부분 점검용이며, 그 결과를 전수 EDA로 해석하면 안 된다. `--skip-wav-headers`는 헤더 검사를 생략한다.

## 검사 방법과 한계

- split별 WAV/JSON 파일명을 Unicode NFC로 정규화해 매칭하고 `._*`, `__MACOSX` 보조 파일을 제외한다.
- JSON의 `utterances[]`에서 `speaker`, `startAt`, `endAt`만 읽어 유효성·역할·길이를 집계한다. 시간 단위는 ms에서 초로 변환한다.
- WAV의 sample rate·채널·sample width·길이는 **헤더**로 검사한다. 전체 오디오를 청취하거나 끝까지 디코딩한 검사는 아니다.
- 통화별 출력의 `call_ref`는 stem의 SHA-256이다. 가명 참조일 뿐 익명성 보증이 아니므로 통화별 CSV는 저장소에 포함하지 않는다.
- 이 EDA는 통화 역할 라벨과 음성 특징의 통계만 확인한다. 동일 인물의 split 간 재등장이나 실제 겹침 발성 여부는 확인하지 않는다.

## 공개 산출물

| 파일 | 내용 |
|---|---|
| [REPORT.md](REPORT.md) | 전수 EDA의 주요 수치와 해석 범위 |
| [results/summary.json](results/summary.json) | 원본 경로·파일명 없는 split별 집계 |
| `run_eda.py`, `test_eda.py` | 재실행 코드와 합성 데이터 테스트 |

개별 발화·통화 CSV, 원본 WAV/JSON, 매니페스트와 전처리 특징은 GitHub에 올리지 않는다.

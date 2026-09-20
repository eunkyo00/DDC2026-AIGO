# Mission 3 데이터 배치

`raw/`에 `mission3_train_json.zip`, `mission3_validation_json.zip`을 둔다.
기존 Colab Drive 경로는 `/content/drive/MyDrive/DDC_M3/data/` (DDC 순서 주의).
WAV는 필요 없다. ZIP을 펼치지 않고 읽으며 `__MACOSX`, `._*.json`은 제외한다.
잘못된 JSON은 조용히 제외하지 않고 오류를 발생시킨다. 원래 발화 순서를 유지한다.

`processed/calls/`에 Training 29,200 / 공식 Validation 3,640행을 생성한다.
이 폴더의 원문·라벨·ID는 Git 관리 대상에서 제외한다.

`splits/e0_original/split_manifest.csv`는 E0 백업에서 복원한 원본 배정표다.
`validation/create_split.py --manifest ...`로 데이터 집합 일치·중복·분할 이름을 확인한다.
Mac 로컬 파일은 Colab의 `/content` 경로로 직접 참조할 수 없다.

# STEP 3 추가 sanity check

고정 split과 EDA 파일은 수정하거나 다시 만들지 않았다. 대상 Training JSON 27,985개를 직접 읽었다.

## Stratification

`sklearn.train_test_split`은 사용하지 않았다. seed 42와 gender/call_id로 SHA256 순서를 만들고 gender별 앞 n//5를 Internal Validation으로 배정했다. 나머지가 있으면 오류를 내므로 반올림 손실은 없다. 남 13,100과 여 14,885가 모두 5의 배수이므로 양쪽 비율이 동일하다. 기존 구현에 문제는 없으며 split은 유지했다.

| ID | row | unique | duplicate 초과 row | Train/Validation overlap |
|---|---:|---:|---:|---:|
| _id | 27985 | 27985 | 0 | 0 |
| recordId | 27985 | 27985 | 0 | 0 |
| audioPath | 27985 | 27985 | 0 | 0 |

gender와 전체 발화 경계/역할을 기존 EDA 인덱스에 대조했고 불일치 0건이다. JSON fingerprint는 가명 call_id와 SHA256만 저장했다. 원본 ID·주소·전사 원문은 내보내지 않았다.

## Speaker-level leakage — known unresolved risk

Call-level leakage는 검증 완료했지만 서로 다른 call_id에 동일 신고자가 반복 등장하는 speaker-level leakage 여부는 현재 메타데이터만으로 확인할 수 없다.

신고자 고유 ID, anonymized phone hash, caller/person ID에 해당하는 별도 필드는 발견하지 못했다. `speaker`는 0/1 역할이며 개인 식별자가 아니다. `_id`, `recordId`, `audioPath`의 고유성은 사람 단위 분리 근거가 아니다. ID 인코딩에 숨겨진 업무 의미나 외부 연결표의 존재도 확인되지 않았으며, 억지로 추정하지 않았다.

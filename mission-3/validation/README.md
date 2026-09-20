# 내부 검증 분할 고정

Training 29,200건을 내부 학습 23,408건 / 내부 검증 5,792건으로 나누었다.
공식 Validation 3,640건은 이 분할이나 학습에 사용하지 않는다.

기존 방법: `MultilabelStratifiedShuffleSplit(test_size=0.2, random_state=42)`.
**seed와 행 수만으로 같은 파일이 배정됐다고 증명할 수 없다.**
E0 export의 `split_manifest.csv`를 우선 사용한다. manifest 없이 생성하면 새 분할로 기록하고
과거와 완전히 동일하다고 주장하지 않는다. manifest 행 순서를 예측 순서로 사용한다.

내부 검증은 모델·threshold·앙상블 선택에 반복 사용한 개발 데이터이며 독립 최종 테스트가 아니다.
예측 NPZ의 ID·정답·클래스 순서가 다르면 혼합을 거부한다.

# TF-IDF baseline

문자 n-gram 2~5, min_df=2, 최대 150,000차원, sublinear TF를 사용한다.
OneVsRest LogisticRegression(C=2, liblinear, max_iter=1000)을 내부 학습에만 fit한다.
고정 내부 검증 예측, 모델, 증상별 threshold와 환경 정보를 지정한 out-dir에 저장한다.
기존 Colab 결과는 threshold 조정 Macro F1 0.6028이다.

실행 명령은 mission-3/README.md 참조. GPU는 필요 없다.

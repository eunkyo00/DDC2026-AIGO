# MFCC + RBF SVM baseline

Mission 1의 고정 call-level split을 그대로 사용해 `speaker=1` segment의 MFCC를 계산하고,
통화 단위 54차원 특징으로 집계해 단일 RBF SVC를 평가한다.

## 실행

```bash
python mission-1/baseline/mfcc_svm/test_mfcc_baseline.py
python mission-1/baseline/mfcc_svm/run_mfcc_baseline.py --workers 6
python mission-1/baseline/mfcc_svm/verify_results.py
```

필요 패키지는 `requirements.txt`에 기록했다. 기본 결과 폴더가 이미 존재하면 덮어쓰지
않고 중단한다. 개발용 `--limit`은 MFCC 추출만 검사하며 SVC를 학습하거나 평가하지 않는다.

MFCC 파라미터, 통화 집계, RBF SVC 자원 판단, Train-only scaler, 품질 그룹과 F0 baseline
비교는 실행 결과의 `results/REPORT.md` 및 `results/metrics.json`에 기록된다.

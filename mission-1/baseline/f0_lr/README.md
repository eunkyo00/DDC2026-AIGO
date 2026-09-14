# F0 + Acoustic Logistic Regression baseline

고정된 `mission-1/validation/split_assignments.csv`만 사용해 신고자(`speaker=1`) segment의
F0와 작은 음향 특징 집합을 통화 단위로 집계하고 Logistic Regression을 평가한다.
새 split을 만들거나 MFCC/SVM/pretrained 모델을 실행하지 않는다.

## 실행

Python 3.12 환경에 `numpy`, `pandas`, `soundfile`, `praat-parselmouth`,
`scikit-learn`이 필요하다.

```bash
python mission-1/baseline/f0_lr/test_f0_baseline.py
python mission-1/baseline/f0_lr/run_f0_baseline.py --workers 6
```

기본 결과 위치는 `results/`이다. 기존 결과 폴더가 있으면 덮어쓰지 않고 중단한다.
재실행 결과를 비교하려면 새로운 `--out-dir`을 지정한다. `--limit`은 feature extractor의
개발 점검용이며 subset 모델을 학습하거나 평가하지 않는다.

특징 정의, F0 필터, 품질 기준, 전처리, 평가 결과와 남은 위험은 실행 결과의
`results/REPORT.md` 및 `results/metrics.json`에 기록된다.

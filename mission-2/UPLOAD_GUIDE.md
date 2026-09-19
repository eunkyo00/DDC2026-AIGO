# Mission 2 GitHub 업로드 안내

## 올릴 위치

팀 저장소 `DDC2026-AIGO`의 **`mission-2/` 폴더**에 이 폴더 전체의 코드·문서·집계 결과를 반영한다. 저장소 루트의 기존 `common/`, `mission-1/`과 사용자 작업 중인 파일은 건드리지 않는다. 현재 준비된 로컬 브랜치는 `codex/mission2-smallstem-90`이며 아직 원격 GitHub에 푸시하지 않았다.

```text
DDC2026-AIGO/
└── mission-2/
    ├── README.md                  문제·데이터·모델·결과 개요
    ├── UPLOAD_GUIDE.md            이 안내
    ├── eda/
    │   ├── README.md              EDA 실행 방법·범위
    │   ├── REPORT.md              EDA 집계 해석
    │   ├── run_eda.py, test_eda.py
    │   └── results/summary.json   비식별 집계
    └── smallstem_resnet18/
        ├── README.md              전처리·학습·추론 재현 방법
        ├── *.py, requirements*.txt 학습·추론 코드와 검사
        ├── DCC_mission2_*.ipynb    기존 Drive 작업용 Colab 노트북
        └── results/
            ├── REPORT.md          90.25% 결과·검증 한계
            ├── metrics.json       기계 판독용 결과 요약
            └── history.json       16 epoch 학습 기록
```

가장 안전한 업로드 방법은 이미 별도 작업 폴더에 커밋된 브랜치를 푸시하고 Pull Request로 `main`에 합치는 것이다. 아래 명령은 **현재 로컬에만 있는 변경사항을 원격 브랜치로 업로드**하므로, 직접 업로드하기로 결정했을 때 실행한다.

```bash
cd "/path/to/DDC2026-AIGO-worktree"
git push -u origin codex/mission2-smallstem-90
```

다른 컴퓨터에서 작업하거나 직접 파일을 옮길 경우에는 **이 작업 폴더의 `mission-2/`만** 팀 저장소의 같은 경로에 복사한다. 로컬 저장소 전체를 새 저장소로 중복 업로드하지 않는다.

## 올리지 않을 것

| 파일/폴더 | 이유 |
|---|---|
| `대학부 데이터/`, WAV/JSON 원본 | 대용량·원본 데이터 재배포 위험 |
| `mission02_features/`, `mission02_features_full/`, 관련 ZIP | 대용량 전처리 배열; 코드로 재생성 가능 |
| `mission02_experiment_smallstem-*.zip` | 내부에 모델 체크포인트 2개 포함 |
| `best_model.pt`, `last_checkpoint.pt` | GitHub용 소스·결과 기록과 별도 보관할 가중치 |
| `manifests/`, 통화별 EDA CSV | 로컬 절대경로·개별 기록이 포함될 수 있음 |
| `mission02.zip`, `mission02_95_experiment.zip` | Colab 실행/후속 실험용 묶음이며 이번 90.25% 공개 폴더가 아님 |

원격 저장소의 `.gitignore`도 `*.pt`, `*.pth`, `*.ckpt`, `manifests/`, `eda_outputs/` 등을 제외한다. 지금은 **검증된 Small-stem 90.25% 기록**만 공개한다. 더 높은 성능의 후속 실험은 실제 체크포인트·로그를 확인한 뒤 별도로 문서화한다.

## 공개 전 확인

```bash
git status --short --branch
python mission-2/eda/test_eda.py
python mission-2/smallstem_resnet18/verify_results.py
```

첫 명령이 깨끗한 브랜치를 보여주고 두 검사가 통과해야 한다. `verify_results.py`는 기록의 **내부 일치**만 확인하며, GitHub만으로 90.25% 성능을 재평가할 수 있다는 뜻은 아니다. 실제 제출에는 Drive 등에 별도로 보관한 `best_model.pt`를 추론 코드에 함께 제공해야 한다.

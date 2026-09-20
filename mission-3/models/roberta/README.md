# E0 / E1 / E3

| 설정 | E0 | E1 | E3 |
|---|---|---|---|
| 모델 | klue/roberta-base | klue/roberta-base | klue/roberta-large |
| 본문 | 처음 510 | 긴 문장 앞255+뒤255 | 긴 문장 앞255+뒤255 |
| 전체 길이 | 512 | 512 | 512 |
| batch / accumulation | 8 / 1 | 8 / 1 | 4 / 2 |
| learning rate | 2e-5 | 2e-5 | 1e-5 |

공통: 4 epoch, warmup 1,000 optimizer steps, weight_decay 0.01, fp16,
multi_label_classification (BCE), seed 42. Large와 Base는 학습률도 달라 모델 크기의 효과만
분리한 실험은 아니다. 과거 seed와 패키지 버전이 완전하게 보존되지 않았다.

짧은 입력은 그대로 유지하고 CLS/SEP를 붙인다. 공식 tokenizer가 BertTokenizer이므로
해당 토큰 ID를 사용한다. 모델 입력에는 텍스트 외 메타데이터를 넣지 않는다.

기존 E1/E3는 save_strategy=no였으나 재개 편의를 위해 재구성 코드는 epoch마다 최신
체크포인트 1개를 저장한다. 평가 대상은 기존처럼 **마지막 epoch**다.
checkpoint는 런타임이 사라지면 함께 사라진다. Drive 공간과 별개로 백업해야 한다.

사전학습 MLM head의 UNEXPECTED / 새 classifier의 MISSING은 예상한 초기화 메시지다.
학습된 모델 복원 때 가중치가 누락되는 오류와는 구분한다.

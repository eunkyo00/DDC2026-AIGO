#!/usr/bin/env python3
"""공개 집계 지표와 history의 내부 일관성을 확인한다.

선택적으로 별도 보관한 best_model.pt를 전달하면 해당 파일과 기록도 대조한다.
원본 Validation 음성을 다시 평가하는 검사는 아니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


RESULTS = Path(__file__).with_name("results")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_number(actual: object, expected: object, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{label}: {actual} != {expected}")


def verify(checkpoint_path: Path | None = None) -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    history_path = RESULTS / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if sha256_file(history_path) != metrics["source"]["history_sha256"]:
        raise ValueError("history.json SHA-256 불일치")
    if not history:
        raise ValueError("history.json이 비어 있습니다")
    best = max(history, key=lambda row: row["valid_accuracy_tuned"])
    if int(best["epoch"]) != int(metrics["best_epoch"]):
        raise ValueError("최고 epoch 불일치")
    validation = metrics["validation"]
    for metric_key, history_key in (
        ("loss", "valid_loss"),
        ("accuracy_at_0_5", "valid_accuracy_0.5"),
        ("accuracy_tuned", "valid_accuracy_tuned"),
        ("threshold", "valid_threshold"),
    ):
        same_number(best[history_key], validation[metric_key], metric_key)
    if best["valid_confusion_matrix"] != validation["confusion_matrix_true_rows_pred_columns"]:
        raise ValueError("혼동행렬 불일치")
    if int(best["valid_samples"]) != int(validation["samples"]):
        raise ValueError("Validation 표본 수 불일치")
    confusion = validation["confusion_matrix_true_rows_pred_columns"]
    if sum(sum(row) for row in confusion) != int(validation["samples"]):
        raise ValueError("혼동행렬 표본 수 불일치")
    same_number(
        (confusion[0][0] + confusion[1][1]) / validation["samples"],
        validation["accuracy_tuned"],
        "confusion-derived accuracy",
    )

    if checkpoint_path is not None:
        if sha256_file(checkpoint_path) != metrics["source"]["checkpoint_sha256"]:
            raise ValueError("best_model.pt SHA-256 불일치")
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["model_name"] != metrics["model"]["name"]:
            raise ValueError("체크포인트 모델명 불일치")
        if str(checkpoint.get("feature_variant", "first_1p5")) != metrics["model"]["feature_variant"]:
            raise ValueError("체크포인트 특징 종류 불일치")
        if int(checkpoint["epoch"]) != int(metrics["best_epoch"]):
            raise ValueError("체크포인트 epoch 불일치")
        same_number(checkpoint["threshold"], validation["threshold"], "checkpoint threshold")
        same_number(checkpoint["validation"]["accuracy_tuned"], validation["accuracy_tuned"], "checkpoint accuracy")
        print("체크포인트·집계 기록 일치 — 원본 음원 재평가는 아님")
    else:
        print("공개 지표·history 내부 일치 — 체크포인트/원본 음원 검증은 아님")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="별도 보관한 best_model.pt 경로")
    args = parser.parse_args()
    verify(args.checkpoint)


if __name__ == "__main__":
    main()

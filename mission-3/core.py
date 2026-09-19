"""Mission 3 shared text-only preprocessing and evaluation utilities."""
import hashlib
import json
from pathlib import Path

TARGETS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output is not empty; choose a new directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_call(data, file_name):
    # No metadata or labels are included in the input text.
    utterances = data["utterances"]
    if not isinstance(utterances, list):
        raise ValueError(f"Invalid utterances: {file_name}")
    texts = []
    for utterance in utterances:
        text = utterance.get("text", "")
        if not isinstance(text, str):
            raise ValueError(f"Non-string text: {file_name}")
        texts.append(text)
    symptoms = data["symptom"]
    if not isinstance(symptoms, list):
        raise ValueError(f"Invalid symptom list: {file_name}")
    return {"file_name": file_name, "text": " ".join(texts),
            **{target: int(target in symptoms) for target in TARGETS}}


def head_tail(ids, limit=510):
    if len(ids) <= limit:
        return ids
    first = limit // 2
    return ids[:first] + ids[-(limit - first):]


def load_split(data_dir, manifest):
    import pandas as pd
    df = pd.read_json(Path(data_dir) / "training.jsonl", lines=True)
    split = pd.read_csv(manifest)
    if df.file_name.duplicated().any() or split.file_name.duplicated().any():
        raise ValueError("Duplicate file names")
    if set(df.file_name) != set(split.file_name):
        raise ValueError("Manifest does not match Training cohort")
    if set(split.split) != {"train", "dev_valid"}:
        raise ValueError("Unexpected split names")
    # Preserve manifest order, including prediction alignment across experiments.
    ordered = split.merge(df, on="file_name", how="left", validate="one_to_one", sort=False)
    return (ordered[ordered.split == "train"].reset_index(drop=True),
            ordered[ordered.split == "dev_valid"].reset_index(drop=True))


def tune(y, probs):
    import numpy as np
    from sklearn.metrics import f1_score
    thresholds, scores = [], []
    for i in range(len(TARGETS)):
        grid = np.arange(0.05, 0.96, 0.01)
        values = [f1_score(y[:, i], probs[:, i] >= t, zero_division=0) for t in grid]
        best = int(np.argmax(values))
        thresholds.append(float(grid[best]))
        scores.append(float(values[best]))
    return thresholds, scores, float(np.mean(scores))


def save_predictions(out, frame, probs):
    import numpy as np
    from sklearn.metrics import f1_score
    y = frame[TARGETS].to_numpy(dtype=int)
    probs = np.asarray(probs)
    if probs.shape != y.shape or not np.isfinite(probs).all():
        raise ValueError("Invalid predictions")
    thresholds, scores, macro = tune(y, probs)
    np.savez_compressed(out / "dev_predictions.npz", file_names=frame.file_name.to_numpy(dtype=str),
                        targets=np.asarray(TARGETS), y_true=y, probs=probs)
    metrics = {"default_macro_f1": float(f1_score(y, probs >= .5, average="macro", zero_division=0)),
               "tuned_macro_f1": macro, "thresholds": thresholds, "targets": TARGETS,
               "per_label_f1": scores, "evaluation": "internal validation; thresholds tuned on same set"}
    write_json(out / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def environment():
    import importlib.metadata
    import platform
    result = {"python": platform.python_version()}
    for name in ["numpy", "pandas", "scikit-learn", "iterative-stratification", "torch", "transformers", "accelerate"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

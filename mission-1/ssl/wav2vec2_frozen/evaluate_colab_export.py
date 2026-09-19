"""Validate a complete Colab export and fit the original CPU-only scaler/LR."""
import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
MISSION = BASE.parents[1]
EXPORT_SHA = '64cd8d608c5e7e18a0db86193f7b4b26fa08d689aa22e4dab6807b7360bd71c4'
SPLIT_SHA = '04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=BASE / 'results/full_l4')
    args = parser.parse_args()
    assert digest(args.embeddings) == EXPORT_SHA, 'Export checksum mismatch'
    split_path = MISSION / 'validation/split_assignments.csv'
    assert digest(split_path) == SPLIT_SHA
    with np.load(args.embeddings, allow_pickle=False) as z:
        x, ids, y, partitions, segments = (
            z[k] for k in ('embeddings', 'call_ids', 'gender', 'partition', 'n_segments'))
        identity = json.loads(z['identity_json'].item())
        summary = json.loads(z['summary_json'].item())
    assert x.shape == (27985, 768) and x.dtype == np.float32
    assert np.isfinite(x).all() and len(set(ids)) == 27985
    assert all(a.shape == (27985,) for a in (ids, y, partitions, segments))
    assert int(segments.sum()) == 442639 and (segments > 0).all()
    assert summary['status'] == 'complete' and summary['identity'] == identity
    assert identity['split_sha256'] == SPLIT_SHA
    assert identity['checkpoint'] == 'facebook/wav2vec2-base'
    assert identity['revision'] == '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8'
    assert identity['dtype'] == 'float32' and identity['batch_size'] == 1
    assert not identity['autocast'] and not identity['tf32']
    assert identity['execution_environment']['gpu_name'] == 'NVIDIA L4'
    assert hashlib.sha256('\n'.join(ids).encode()).hexdigest() == summary['call_order_sha256']
    split_rows = read_rows(split_path)
    split = {r['call_id']: r for r in split_rows}
    assert len(split_rows) == len(split) == 27985 and set(ids) == set(split)
    for c, g, p in zip(ids, y, partitions):
        assert (g, p) == (split[c]['gender'], split[c]['partition'])
    train = partitions == 'train'
    val = partitions == 'internal_validation'
    assert train.sum() == 22388 and val.sum() == 5597
    mfcc_rows = read_rows(MISSION / 'baseline/mfcc_svm/results/val_predictions.csv')
    mfcc = {r['call_id']: r for r in mfcc_rows}
    assert len(mfcc_rows) == len(mfcc) == 5597 and set(mfcc) == set(ids[val])
    for c, g in zip(ids[val], y[val]):
        assert mfcc[c]['gender'] == g
    assert np.__version__ == '2.2.2'
    assert scipy.__version__ == '1.18.1' and sklearn.__version__ == '1.9.1'
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'metrics.json').exists(), 'Preserve existing evaluation; choose a new output'
    save_json(output / 'export_validation.json', {
        'status': 'pass', 'export_sha256': EXPORT_SHA, 'identity': identity,
        'shape': list(x.shape), 'dtype': str(x.dtype), 'nan_inf': 0,
        'split_counts': dict(Counter(partitions.tolist())), 'caller_segments': int(segments.sum()),
        'call_order_sha256': summary['call_order_sha256'],
    })
    print('LOCAL EMBEDDING VALIDATION PASS; fitting original scaler/LR...', flush=True)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                          class_weight=None, random_state=42)),
    ])
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x[train], y[train])
    fit_seconds = time.perf_counter() - started
    assert int(model['scaler'].n_samples_seen_) == 22388
    np.testing.assert_allclose(model['scaler'].mean_, x[train].mean(axis=0, dtype=np.float64),
                               rtol=1e-10, atol=1e-12)
    predictions = model.predict(x[val])
    scores = model.decision_function(x[val])
    assert list(model['classifier'].classes_) == ['F', 'M']
    truth = y[val]
    correct = predictions == truth
    metrics = {
        'overall': {'n': 5597, 'correct': int(correct.sum()), 'accuracy': float(correct.mean())},
        'gender_accuracy': {g: {'n': int((truth == g).sum()),
                               'correct': int(correct[truth == g].sum()),
                               'accuracy': float(correct[truth == g].mean())} for g in ('M', 'F')},
        'confusion_matrix': {'labels': ['M', 'F'], 'rows': 'true', 'columns': 'predicted',
                             'matrix': confusion_matrix(truth, predictions, labels=['M', 'F']).tolist()},
        'fit_seconds': fit_seconds, 'iterations': model['classifier'].n_iter_.tolist(),
        'scaler_fit_calls': int(model['scaler'].n_samples_seen_),
        'classifier': model['classifier'].get_params(), 'decision_score_positive_class': 'M',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__},
        'export_sha256': EXPORT_SHA, 'split_sha256': SPLIT_SHA,
    }
    overlap = Counter()
    rows = []
    for c, g, pred, score in zip(ids[val], truth, predictions, scores):
        mp = mfcc[c]['prediction']
        key = ('both_correct' if pred == g and mp == g else 'wav2vec2_only_correct'
               if pred == g else 'mfcc_only_correct' if mp == g else 'both_wrong')
        overlap[key] += 1
        rows.append({'call_id': c, 'gender': g, 'prediction': pred,
                     'decision_score': float(score), 'mfcc_prediction': mp, 'error_overlap': key})
    metrics['mfcc_error_overlap'] = dict(overlap)
    metrics['mfcc_accuracy'] = sum(r['prediction'] == r['gender'] for r in mfcc_rows) / 5597
    with (output / 'val_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifact_dir = output / 'artifacts'
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump(model, artifact_dir / 'scaler_logistic_regression.joblib')
    save_json(output / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    print('LOCAL EVALUATION PASS', flush=True)


if __name__ == '__main__':
    main()

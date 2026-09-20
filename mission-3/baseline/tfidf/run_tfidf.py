import argparse
from pathlib import Path
import sys
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import ROOT, TARGETS, fresh_dir, load_split, save_predictions, write_json, environment

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/calls")
    p.add_argument("--manifest", type=Path, default=ROOT / "data/splits/fixed/split_manifest.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fixed-threshold", action="store_true",
                   help="Evaluate only at threshold 0.5; do not search label thresholds.")
    args = p.parse_args()
    train, valid = load_split(args.data_dir, args.manifest)
    out = fresh_dir(args.out_dir)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                 max_features=150000, sublinear_tf=True, dtype=np.float32)),
        ("classifier", OneVsRestClassifier(LogisticRegression(C=2., max_iter=1000,
                                                              solver="liblinear", random_state=42), n_jobs=-1))])
    model.fit(train.text.fillna(""), train[TARGETS])
    save_predictions(out, valid, model.predict_proba(valid.text.fillna("")),
                     tune_thresholds=not args.fixed_threshold)
    joblib.dump(model, out / "model.joblib")
    write_json(out / "environment.json", environment())

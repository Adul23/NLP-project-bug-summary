"""Near-duplicate check across stratified_v1 splits.

For every clean function in validation and test, find the most similar clean function in
train (cosine similarity of TF-IDF over code tokens, 1-3 grams). Exact duplicates were
already removed during cleaning; this looks for near copies (renamed variables, small edits).

    python scripts/near_duplicates.py [split_dir] [output_json]
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
SPLITS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/stratified_v1"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/near_duplicates.json"
THRESHOLDS = [0.8, 0.9, 0.95, 0.99]


def clean(split):
    frame = pd.read_csv(SPLITS / f"{split}.csv", dtype={"source_id": str}, keep_default_na=False)
    return frame[frame.label == 0].reset_index(drop=True)


train = clean("train")
vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z_]\w*|\d+|[^\w\s]", ngram_range=(1, 3),
                             lowercase=False, sublinear_tf=True)
train_matrix = vectorizer.fit_transform(train.code)

report = {"method": "max cosine similarity of TF-IDF (code tokens, 1-3 grams) to any train "
                    "clean function", "train_functions": len(train)}
for split in ("validation", "test"):
    frame = clean(split)
    similarity = (vectorizer.transform(frame.code) @ train_matrix.T).max(axis=1).toarray().ravel()
    worst = similarity.argsort()[::-1][:3]
    report[split] = {
        "functions": len(frame),
        "max_similarity_quantiles": {q: round(float(pd.Series(similarity).quantile(q)), 4)
                                     for q in (0.5, 0.9, 0.99, 1.0)},
        "at_or_above": {str(t): int((similarity >= t).sum()) for t in THRESHOLDS},
        "most_similar_source_ids": [frame.source_id[i] for i in worst],
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

"""Show how one Python function becomes model input, and what the model says about it.

    python scripts/demo_tokenize.py --model tfidf --code-file f.py
    printf 'def f(a, b):\\n    return a < b\\n' | python scripts/demo_tokenize.py --model tfidf
    python scripts/demo_tokenize.py --model codebert --model-dir models/codebert_lora_seed42 \\
        --device cuda --code-file f.py          # Colab: needs torch, transformers, peft

CodeBERT: the first BPE tokens and their ids, length before and after truncation, the
attention mask, the bug probability and the localized fix. TF-IDF: the number of non-zero
character n-gram features, the n-grams that push the decision most (coefficient x TF-IDF
weight), the bug probability and the localized fix. The code is never executed.
"""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bug_detection.localize import localize  # noqa: E402


def read_code(path):
    """The function's source from a file, or from stdin when no file is given."""
    code = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    if not code.strip():
        raise ValueError("no code given")
    return code


def _localized(code, predictor, top_k=3):
    try:
        result = localize(code, predictor, top_k=top_k)
    except SyntaxError as error:
        return {"error": f"cannot parse: {error.msg}"}
    return {
        "n_candidates": result["n_candidates"],
        "candidates": [
            {k: c[k] for k in ("line", "old", "new", "mutation_type", "drop")}
            for c in result["candidates"]
        ],
    }


def tfidf_report(code, predictor, top=10):
    """Non-zero features, the n-grams that weigh most in the decision, the prediction."""
    features = predictor.vectorizer.transform([code])
    names = predictor.vectorizer.get_feature_names_out()
    coef = predictor.model.coef_[0]
    contributions = sorted(
        (
            {
                "ngram": names[i],
                "tfidf": float(value),
                "coef": float(coef[i]),
                "contribution": float(value * coef[i]),
            }
            for i, value in zip(features.indices, features.data)
        ),
        key=lambda row: -abs(row["contribution"]),
    )
    intercept = float(predictor.model.intercept_[0])
    logit = sum(row["contribution"] for row in contributions) + intercept
    prediction = predictor.predict(code)
    return {
        "model": "tfidf",
        "n_features": len(names),
        "nonzero_features": int(features.nnz),
        "top_ngrams": contributions[:top],
        "intercept": intercept,
        "logit": logit,
        "sigmoid_of_logit": 1 / (1 + math.exp(-logit)),
        "bug_probability": prediction["bug_probability"],
        "label": prediction["label"],
        "localize": _localized(code, predictor),
    }


def codebert_report(code, predictor, show=40):
    """BPE tokens and ids, truncation, attention mask, the prediction."""
    tokenizer, max_length = predictor.tokenizer, predictor.config["max_length"]
    full = tokenizer(code, truncation=False, return_token_type_ids=False)["input_ids"]
    encoded = tokenizer(code, truncation=True, max_length=max_length,
                        return_tensors="pt", return_token_type_ids=False)
    ids = encoded["input_ids"][0].tolist()
    mask = encoded["attention_mask"]
    prediction = predictor.predict(code)
    return {
        "model": "codebert",
        "vocab_size": tokenizer.vocab_size,
        "tokens": tokenizer.convert_ids_to_tokens(ids[:show]),
        "ids": ids[:show],
        "length_before_truncation": len(full),
        "length_after_truncation": len(ids),
        "max_length": max_length,
        "truncated": len(full) > max_length,
        "attention_mask_shape": list(mask.shape),
        "attention_mask_sum": int(mask.sum()),
        "bug_probability": prediction["bug_probability"],
        "label": prediction["label"],
        "localize": _localized(code, predictor),
    }


def format_report(report):
    lines = []
    if report["model"] == "tfidf":
        lines += [
            f"TF-IDF character n-grams: {report['nonzero_features']} non-zero of "
            f"{report['n_features']} features",
            "Top n-grams by contribution (coef x tf-idf; > 0 pushes towards BUGGY):",
        ]
        for row in report["top_ngrams"]:
            lines.append(f"  {row['ngram']!r:12} tfidf {row['tfidf']:.3f}  coef "
                         f"{row['coef']:+.2f}  contribution {row['contribution']:+.3f}")
        lines.append(f"logit = sum of contributions + intercept ({report['intercept']:+.3f}) "
                     f"= {report['logit']:+.3f}; sigmoid = {report['sigmoid_of_logit']:.3f}")
    else:
        lines += [
            f"CodeBERT BPE (vocabulary {report['vocab_size']}; Ġ = space before the token, "
            "Ċ = newline)",
            f"First {len(report['tokens'])} tokens:",
            "  " + " ".join(report["tokens"]),
            "Their ids:",
            "  " + " ".join(map(str, report["ids"])),
            f"Length: {report['length_before_truncation']} tokens before truncation, "
            f"{report['length_after_truncation']} after (max {report['max_length']}); "
            f"truncated: {'yes' if report['truncated'] else 'no'}",
            f"Attention mask: shape {report['attention_mask_shape']}, sum "
            f"{report['attention_mask_sum']} (1 = real token; 0 appears only as padding in a "
            "batch of different lengths)",
        ]
    lines.append(f"Prediction: {report['label']}, bug probability {report['bug_probability']:.3f}")
    loc = report["localize"]
    if "error" in loc:
        lines.append(f"Localization: {loc['error']}")
    elif not loc["candidates"]:
        lines.append("Localization: no operator to flip")
    else:
        lines.append(f"Localization ({loc['n_candidates']} candidates), top by drop in bug "
                     "probability:")
        for c in loc["candidates"]:
            lines.append(f"  line {c['line']}: `{c['old']}` -> `{c['new']}` "
                         f"({c['mutation_type']}), drop {c['drop']:+.3f}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["tfidf", "codebert"], default="tfidf")
    p.add_argument("--code-file", type=Path, default=None, help="default: stdin")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="default: models/tfidf_v2 or models/codebert_lora_seed42")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tokens", type=int, default=40, help="CodeBERT tokens to show")
    a = p.parse_args(argv)
    code = read_code(a.code_file)

    from src.inference.models import LoraPredictor, TfidfPredictor

    if a.model == "tfidf":
        report = tfidf_report(code, TfidfPredictor(a.model_dir or ROOT / "models/tfidf_v2"))
    else:
        predictor = LoraPredictor(a.model_dir or ROOT / "models/codebert_lora_seed42",
                                  device=a.device)
        report = codebert_report(code, predictor, show=a.tokens)
    print(format_report(report))
    return report


if __name__ == "__main__":
    main()

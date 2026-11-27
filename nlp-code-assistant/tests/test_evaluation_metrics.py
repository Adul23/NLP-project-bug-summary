"""Evaluation metrics: classification, paired tests, retrieval and summarization."""

import math

import numpy as np
import pytest
from src.evaluation import bug_metrics as bm
from src.evaluation import retrieval_metrics as rm
from src.summarization import metrics as sm

# Small example used for hand-computed values:
#   y_true = 1 1 0 0 0, y_pred = 1 0 0 0 1  ->  TP=1, FN=1, TN=2, FP=1
Y_TRUE = [1, 1, 0, 0, 0]
Y_PRED = [1, 0, 0, 0, 1]
Y_PROB = [0.9, 0.4, 0.2, 0.1, 0.6]


def test_known_small_example():
    m = bm.compute_bug_metrics(Y_TRUE, Y_PRED, Y_PROB, n_bootstrap=0)
    assert m["accuracy"] == pytest.approx(3 / 5)
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) = (2 - 1) / sqrt(2*2*3*3)
    assert m["mcc"] == pytest.approx(1 / 6)
    # Kappa: po = 0.6, pe = 0.4*0.4 + 0.6*0.6 = 0.52 -> (0.6 - 0.52) / 0.48
    assert m["cohen_kappa"] == pytest.approx(1 / 6)
    # ROC-AUC: positives (0.9, 0.4) beat negatives (0.2, 0.1, 0.6) in 5 of 6 pairs.
    assert m["roc_auc"] == pytest.approx(5 / 6)
    # Ranking P, N, P, N, N -> AP = (1/1 + 2/3) / 2
    assert m["pr_auc"] == pytest.approx(5 / 6)
    assert m["pr_auc_baseline"] == pytest.approx(2 / 5)
    assert m["balanced_accuracy"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert m["bug_precision"] == m["bug_recall"] == m["bug_f1"] == pytest.approx(0.5)
    assert m["clean_precision"] == m["clean_recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["false_negative_rate"] == pytest.approx(1 / 2)
    assert m["confusion_matrix"]["counts"] == [[2, 1], [1, 1]]
    assert m["confusion_matrix"]["row_normalized"] == [[pytest.approx(2 / 3), pytest.approx(1 / 3)], [0.5, 0.5]]
    calibration = m["calibration"]
    # Brier = (0.1² + 0.6² + 0.2² + 0.1² + 0.6²) / 5
    assert calibration["brier"] == pytest.approx(0.156)
    # Each probability falls in its own bin: |0.9-1|, |0.4-1|, |0.2-0|, |0.1-0|, |0.6-0| averaged.
    assert calibration["ece"] == pytest.approx(1.6 / 5)
    assert sum(b["count"] for b in calibration["reliability_curve"]) == 5
    assert len(calibration["reliability_curve"]) == 10


def test_perfect_classifier():
    y = [0, 0, 0, 1, 1]
    m = bm.compute_bug_metrics(y, y, [0.0, 0.1, 0.2, 0.8, 1.0], n_bootstrap=50)
    for key in ("accuracy", "balanced_accuracy", "bug_f1", "clean_f1", "macro_f1", "mcc", "cohen_kappa", "roc_auc", "pr_auc"):
        assert m[key] == pytest.approx(1.0), key
    assert m["false_positive_rate"] == m["false_negative_rate"] == 0
    assert m["bootstrap"]["ci"]["accuracy"] == {"low": 1.0, "high": 1.0, "resamples": 50}


def test_all_negative_classifier():
    y_true = [1, 0, 0, 0]
    m = bm.compute_bug_metrics(y_true, [0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1], n_bootstrap=0)
    assert m["accuracy"] == pytest.approx(0.75)  # looks fine...
    assert m["balanced_accuracy"] == pytest.approx(0.5)  # ...but it is a coin flip
    assert m["bug_recall"] == m["bug_f1"] == m["bug_precision"] == 0
    assert m["mcc"] == 0 and m["cohen_kappa"] == 0
    assert m["roc_auc"] == pytest.approx(0.5)  # constant scores rank nothing
    assert m["false_negative_rate"] == 1 and m["specificity"] == 1


def test_single_class_and_input_validation():
    m = bm.compute_bug_metrics([0, 0], [0, 1], [0.2, 0.7], n_bootstrap=0)
    assert m["roc_auc"] is None and m["pr_auc"] is None and m["balanced_accuracy"] is None
    assert m["false_negative_rate"] is None  # no positive rows
    with pytest.raises(ValueError):
        bm.compute_bug_metrics([0, 1], [0, 2])
    with pytest.raises(ValueError):
        bm.compute_bug_metrics([0, 1], [0, 1], [0.5, 1.5])
    with pytest.raises(ValueError):
        bm.compute_bug_metrics([], [])


def test_bootstrap_is_deterministic_with_the_seed():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_prob = np.clip(y_true * 0.3 + rng.random(200) * 0.7, 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    first = bm.bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=200, seed=42)
    assert first == bm.bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=200, seed=42)
    assert first != bm.bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=200, seed=7)
    point = bm.compute_bug_metrics(y_true, y_pred, y_prob, n_bootstrap=0)
    for key in bm.CI_METRICS:
        ci = first["ci"][key]
        assert ci["low"] <= point[key] <= ci["high"], key


def test_mutation_breakdown_and_quartiles():
    types = ["clean", "clean", "comparison_operator", "boolean_operator", "boolean_operator"]
    result = bm.by_mutation_type([0, 0, 1, 1, 1], [1, 0, 1, 0, 1], types)
    assert result["comparison_operator"] == {"n": 1, "detected": 1, "recall": 1.0}
    assert result["boolean_operator"] == {"n": 2, "detected": 1, "recall": 0.5}
    assert result["clean"] == {"n": 2, "false_positives": 1, "false_positive_rate": 0.5}
    groups, labels = bm.length_quartiles([10, 20, 30, 40, 50, 60, 70, 80])
    assert len(labels) == 4 and labels[0].startswith("Q1 (10–")
    assert [groups.count(label) for label in labels] == [2, 2, 2, 2]


def test_threshold_is_chosen_from_given_rows_only():
    tuned = bm.best_f1_threshold([0, 0, 1, 1, 1], [0.1, 0.45, 0.42, 0.6, 0.9])
    assert tuned["threshold"] == pytest.approx(0.42) and tuned["validation_bug_f1"] == pytest.approx(6 / 7)
    analysis = bm.threshold_analysis(Y_TRUE, Y_PROB, tuned["threshold"], n_bootstrap=0)
    assert analysis["default_0.5"]["threshold"] == 0.5
    assert analysis["validation_tuned"]["threshold"] == pytest.approx(0.42)
    # At 0.42 the 0.4 positive is still missed; the result is just scored, nothing is re-tuned.
    assert analysis["validation_tuned"]["metrics"]["bug_recall"] == pytest.approx(0.5)


def test_mcnemar_on_a_hand_made_table():
    # 10 rows where only A is right (b), 2 where only B is right (c), 8 where both are right.
    y_true = [1] * 20
    pred_a = [1] * 8 + [0] * 2 + [1] * 8 + [0] * 2
    pred_b = [0] * 8 + [1] * 2 + [1] * 8 + [0] * 2
    result = bm.mcnemar(y_true, pred_a, pred_b)
    assert (result["b"], result["c"]) == (8, 2)
    assert result["table"] == [[8, 8], [2, 2]]
    # Exact: 2 * P(X <= 2 | n=10, p=0.5) = 2 * (1 + 10 + 45) / 1024
    assert result["exact_p"] == pytest.approx(112 / 1024)
    # Continuity-corrected chi² = (|8 - 2| - 1)² / 10 = 2.5 -> p = erfc(sqrt(2.5 / 2))
    assert result["chi2"] == pytest.approx(2.5)
    assert result["chi2_p"] == pytest.approx(math.erfc(math.sqrt(1.25)))
    same = bm.mcnemar([1, 0], [1, 0], [1, 0])
    assert same["exact_p"] == 1.0 and same["b"] == same["c"] == 0


def test_paired_bootstrap_f1_difference():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0] * 10)
    weak = np.array([1, 0, 0, 0, 1, 0, 0, 0] * 10)
    strong = y_true.copy()
    result = bm.paired_bootstrap_f1(y_true, weak, strong, n_bootstrap=300)
    assert result["observed"] == pytest.approx(1 - 2 / 6)  # F1 1.0 vs 2*1/(2+1+3)
    assert 0 < result["ci_low"] <= result["observed"] <= result["ci_high"]
    assert result["share_b_better"] == 1.0
    assert result == bm.paired_bootstrap_f1(y_true, weak, strong, n_bootstrap=300)


# --- retrieval ------------------------------------------------------------------------


def test_retrieval_metrics():
    ranked, relevant = ["a", "b", "c", "d"], {"b", "d", "z"}
    assert rm.precision_at_k(ranked, relevant, 2) == 0.5
    assert rm.recall_at_k(ranked, relevant, 4) == pytest.approx(2 / 3)
    assert rm.hit_at_k(ranked, relevant, 1) == 0 and rm.hit_at_k(ranked, relevant, 2) == 1
    assert rm.reciprocal_rank(ranked, relevant) == 0.5
    assert rm.average_precision(ranked, relevant) == pytest.approx((1 / 2 + 2 / 4) / 3)
    # nDCG@2 with binary gains: DCG = 1/log2(3); ideal = 1 + 1/log2(3)
    assert rm.ndcg_at_k(ranked, relevant, 2) == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert rm.ndcg_at_k(["a"], {"a": 3}, 1) == 1.0
    summary = rm.evaluate_rankings([ranked, ["z"]], [relevant, {"z"}], ks=(1,))
    assert summary["mrr"] == pytest.approx((0.5 + 1) / 2) and summary["hit@1"] == 0.5
    with pytest.raises(ValueError):
        rm.precision_at_k(ranked, relevant, 0)


# --- summarization --------------------------------------------------------------------


def test_rouge_n_on_tiny_examples():
    # ref "the cat sat", hyp "the cat ran": 2 of 3 unigrams, 1 of 2 bigrams match.
    r1 = sm.rouge_n("The cat sat", "the cat ran", 1)
    r2 = sm.rouge_n("The cat sat", "the cat ran", 2)
    assert r1 == {"precision": pytest.approx(2 / 3), "recall": pytest.approx(2 / 3), "f1": pytest.approx(2 / 3)}
    assert r2["precision"] == r2["recall"] == r2["f1"] == pytest.approx(0.5)
    # Different lengths: hyp "the cat" vs ref "the cat sat" -> P = 1, R = 2/3.
    short = sm.rouge_n("the cat sat", "the cat", 1)
    assert short["precision"] == 1 and short["recall"] == pytest.approx(2 / 3)
    assert short["f1"] == pytest.approx(0.8)
    assert sm.rouge_l("a b c d", "a c d")["f1"] == pytest.approx(sm.rouge_l_f1("a b c d", "a c d"))
    assert sm.rouge_n("", "x", 1)["f1"] == 0


def test_chrf_on_tiny_examples():
    assert sm.corpus_chrf(["abc"], ["abc"]) == pytest.approx(1.0)
    assert sm.corpus_chrf(["ab"], ["cd"]) == 0.0
    # Orders 1-2: unigram P = R = 1/2, bigram P = R = 0 -> averages 1/4 -> F2 = 1/4.
    assert sm.corpus_chrf(["ab"], ["ac"], max_n=2) == pytest.approx(0.25)
    # Whitespace is ignored.
    assert sm.corpus_chrf(["a b c"], ["abc"]) == pytest.approx(1.0)


def test_summary_metrics_bundle():
    refs = ["add two numbers", "read a json file", "return the max"]
    hyps = ["add two numbers", "", "def f x return max x"]
    codes = ["def add(a, b): return a + b", "def r(p): return json.load(open(p))", "def f(x): return max(x)"]
    result = sm.summary_metrics(refs, hyps, codes, n_bootstrap=100)
    assert result["examples"] == 3
    assert result["exact_match"] == pytest.approx(1 / 3)
    assert result["empty_output_rate"] == pytest.approx(1 / 3)
    assert result["copy_rate"] == pytest.approx(1 / 3)  # the third output echoes the code
    assert result["avg_reference_tokens"] == pytest.approx(10 / 3)
    assert 0 <= result["chrf"] <= 1 and 0 <= result["rouge_l"]["f1"] <= 1
    assert result["sentence_bleu4_smoothed"] > 0
    assert set(result["bootstrap"]) >= {"bleu4", "rouge_l_f1"}
    assert result == sm.summary_metrics(refs, hyps, codes, n_bootstrap=100)  # seeded
    if result["meteor"] is None:
        assert result["meteor_note"].startswith("METEOR skipped")
    assert sm.sentence_bleu("a b c d", "a b c d") == pytest.approx(1.0)


def test_summarization_evaluation_script(tmp_path):
    import json

    import pandas as pd
    from src.summarization.evaluate_summaries import evaluate

    rows = []
    for split, count in (("train", 12), ("test", 6)):
        for i in range(count):
            rows.append(
                {
                    "source_id": f"{split}{i}",
                    "code": f"def add_{i}(a, b):\n    return a + b + {i}",
                    "summary": f"Add two numbers and {i}.",
                    "split": split,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "data.csv", index=False)
    result = evaluate(
        tmp_path / "data.csv",
        "test",
        ("ast", "tfidf", "bm25", "codet5"),
        model_dir=tmp_path / "no_model",  # CodeT5 missing: skipped with a note, not a crash
        output=tmp_path / "out/summarization_metrics.json",
        human_eval=tmp_path / "out/human_eval.csv",
        human_eval_rows=4,
        n_bootstrap=50,
    )
    assert set(result["systems"]) == {"ast", "tfidf", "bm25"}
    assert result["notes"] and result["notes"][0].startswith("CodeT5 skipped")
    for system in result["systems"].values():
        assert system["examples"] == 6 and 0 <= system["rouge_l"]["f1"] <= 1
    saved = json.loads((tmp_path / "out/summarization_metrics.json").read_text())
    assert saved["systems"]["bm25"]["bleu4"] == result["systems"]["bm25"]["bleu4"]
    sheet = pd.read_csv(tmp_path / "out/human_eval.csv", keep_default_na=False)
    assert len(sheet) == 4 and set(sheet.source_id) <= {f"test{i}" for i in range(6)}
    for column in ("correctness_0_2", "fluency_0_2", "specificity_0_2"):
        assert (sheet[column] == "").all()  # left blank for human raters
    predictions = pd.read_csv(tmp_path / "out/summarization_test_predictions.csv")
    assert list(predictions.columns) == ["source_id", "code", "reference", "ast", "tfidf", "bm25"]

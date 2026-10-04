# TF-IDF Baseline Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run the requested character TF-IDF + Logistic Regression baseline end-to-end.

**Architecture:** Split unique source IDs 70/15/15 with seed 42, keeping original/mutated versions together. Train three configurations on training data only, select by validation Buggy F1, and evaluate the retained winner once on test. Save models and complete evaluation artifacts.

**Tech Stack:** Python, pandas, scikit-learn, joblib, pytest.

**Spec:** User's attached sections 4–14 plus approval of the 70/15/15 source-ID split and seed 42.

## Global Constraints

- Character n-gram ranges: (2, 4), (3, 5), (3, 6); initial baseline (3, 5).
- max_features=30000, lowercase=False, sublinear_tf=True.
- LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).
- Fit only on training data; select only on validation Buggy F1. Keep first evaluated candidate on ties.
- No AST, transformer, retrieval, linting, or application work.

## Review Focus

- Preserve leading zeros in source IDs when reading and saving CSVs.
- Reject invalid labels, missing/empty code, missing IDs, and leaking splits.
- Vocabulary must exclude validation/test-only n-grams.
- Absent mutation categories must have zero counts and null recall.
- Prediction must load the saved artifacts independent of working directory.

## Tasks

- [x] 1. Write failing tests for split coverage, reproducibility, source-ID isolation, invalid inputs, and CSV round trips. Implement `split_dataset.py`: `read_dataset(path)`, `split_dataset(frame)`, `validate_splits(train, validation, test)`, `save_splits(splits, output_dir)` and CLI.
- [x] 2. Write failing tests for train-only fitting, validation selection, metrics, mutation analysis, error examples, saved model prediction, and a fixture CLI run. Implement `train_tfidf.py` and `predict_tfidf.py`. Report metrics for both classes, confusion matrix in [[TN, FP], [FN, TP]] order, and up to five examples per outcome.
- [x] 3. Run real training once, retaining the selected trained model. Save split CSVs, model artifacts, metrics JSON, and complete error examples. Document commands, results, interpretation, and paths in both READMEs and a run report.
- [x] 4. Run `pytest -v`, inspect the diff and saved artifacts, and report all requested results.

## Execution Record

- Ruling: Implement in the shared checkout and leave changes uncommitted for user review; this task includes local ignored data and models already present here.
- Ruling: Prior user approval authorizes execution of the supplied design; no further design permission gate is needed.

- Verification: initial 19 new tests failed for missing behavior; final full suite passed 45 tests and 31 subtests.
- Independent read-only review found no code bugs; added later-winner/tie and fitting/evaluation-order tests for both noted coverage gaps.
- Real run: (2, 4) selected on validation Buggy F1 0.611182; one final test evaluation, Buggy F1 0.618160. Models, reports, split CSVs, and 20 examples saved and audited.

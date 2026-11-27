# Python Bug Detection and Synthetic Bug Explanation

This project reuses the CodeSearchNet preprocessing and operator-mutation pipeline
and adds two classification paths, evidence-based explanations, optional LoRA
summarization, and a small FastAPI backend. Source code is never executed.

**Current status (2026-11-26):**

- **Bug detection.** TF-IDF + LR (test BUGGY F1 0.585) and CodeBERT + LoRA are trained on the
  shared `stratified_v1` split. CodeBERT + LoRA reaches test BUGGY F1 **0.764 ± 0.007** over
  three seeds (42, 13, 7), training in about 11 minutes per run on a Colab T4
  (`results/codebert_seeds.md`).
- **Bug localization.** Without reference code, ranking single-operator repairs by the drop in
  bug probability finds the exact fix first for **78.2%** of test mutants with CodeBERT (67.8%
  TF-IDF, 48.8% random); see "Bug localization" below.
- **Code summarization.** CodeT5-small is fully fine-tuned and LoRA-tuned (r8, r16); the final
  test scores are in "Code summarization" below and `docs/summarization-results.md`. The full
  fine-tune serves `/summarize`. The separate CodeT5 LoRA *bug explainer*
  (`models/codet5_lora`) is not trained, so `"explanation": "lora"` returns 503.
- **Product.** FastAPI backend and React frontend; feedback-driven CodeBERT retraining runs as
  a background job.

All training bugs are synthetic single-operator mutations. On 500 real operator fixes from
GitHub (CTSSB-1M) neither classifier separates the buggy version from the fix (CodeBERT ROC AUC
0.53, TF-IDF 0.50); localization keeps some signal (CodeBERT ranks the exact fix first in 44%
of fixable cases vs 28.5% at random). See [Real bugs](#real-bugs).

## Setup and CPU baseline

Python 3.10+ (tested with 3.12.3). From the repository root:

```bash
cd nlp-code-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Existing dataset: data/processed/bug_detection_dataset.csv
python -m src.bug_detection.experiment_data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.bug_detection.train_comparison_tfidf
python -m src.summarization.prepare
python -m src.evaluation.compare_models
```

The split/preparation/training commands require an empty destination and refuse to
overwrite completed runs. If their artifacts already exist, use them directly or
choose new `--output` directories. Pass the same `--split-dir` to both classifiers
and explanation preparation. Keep `--data-dir` aligned for summarization.
The original dataset and historical `train.csv`, `validation.csv`, `test.csv`,
models and reports remain unchanged.

New shared splits live in `data/processed/stratified_v1/`. They stratify source groups
by their label composition (clean-only versus clean+buggy) while keeping each
`source_id` in exactly one partition. This preserves class balance without leaking
originals across splits. `manifest.json` records dataset and split SHA-256 hashes,
class/mutation counts, seed, and zero-overlap checks. All new trainers verify hashes.
This does not guarantee repository-level separation or remove semantic clones.

Character TF-IDF preserves punctuation, operators and local syntax. The new run uses
`analyzer="char"`, `lowercase=False`, `max_features=30000`, `sublinear_tf=False`.
It tries `(3,5)`, `(2,4)`, `(3,6)` and retains the model with the best validation
BUGGY F1. Logistic Regression uses 2000 maximum iterations, balanced class weights,
and seed 42. Test is evaluated once after selection. Reloaded predictions and
probabilities are checked on validation samples.

## Architecture

```text
CodeSearchNet Python functions
              |
              v
    Preprocessing and AST validation
              |
              v
    Synthetic operator mutation
              |
              v
       Labeled dataset
              |
              v
   Stratified source-group train/val/test
              |
       +------+------+
       |             |
       v             v
   TF-IDF        CodeBERT
      +             +
 Logistic Reg.     LoRA
       |             |
       +------+------+
              |
              v
     Evaluation / detection
              |
              v
  Explanation: verified code pair
  or optional CodeT5 LoRA generation
              |
              v
           FastAPI
              |
              v
     React web interface
```

| Component | Entry point |
|---|---|
| Existing preprocessing/mutations | `src/data/{preprocess,mutations,build_dataset}.py` |
| Frozen shared split protocol | `src/bug_detection/experiment_data.py` |
| New TF-IDF experiment | `src/bug_detection/train_comparison_tfidf.py` |
| CodeBERT LoRA | `src/bug_detection/train_lora.py` |
| Mutation verification and synthetic supervision | `src/summarization/prepare.py` |
| CodeT5 LoRA | `src/summarization/train_lora.py` |
| CodeT5 full fine-tune (function summaries) | `src/summarization/train_codet5.py` |
| Local inference and CLI | `src/inference/{models,cli}.py` |
| Recorded metrics and warm latency comparison | `src/evaluation/compare_models.py` |
| HTTP backend | `src/api/app.py` |
| Web interface (React + Vite) | `frontend/` |

## Optional transformer installation and training

The baseline/API do not require torch, transformers, GPU access, or model downloads.
Install the optional dependencies only to train/use adapters:

```bash
# CPU PyTorch; choose the appropriate official PyTorch wheel for your CUDA system.
python -m pip install 'torch>=2.6,<2.7' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-transformers.txt
export HF_HOME="$PWD/.cache/huggingface"
```

Train CodeBERT + LoRA on a GPU. The recommended run on a Colab T4 takes about 11 minutes
(the old setting, batch 1 with 8-step accumulation in fp32 for 3 epochs, took 66 minutes):

```bash
python -m src.bug_detection.train_lora --device cuda --batch-size 16 --eval-batch-size 64 \
    --precision fp16 --epochs 5 --seed 42 --output models/codebert_lora_seed42
```

`--precision auto` (the default) picks bf16 on GPUs with native bf16, fp16 on other CUDA
GPUs (the T4 has no bf16) and fp32 on CPU; TF32 matmuls are enabled on Ampere and newer.
Batches are grouped by length to reduce padding, padded to a multiple of 8 under mixed
precision, and the learning rate (2e-4) uses a 6% linear warm-up. Training on CPU is
possible but slow (`--device cpu --batch-size 1 --accumulation 8 --gradient-checkpointing`).
Optional flags, off by default so runs stay comparable: `--class-weights`,
`--lora-targets {qv,qkv,all-linear}`, `--lora-r`, `--threshold-tune`. Each run records its
wall-clock time, throughput and peak GPU memory in `training_config.json` (`runtime`).
`scripts/bench_training.py` compares the old and new settings for 50 steps on a tiny
fixture. Start at 512 tokens and lower `--max-length` only as a separately documented
experiment.
CUDA out-of-memory errors include reduction options. Long functions are truncated;
the classifier records truncation fractions and the changed token can be lost.

LoRA freezes the base weights and learns low-rank updates. The CodeBERT path checks
for RoBERTa architecture, discovers attention `query`/`value` linear layers, uses
`TaskType.SEQ_CLS`, `r=8`, alpha 16, dropout 0.1, and explicitly trains/saves the
`classifier` head. Validation BUGGY F1 selects the checkpoint; patience is 2 epochs.
CodeT5 uses T5 attention `q`/`v` layers and `TaskType.SEQ_2_SEQ_LM`; validation loss
selects its checkpoint. Both save adapter/tokenizer, training metadata, base model
revision, and metrics. The seed defaults to 42 (`--seed`); different devices may still differ numerically.
Use `--revision <commit>` to pin a model download; the resolved revision is recorded.

Classification predicts a label, while summarization generates tokens. They are
separate models and objectives, not interchangeable uses of a classification head.
CodeT5-small is chosen as a smaller code-oriented encoder-decoder with an
[Apache-2.0 model card](https://huggingface.co/Salesforce/codet5-small).
CodeBERT's [configuration](https://huggingface.co/microsoft/codebert-base/raw/main/config.json)
identifies RoBERTa attention; the official [CodeBERT repository](https://github.com/microsoft/CodeBERT/blob/master/LICENSE)
uses MIT. Architecture-specific adapter saving follows the
[PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora).
Both models were pretrained on CodeSearchNet, so prior exposure is a further
limitation when interpreting this downstream evaluation.

## Explanation supervision and interpretation

`mutation_type` alone is not a human-written explanation. Preparation pairs each
buggy function with its original in the **same split**, verifies exactly one
supported AST operator change and one changed source token, and rejects conflicts.
It reuses the existing mutation maps and AST checker. Each JSONL record contains:

```json
{
  "source_id": "000001",
  "sample_id": 2,
  "buggy_code": "def f(x): return x - 1",
  "original_code": "def f(x): return x + 1",
  "explanation": "Synthetic mutation: arithmetic operator changed from `+` to `-` on line 1. This may alter behavior; an actual failure is not established.",
  "supervision": "verified_synthetic_pair",
  "change": {"mutation_type": "arithmetic_operator", "old_operator": "+", "new_operator": "-", "line": 1}
}
```

The saved records also include a null suggested fix and the verified change's
explanation/type. Files are hash-checked and pair verification runs again before
training. There are **3320 / 712 / 711** training/validation/test summaries.
These template targets are synthetic, not human annotations. The generator sees
only buggy code; it cannot always identify the original operator without evidence.
It may hallucinate, and token loss on templates does not measure real bug reasoning.

Training exports 20 fixed test examples with generated/reference text and blank
human-review fields. Rate correctness and specificity 0 (wrong/absent), 1 (partial),
2 (supported and precise), and mark unsupported claims true/false. Inspect against
both original and mutated functions, including operator and line. Report counts,
rater agreement and examples; do not fill ratings automatically. No human quality
scores are claimed, and this LoRA explainer is not trained in this checkout (the separate
summarizer in "Code summarization" is).

## Inference and API

```bash
printf 'def f(a,b): return a != b\n' | python -m src.inference.cli --model tfidf
python -m src.inference.cli --model codebert --file function.py
python -m src.inference.cli --model summarizer --file function.py       # CodeT5 LoRA explainer
python -m src.inference.cli --model code_summarizer --file function.py  # fine-tuned CodeT5 summary
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Use `--original-file original.py` with the CLI for verified pair evidence.
The API caches models after their first successful load. Defaults are
`models/tfidf_v2`, `models/codebert_lora`, `models/codet5_lora` (LoRA explainer) and
`models/codet5_summarizer` (fine-tuned summarizer; needs `best/config.json` and
`best/model.safetensors`, plus `training_summary.json` for its max lengths, else 384/48).
`BUG_MODEL_ROOT` can change the parent directory. Inference never downloads models;
transformer base weights must already be in the local cache or at the saved path.

- `GET /health`: service status and each model's unavailable/artifacts-present/loaded/load-failed state.
- `POST /predict`: `{"code":"def f(a,b): return a != b", "model":"tfidf"}`.
  Returns `label`, `bug_probability`, predicted-label `confidence`, and `model`.
- `POST /analyze`: same fields, with optional `original_code` for a verified change
  or `"explanation":"lora"` for explicitly unverified generated text.
  Returns classification plus `is_buggy`, `bug_type`, `explanation`,
  `suggested_fix`, `explanation_source`, and `verified`. With `"summarize": true` it also
  returns `summary` and `summary_source` (`"codet5_finetuned"`, or `null` and
  `"unavailable"` if the summarizer cannot run; the analysis itself still succeeds).
  Without `original_code`, in the default `"deterministic"` mode and with
  `bug_probability >= 0.40`, the response carries the top localization repair (see
  [Bug localization](#bug-localization)): `explanation_source: "operator_flip"`,
  `verified: false`, `explanation` like ``"Line 2: `<` may be `>`"``, `bug_type` (the
  mutation type), `suspected_line`, `suggested_fix` (the whole function with that operator
  flipped) and `repair_candidates` (up to 3: `line`, `old`, `new`, `mutation_type`, `drop`).
  Only repairs that lower the bug probability are returned; if none does, or the code does
  not parse, the explanation stays `"unavailable"`. The UI highlights `suspected_line`.
- `POST /summarize`: `{"code": "..."}` (same code limits, no `model` field). Returns
  `{"summary", "summary_source": "codet5_finetuned", "model": "codet5_summarizer"}`;
  503 if the model is missing or fails.

The summary model is a full CodeT5-small fine-tune that takes the raw code (no prompt)
and generates with beam search (4 beams). It is separate from the CodeT5 LoRA explainer
and describes what a function does, not whether it is buggy. It leans on function names:
for `def add(a, b): return a - b` it still says "Add two elements". Early spot checks
were mixed: `def add(a, b): return a + b` gave "Add a new value to a list", while
`is_adult`, `read_json` and `factorial` got reasonable summaries. Treat summaries as hints.

Without an original, a localization repair or a trained summarizer, explanation/type/fix
are null. An operator-flip repair is the model's guess, not a verified change. A verified
pair can provide evidence even if the classifier predicts CLEAN; the classifier
label is not silently replaced. Verification proves a supported code difference,
not semantic incorrectness or the trustworthiness of a user-supplied original.
Generated explanations always have `verified=false`, unknown `bug_type`, and no
suggested fix. Classifier confidence is not explanation confidence or calibration.

Requests have a 20,000-character code limit (also for originals) and 128,000-byte
body limit, including streamed bodies. Blank/non-string inputs and unknown models
return 422; missing model artifacts/dependencies return 503; oversized bodies return
413. No `eval`, `exec`, or subprocess execution of source; repairs are found by parsing
and re-scoring, never by running the code.

The API also collects feedback for retraining:

- `GET /feedback`: counts of user feedback, mistakes and uncertain cases, plus the five most recent feedback rows.
- `POST /feedback`: `{"code", "model", "predicted_label", "actual_label"}` with optional
  `user_explanation` and `fixed_code`. Saved to `data/feedback/user_feedback.jsonl`.
- `POST /feedback/retrain`: `{"epochs": 1-10, "mistakes_only": bool}` starts a background job
  and returns 202 with a `job_id` (409 if a job is already running). The job fine-tunes the
  active CodeBERT adapter on feedback mixed with replayed training rows, writes it to a new
  `models/codebert_lora_v{N}_{timestamp}` folder, and swaps it in only if validation BUGGY F1
  drops by at most 0.01 (`models/codebert_lora_active.json` records the active adapter).
- `GET /feedback/retrain/{job_id}`: job status (`queued`, `running`, `done`, `rejected`,
  `failed`), progress (phase, epoch, loss) and old/new validation metrics.

Predictions with bug probability from 0.40 to 0.60 are logged to
`data/feedback/uncertain_cases.jsonl`. `BUG_FEEDBACK_DIR` changes the feedback directory.

## Web interface

`frontend/` is a single-page React 18 + Vite + TypeScript + Tailwind app on top of the API.
Run the backend from `nlp-code-assistant/` (the folder that contains `src/`), then the
frontend in a second terminal:

```bash
uvicorn src.api.app:app --reload --port 8000
cd frontend && npm install && npm run dev   # http://localhost:5173
```

The Vite dev server proxies `/api/*` to `http://localhost:8000/*`, so no CORS setup is needed.
Requires Node 18.18+.

- **Analyze:** Python editor (CodeMirror) with an optional original-code editor for verified
  explanations; model switch TF-IDF / CodeBERT and explanation switch rule-based / LoRA. The
  result shows the BUGGY/CLEAN verdict, client-measured latency, bug probability with the
  40-60% uncertain zone, bug type, verified vs. generated explanation, and highlights the line
  the explanation mentions. "Was this right?" sends feedback. A "Summarize function"
  toggle (on by default when the summary model is available) adds a "What this function
  does" section above the verdict, generated by the fine-tuned CodeT5 summarizer.
- **Feedback:** feedback, mistake and uncertain-case counts, recent feedback, and a CodeBERT
  retraining panel.
- **Models:** precision, recall, F1, accuracy, median latency and artifact size for both
  classifiers, read from `results/model_comparison.json` at build time. Run
  `npm run sync-metrics` after re-running `src.evaluation.compare_models`.

Model status pills in the top bar (TF-IDF, CodeBERT, Summary, LoRA explainer) poll
`/health` every 15 s; models that are not trained or failed to load are disabled with a
tooltip. Checks: `npm run typecheck`, `npm run lint`,
`npm test` (Vitest), `npm run build`. See `frontend/README.md` for details.

## Current measured experiment

This is a **new protocol** (stratified groups and ordinary TF). It is not directly
comparable to the historical sublinear-TF run below or the student notebook.

| Model | BUGGY precision | BUGGY recall | BUGGY F1 | Accuracy | Inference latency |
|---|---:|---:|---:|---:|---|
| TF-IDF + LR | 0.5167 | 0.6751 | 0.5854 | 0.6924 | 1.47 ms median |
| CodeBERT + LoRA, 3 seeds (mean ± std) | 0.766 ± 0.035 | 0.763 ± 0.038 | **0.764 ± 0.007** | 0.848 ± 0.007 | not re-measured |
| CodeBERT + LoRA, old single run | 0.7698 | 0.7384 | 0.7538 | 0.8449 | 244 ms median |

The three-seed rows use the new training configuration (batch 16, fp16 on a Colab T4,
5 epochs, about 11 minutes per run); per-seed results are in `results/codebert_seeds.md`.
The old single run (seed 42, batch 1 with 8-step accumulation, fp32, 3 epochs, 66 minutes)
is inside the seed spread, so the new configuration keeps quality and trains about 6 times
faster. The latency and the details below refer to the old run (`models/codebert_lora`).

Latency: 30 validation functions, warm CPU, batch 1, includes vectorization and
prediction, excludes loading. Adapter-only size is not total model size: the TF-IDF
artifacts total 1,246,638 bytes, the CodeBERT LoRA adapter 8,377,131 bytes, and the loaded
base-plus-adapter parameters 502,136,848 bytes. CodeBERT + LoRA improves BUGGY F1 by
0.17 and accuracy by 0.15 over TF-IDF, but is about 170x slower per function on CPU (median latency) and
much larger.

CodeBERT + LoRA validation: BUGGY F1 **0.7464**, accuracy **0.8391**, eval loss 0.3261.
Test confusion matrix (rows = actual clean/buggy):

```text
[[1343, 157],
 [ 186, 525]]
```

New split row counts: **10,320 / 2,212 / 2,211**, with **3,320 / 712 / 711** buggy rows.
All pairwise source-ID overlaps are zero. For TF-IDF, validation selected **(2,4)** with F1
**0.5761**; (3,5) scored 0.5664 and (3,6) scored 0.5650. Final TF-IDF test confusion matrix:

```text
[[1051, 449],
 [ 231, 480]]
```

Test mutation recall: comparison 214/303 = 0.7063; arithmetic 194/277 = 0.7004;
boolean 72/131 = 0.5496. Comparison was easiest and boolean hardest in this run.
See `results/model_comparison.json`, `results/tfidf_v2_metrics.json` and
`models/codebert_lora/metrics.json` for measured values, provenance and per-class reports. Models and generated data remain ignored.

## Bug localization

The classifier says whether a function looks buggy; `src/bug_detection/localize.py` guesses
where. For every operator it proposes the operators that could have been mutated into it (the
mutation tables read backwards), scores the original and all candidates in one batch, and
ranks candidates by how much they lower the bug probability. No reference code is needed.
Evaluated on the 711 test mutants (`src/evaluation/localization_eval.py`; CodeBERT is the
seed-42 model of the new configuration):

| | CodeBERT | TF-IDF | Random choice |
|---|---:|---:|---:|
| Exact fix ranked first (top-1) | **78.2%** | 67.8% | 48.8% |
| Exact fix in top 3 | **92.3%** | 86.9% | 78.6% |
| Correct line ranked first | **86.2%** | 78.1% | — |

The CodeBERT–TF-IDF top-1 gap is significant (paired McNemar test: CodeBERT alone right on
112 mutants, TF-IDF alone on 38, exact p = 1.2 × 10⁻⁹; `src/evaluation/localization_compare.py`).
When CodeBERT flags a function as BUGGY (529 of 711), its first candidate is the exact fix
86.2% of the time. The true fix is always among the candidates here because the bugs come from
the same mutation tables; `<=` and `>=` never get candidates. `/analyze` returns the top
repair as an unverified explanation when no original code is given (see API above). Method,
per-type results and limitations: `docs/bug-localization-results.md`.

## Real bugs

Evaluation only (nothing is trained on it). `scripts/collect_real_bugs.py` turns CTSSB-1M
binary-operator fixes into 500 buggy/fixed function pairs from 488 GitHub repositories;
344 pairs are a change our mutation tables can produce (`in_tables`).

| | CodeBERT | TF-IDF |
|---|---:|---:|
| Classifier ROC AUC, all 1,000 functions | 0.530 | 0.503 |
| Classifier ROC AUC, `in_tables` | 0.556 | 0.509 |
| Localization top-1, `in_tables` (random: 28.5%) | **44.2%** | 37.8% |
| Localization top-3, `in_tables` (random: 61.3%) | **78.2%** | 69.5% |

The classifiers do not transfer: on real bugs they are at or barely above chance. Given a
buggy function, localization still beats random and CodeBERT beats TF-IDF (McNemar top-1
p = 0.033), but at about half the synthetic accuracy. 31% of the real fixes cannot be proposed
at all (mostly a correct `>`/`<` that became `>=`/`<=`). The pairs CSV holds third-party code
and is not committed; results are in `results/real_bugs/`, details in `docs/real-bugs.md`.

## Metrics

Bug detection is scored by `src/evaluation/bug_metrics.py` and `src/evaluation/compare_models.py`;
summaries by `src/summarization/metrics.py` and `src/summarization/evaluate_summaries.py`;
figures by `src/evaluation/make_figures.py` (`results/figures/`). The tables below are copied
from `results/evaluation_report.md`, which the comparison script writes from
`results/model_comparison.json`. Thresholds and settings are chosen on validation only; the
test split is only scored.

### What each metric means and why it is here

The test split is 68% CLEAN / 32% BUGGY, so a model that always says CLEAN already gets 68%
accuracy. Metrics that stay honest under that imbalance (balanced accuracy, MCC, PR-AUC) are
reported next to accuracy. Both models output a probability, so calibration is checked too.

| Metric | One line | Why it matters here |
|---|---|---|
| Accuracy | Share of correct predictions | Familiar, but inflated by the 68% CLEAN majority |
| Balanced accuracy | Mean of BUGGY recall and CLEAN recall | Always-CLEAN scores 0.5, not 0.68 |
| BUGGY precision / recall / F1 | Of flagged functions, how many are buggy / of buggy functions, how many are flagged / their harmonic mean | The class we care about; F1 was the checkpoint-selection metric |
| CLEAN F1, macro-F1, weighted F1 | F1 for CLEAN; unweighted and support-weighted mean over both classes | Shows whether BUGGY gains cost CLEAN performance |
| Specificity, FPR, FNR | Clean rows kept clean; clean rows flagged; buggy rows missed | False alarms vs. missed bugs |
| MCC | Correlation between prediction and truth, −1 to 1, 0 = chance | Single number that uses all four confusion cells; robust to imbalance |
| Cohen's kappa | Agreement beyond chance | Same idea as MCC, common in annotation work |
| ROC-AUC | Probability a random buggy function scores above a random clean one | Threshold-free ranking quality |
| PR-AUC (average precision) | Area under the BUGGY precision-recall curve | Threshold-free and imbalance-aware; a random ranker scores the positive rate (0.32), not 0.5 |
| Brier score, log loss | Squared / logarithmic error of P(BUGGY) | Overall probability quality; lower is better |
| ECE (10 bins) + reliability diagram | Average gap between predicted P(BUGGY) and the observed BUGGY rate | Whether "80% buggy" really means 80%; matters for the UI's probability bar and uncertain zone |
| Bootstrap 95% CI | Percentile interval from 1000 row resamples, seed 42 | How much a number could move on another sample of the same size |
| Validation-tuned threshold | Threshold maximizing BUGGY F1 on validation, then applied to test | Shows what threshold choice alone buys, without peeking at test |
| Per-mutation recall, clean FPR | Recall per comparison / arithmetic / boolean mutation; false-positive rate on clean rows | Which bug types are hard |
| By length / truncation | Metrics per token-count quartile and for functions over 512 tokens | CodeBERT truncates at 512 tokens, so the changed token can be cut off |
| McNemar's test | Exact binomial test on rows where exactly one model is right | Whether two models differ on the same test rows |
| Paired bootstrap of the F1 difference | CI of F1(CodeBERT) − F1(TF-IDF) from shared resamples | Size and uncertainty of the improvement |
| Latency p50 / p95, throughput | Warm CPU per-function time at batch 1; functions/second at batch 32 | Cost of the accuracy gain |

Summarization metrics (all in 0–1): corpus BLEU-4 and smoothed sentence BLEU-4 (n-gram precision
with a brevity penalty), ROUGE-1/2/L precision, recall and F1 (unigram, bigram and longest-common-
subsequence overlap), chrF (character n-gram F-score, robust to word-form differences), METEOR
(only when nltk and its WordNet data are installed locally; nothing is downloaded), exact match,
average generated vs. reference length, empty-output rate, and the share of outputs that copy more
than 80% of the input's distinct tokens. BLEU-4 and ROUGE-L get bootstrap 95% CIs.
`results/summarization_human_eval.csv` holds 50 random test rows with blank correctness, fluency
and specificity columns (0–2) for human raters; they are never filled automatically.

### Measured: bug detection (test split, 2211 rows)

**Status.** TF-IDF + LR is fully evaluated. CodeBERT + LoRA has not been run on the new
per-row protocol yet, because that needs about 4,400 predictions and this machine has no GPU.
Its training-time test scores are in the table in "Current measured experiment" below, but its
MCC, AUCs, calibration, CIs and the McNemar test are still missing. To complete them, run in
Colab from the repository root (with `data/processed/stratified_v1` and `models/codebert_lora`):

```bash
python -m src.evaluation.compare_models --models codebert --device cuda --batch-size 64 --output /content/colab_comparison.json
```

then copy `results/codebert_validation_predictions.csv` and `results/codebert_test_predictions.csv`
into `results/` here and run:

```bash
python -m src.evaluation.compare_models --skip-predict
python -m src.evaluation.make_figures
cd frontend && npm run sync-metrics
```

#### Main metrics (test, threshold 0.5)

| Metric | TF-IDF + LR |
|---|---:|
| Accuracy | 0.692 [0.672, 0.711] |
| Balanced accuracy | 0.688 |
| BUGGY precision | 0.517 |
| BUGGY recall | 0.675 |
| BUGGY F1 | 0.585 [0.554, 0.613] |
| CLEAN F1 | 0.756 |
| Macro-F1 | 0.670 |
| Specificity (TNR) | 0.701 |
| False positive rate | 0.299 |
| False negative rate | 0.325 |
| MCC | 0.356 [0.313, 0.394] |
| Cohen's kappa | 0.348 |
| ROC-AUC | 0.763 [0.743, 0.782] |
| PR-AUC (baseline = positive rate) | 0.525 [0.485, 0.566] |
| PR-AUC random baseline | 0.322 |
| Brier score (lower is better) | 0.199 |
| Log loss | 0.582 |
| ECE, 10 bins | 0.134 |
| CPU latency p50, batch 1 (ms) | 1.47 |
| CPU latency p95, batch 1 (ms) | 9.62 |
| CPU throughput, batch 32 (functions/s) | 805.7 |

Values in brackets are 95% bootstrap confidence intervals.

#### Recall per mutation type and clean false positives (test, threshold 0.5)

| Rows | TF-IDF + LR |
|---|---:|
| comparison operator recall | 0.706 (214/303) |
| arithmetic operator recall | 0.700 (194/277) |
| boolean operator recall | 0.550 (72/131) |
| clean rows flagged BUGGY (FPR) | 0.299 (449/1500) |

#### Threshold analysis (tuned on validation, applied to test)

| Model | Threshold | Test BUGGY P | Test BUGGY R | Test BUGGY F1 | Test MCC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LR (default 0.5) | 0.500 | 0.517 | 0.675 | 0.585 | 0.356 |
| TF-IDF + LR (validation tuned) | 0.455 | 0.496 | 0.775 | 0.605 | 0.376 |

#### Confusion matrices (test, threshold 0.5)

**TF-IDF + LR** (rows = actual, columns = predicted)

| | Pred. CLEAN | Pred. BUGGY |
|---|---:|---:|
| Actual CLEAN | 1051 (70.1%) | 449 (29.9%) |
| Actual BUGGY | 231 (32.5%) | 480 (67.5%) |

#### By code length (CodeBERT token-count quartiles of the test rows)

| Group | n | TF-IDF + LR F1 / acc. |
|---|---:|---:|
| Q1 (43–114 tokens) | 553 | 0.500 / 0.830 |
| Q2 (115–199 tokens) | 554 | 0.587 / 0.751 |
| Q3 (200–409 tokens) | 552 | 0.579 / 0.641 |
| Q4 (410–9551 tokens) | 552 | 0.614 / 0.547 |
| fits | 1807 | 0.574 / 0.727 |
| truncated (> 512 tokens) | 404 | 0.613 / 0.537 |

#### Efficiency (warm CPU)

| Model | Latency p50 (ms) | Latency p95 (ms) | Mean (ms) | Batch-32 throughput (fn/s) | Trainable params | Total params | Artifact size | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TF-IDF + LR | 1.47 | 9.62 | 2.58 | 805.7 | 30,001 | 30,001 | 1.2 MB | not recorded |
| CodeBERT + LoRA | 244.49 | 902.39 | 388.12 | 0.7 | 887,042 | 125,534,212 | 8.4 MB | not recorded |

Latency: Warm CPU, batch=1, includes tokenization/vectorization, excludes loading; validation rows only. Throughput: batch 32 over the first 64 validation rows.

CPU latency varies noticeably between runs on this machine; the numbers above are from the
saved run in `results/model_comparison.json`. TF-IDF probabilities are over-confident for BUGGY
(see `results/figures/reliability.png`): above 0.8 predicted, only about 50–60% of rows are buggy,
consistent with its `class_weight="balanced"` training.

#### Efficiency on GPU (CodeBERT, Tesla T4)

`scripts/bench_inference.py` (results in `results/bench_inference.json`) measures the seed-42
model in every combination of unmerged adapter vs merged LoRA weights, fp32 vs fp16 autocast
and eager vs `torch.compile`. Latency is the median of 50 forward passes after 10 warm-up
passes, synchronized with `torch.cuda.synchronize()`, tokenization excluded; peak memory
includes the weights. Agreement is against the adapter / fp32 / eager mode on 200 random
validation functions.

| Weights | Precision | Mode | Batch 1 (ms) | Batch 16 (ms) | Peak memory (MB) | Same label | Max prob. difference |
|---|---|---|---:|---:|---:|---:|---:|
| adapter | fp32 | eager | 11.9 | 423.1 | 768 | reference | reference |
| adapter | fp16 | eager | 14.7 | 112.9 | 685 | 200/200 | 0.0017 |
| adapter | fp32 | compile | 12.6 | 448.0 | 665 | 200/200 | 2e-6 |
| adapter | fp16 | compile | 7.7 | 105.8 | 595 | 200/200 | 0.0015 |
| merged | fp32 | eager | 12.9 | 487.9 | 766 | 200/200 | 2e-6 |
| merged | fp16 | eager | 9.8 | 127.5 | 682 | 200/200 | 0.0025 |
| merged | fp32 | compile | 13.0 | 486.8 | 766 | 200/200 | 3e-6 |
| merged | fp16 | compile | **7.1** | **104.8** | **591** | 200/200 | 0.0019 |

- **fp16 autocast is the main gain:** batch 16 is about 3.8x faster (423 → 113 ms adapter,
  488 → 127 ms merged), i.e. about 150 instead of 38 functions per second, with the same label
  on all 200 functions and probabilities within 0.0025.
- **`torch.compile` helps only together with fp16, and mostly at batch 1:** 7.1–7.7 ms against
  9.8–14.7 ms in fp16 eager; at batch 16 it adds 6–18%. In fp32 it gives nothing. Compiling takes
  time once per input shape, so with varying function lengths in the API the first requests
  are slow.
- **Merging the adapter gave no measurable gain** here; merged fp32 was even slower at
  batch 16 (488 vs 423 ms). This is one run on a shared Colab GPU, where clocks vary; a
  difference of this size is not reliable.
- Best mode (merged, fp16, compile): batch 1 1.7x and batch 16 4.0x faster than the
  reference, 23% less peak memory (591 vs 768 MB). The API still runs fp32 eager; this was
  measured, not switched on.

### Measured: summarization

Summarization is evaluated in `docs/summarization-results.md` (validation for every choice,
test scored once). Test split, 1,400 examples, from `results_summarization/comparison_test.md`:

| Method | BLEU-4 | ROUGE-L F1 |
|---|---:|---:|
| AST template | 0.0079 | 0.2237 |
| TF-IDF retrieval | 0.1840 | 0.3283 |
| BM25 retrieval | 0.1770 | 0.3251 |
| CodeT5-small full fine-tune | 0.1398 | **0.3862** |
| CodeT5-small LoRA (r16) | 0.1066 | 0.3620 |

`python -m src.summarization.evaluate_summaries` adds further metrics (ROUGE-1/2, chrF,
bootstrap CIs) on the same data; it has not been run on the full test split yet.

## Tests and limitations

```bash
OMP_NUM_THREADS=2 pytest -v                    # everything
OMP_NUM_THREADS=2 pytest -v --ignore=tests/test_lora_local.py \
    --ignore=tests/test_training_optimizations.py   # skip tiny transformer training runs
python -m pip check
cd frontend && npm test                        # frontend
```

Transformer tests skip if optional dependencies are absent; with them installed,
`tests/test_lora_local.py` and `tests/test_training_optimizations.py` train tiny random
local RoBERTa/T5 fixtures (gradient checks, adapter save/reload, generation). Those fixture
scores are not model comparison results. Training never starts at import. A real Uvicorn
smoke test is saved in `results/api_smoke.json`.

Limitations: the bugs are synthetic single-operator replacements, which may not change
behaviour, and the "clean" originals may already contain bugs. On real bugs the
classifiers are near chance (see [Real bugs](#real-bugs)). Dataset sampling, repository overlap and semantic clones limit generalization,
and CodeBERT and CodeT5 were pretrained on CodeSearchNet, so they may have seen the test
functions. Latencies were measured on different hardware (TF-IDF and CodeBERT on a laptop
CPU, CodeT5 on an Apple M4) and are not directly comparable.

---

## Code summarization

One-sentence summaries of Python functions (CodeT5-small, full fine-tuning or LoRA) with
retrieval and rule-based baselines. Code is never executed. Details and results:
`docs/summarization-results.md`. Run from `nlp-code-assistant/`; install
`requirements-transformers.txt` first. Checkpoints go to `checkpoints/` (outside git).

```bash
# 1. Dataset (data/processed/summarization_dataset.csv), then the shared split
python -m src.summarization.build_summary_dataset
python -m src.summarization.apply_shared_split

# 2. Baselines (validation): ast | tfidf | bm25
python -m src.summarization.evaluate_baseline --method bm25

# 3. Training (the output folder must be empty; add --limit 64 --epochs 1 for a smoke test)
python -m src.summarization.train_codet5        # full fine-tuning -> checkpoints/codet5_summarizer
python -m src.summarization.train_codet5_lora   # LoRA r=8 -> checkpoints/codet5_summarizer_lora

# 4. Evaluation on validation (full model, LoRA adapter, or one comparison table)
python -m src.summarization.evaluate_model
python -m src.summarization.evaluate_model --lora-adapter checkpoints/codet5_summarizer_lora/best
python -m src.summarization.evaluate_model --compare    # -> results_summarization/

# 5. Error review of the saved validation predictions
python -m src.summarization.review_examples
```

The test split is only scored with `--confirm-final-test` (final evaluation, once).

```python
from src.summarization.summarize import summarize

summarize("def add(a, b):\n    return a + b")                # auto: full if found, else LoRA
summarize(code, backend="lora")                               # or backend="full"
```

CLI: `python -m src.summarization.summarize --file f.py --backend {auto,full,lora}`.
`CODET5_SUMMARIZER_CHECKPOINT` can point to any full or LoRA checkpoint folder.

## Historical data preparation and baseline (2026-09-28)

The pipeline collects **10,000 unique CodeSearchNet Python functions** and creates at most
one synthetic version of each. `label=0` is the original function, `label=1` the function
with one operator replaced. `clean` means "the original code": CodeSearchNet does not
guarantee that it is free of real bugs.

Previous run: **10,000 clean + 4,743 buggy = 14,743 rows**.
Current local run: **10,000 clean + 5,304 buggy = 15,304 rows**; see the baseline section below.

### Running

Python **3.10+**, from `nlp-code-assistant/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/data/download_codesearchnet.py
python src/data/preprocess.py
python src/data/build_dataset.py
```

Only the first step needs the internet; the rest work on local files. Default paths are
resolved relative to the project and all options are listed by `--help`. For a small run,
pass the same `--target 100` to the first two commands. `download_codesearchnet.py --input
local.jsonl` accepts a local JSONL file with a `code` field and keeps its order.

### Files

| File | Purpose |
|---|---|
| `src/data/download_codesearchnet.py` | Streams up to 10,000 accepted functions |
| `src/data/preprocess.py` | Cleaning, AST checks, de-duplication, `source_id` |
| `src/data/mutations.py` | One AST mutation that keeps the original formatting |
| `src/data/build_dataset.py` | Assembly, quality checks, statistics and examples |
| `src/bug_detection/split_dataset.py` | 70/15/15 split by `source_id` |
| `src/bug_detection/train_tfidf.py` | Training, configuration choice, evaluation and model saving |
| `src/bug_detection/predict_tfidf.py` | `predict_bug(code)` for new Python code |
| `data/raw/codesearchnet/python_functions.jsonl` | Processed source records and provenance |
| `data/raw/codesearchnet/python_functions.manifest.json` | Revision, URL, seed, library versions and counters |
| `data/processed/clean_functions.csv` | `source_id,code,label,mutation_type` |
| `data/processed/bug_detection_dataset.csv` | `sample_id,source_id,code,label,mutation_type` |
| `data/processed/*.stats.json` | Cleaning counters and class/mutation distributions |
| `data/processed/bug_detection_dataset.examples.txt` | Five full pairs per category |

Data, models, caches and the virtual environment are excluded by `.gitignore`.

### Source and reproducibility

The source is the [official CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net)
Python data, upstream `train` partition. This chooses a source; it is not a split of the new
dataset. The Parquet files are pinned to revision `7e3332b8032ff895377dba3e57f0152fcb32ff7d`.

The loader uses [Hugging Face Datasets streaming](https://huggingface.co/docs/datasets/en/stream)
and reads only the code and provenance columns. Seed **42** and a shuffle buffer of **1000**
give a reproducible sample from the start of the shuffled stream, **not a uniform sample of
the whole corpus**. Reading stops after 10,000 accepted functions (the library may read
ahead a buffer and Parquet row groups); at most 100,000 records are scanned (`--max-scan`).
The full CodeSearchNet archive is not downloaded.

Parquet is read in sequential batches of 256 records with PyArrow and shuffled by
`datasets.IterableDataset`. This allows stopping the stream without a background
multi-threaded scanner, which crashed at Python shutdown after an early stop in the tested
environment.

To repeat a run, use the saved JSONL and the versions in the manifest: the stream order
depends on `datasets` and the AST on Python. In the raw JSONL, accepted records have the same
`source_id` as the clean CSV; rejected ones have `source_id=null`. Repository, path, function
name and source URL are kept. Separate documentation fields are not downloaded.

### Cleaning

- CRLF/CR are normalized to LF, surrounding whitespace and empty samples are removed.
- `ast.parse()` drops syntactically invalid samples.
- Exactly one top-level `FunctionDef` or `AsyncFunctionDef` (decorators allowed) is
  accepted; fragments and whole classes are dropped.
- Leading docstrings are removed by AST coordinates without reformatting the rest of the
  body. If the body consisted only of docstrings, `pass` remains.
- An exact match of the cleaned code counts as a duplicate.
- IDs are assigned after cleaning, starting from `000001`.

`Valid Python` includes syntactically valid duplicates and rejected non-function fragments.
`Invalid Python` counts parse/processing errors; the first five messages are kept in the
statistics. Empty samples are counted separately. If there is not enough data, the command
fails instead of publishing an incomplete CSV as a success.

### Mutations

| Category | Replacements |
|---|---|
| `comparison_operator` | `== → !=`, `!= → ==`, `> → <`, `< → >`, `>= → >`, `<= → <` |
| `arithmetic_operator` | `+ → -`, `- → +`, `* → /`, `/ → *` |
| `boolean_operator` | `and → or`, `or → and` |

An applicable category and node are chosen at random. **One AST operator** is changed, the
result is rendered with `ast.unparse()` and re-parsed. In the final CSV only that operator's
token is replaced in the source text, and the resulting tree must match the mutated AST:
there is no global string replacement. Strings, comments and formatting are preserved in both
classes.

Operators in decorators, annotations and default argument values are not changed. In
comparison chains one comparison is changed. Boolean operations use nodes with two operands:
a flat `a and b and c` would need two token replacements for a direct `BoolOp.op` change and
is skipped. Nested binary operations are allowed.

Invalid candidates, or ones that change operator precedence, are skipped and the remaining
ones are tried. Each mutation is checked against **all** clean samples and accepted
mutations; if no unique valid variant exists, only the clean function is kept.
`Invalid mutations skipped` counts attempts, not functions; collisions and functions without
a mutation are reported separately.

Replacing an operator creates a synthetic label; it does not prove a behavioural error on real
inputs. Dataset code is never executed.

### Checks and reading the CSV

Before writing, the pipeline checks that all code parses, global uniqueness, allowed labels,
categories and IDs, that every mutation has its clean original, and that each pair differs by
exactly one allowed AST replacement.

```python
import pandas as pd

df = pd.read_csv(
    "data/processed/bug_detection_dataset.csv",
    dtype={"source_id": str},  # keep leading zeros
    keep_default_na=False,
)
```

Source-ID splitting, training, and evaluation are implemented in the baseline below.

### TF-IDF + Logistic Regression Baseline

Character-level TF-IDF represents Python source code using character n-grams.
These features capture local syntax and operators such as `==`, `!=`, `>=`, `+`,
and `-`. Logistic Regression predicts CLEAN (0) versus BUGGY (1).

The baseline validates all required fields, removes exact duplicate rows, and
splits unique `source_id` values into 70% train, 15% validation, and 15% test
with seed 42. Original and mutated versions stay together to prevent source-ID
leakage. Row proportions can differ slightly because groups have different sizes.
All three pairwise source-ID intersections are checked explicitly.

Only training code is used to fit TF-IDF and Logistic Regression. The experiment
compares `(2, 4)`, `(3, 5)`, and `(3, 6)` character n-grams with `max_features=30000`,
`lowercase=False`, and `sublinear_tf=True`. Logistic Regression uses
`max_iter=2000`, `class_weight="balanced"`, and `random_state=42`.
The largest validation Buggy F1 wins; ties keep the first configuration in that
order. The selected train-only model is retained without refitting and evaluated
once on test data. Running the training command again repeats that experiment.

From this directory, in PowerShell with standard CPython installed:

```powershell
py -3.12 -m venv .venv-baseline
.\.venv-baseline\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-baseline\Scripts\python.exe src/bug_detection/train_tfidf.py
.\.venv-baseline\Scripts\python.exe -m pytest -v
```

On other platforms, activate your Python environment and use `python` for the
same commands. The separate split command is
`python src/bug_detection/split_dataset.py`. Both scripts support `--help` and
resolve default paths relative to this project, independent of the working directory.

This checkout originally lacked the processed CSV. The current dataset was
regenerated from the first 10,000 valid unique functions in the local training
shard `data/raw/python/final/jsonl/train/python_train_0.jsonl`. This is a new local
subset, not a reproduction of the older Hugging Face subset described above.
To reproduce its preparation using the existing scripts:

```powershell
.\.venv-baseline\Scripts\python.exe src/data/download_codesearchnet.py --input data/raw/python/final/jsonl/train/python_train_0.jsonl
.\.venv-baseline\Scripts\python.exe src/data/preprocess.py
.\.venv-baseline\Scripts\python.exe src/data/build_dataset.py
```

The preparation manifest is saved beside the raw subset. The baseline metrics
include the processed CSV SHA-256 and package versions for reproducibility.
No collected Python source is executed. Existing AST-based mutation generation
is used solely to prepare labels; the model uses only character TF-IDF features.

Artifacts:

- `data/processed/train.csv`, `validation.csv`, `test.csv`: grouped splits.
- `models/tfidf_vectorizer.joblib`, `models/logistic_regression.joblib`: selected model.
- `results/tfidf_logistic_regression_metrics.json`: dataset/split statistics,
  validation comparison, both class reports, confusion matrices, mutation recall,
  and up to five examples in each prediction category.
- `results/tfidf_error_examples.txt`: full readable TP/TN/FP/FN code examples.
- `results/tfidf_test_predictions.csv`: test predictions and bug probabilities.
- `docs/tfidf-baseline-results.md`: results from the completed local run.

```python
from src.bug_detection.predict_tfidf import predict_bug

result = predict_bug("def calculate_total(price, tax):\n    return price - tax")
print(result)  # {"label": "clean" or "buggy", "bug_probability": a float in [0, 1]}
```

Mutation-type recall is `null` when a category has no test samples. Undefined
classification precision/recall is reported as zero. Probabilities are model
estimates, not proof that a function is correct or buggy. Synthetic operator
mutations and source-ID separation do not establish performance on real bugs or
on entirely unseen repositories.

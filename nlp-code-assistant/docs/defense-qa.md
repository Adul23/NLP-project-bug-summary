# Defense Q&A

Short answers to likely questions, for both parts. Each answer names the file where it can be
shown. Numbers are from the repository (see `docs/report.md` for the full picture).

## Problem and data

**1. What problem do you solve, and for whom?**
Given one Python function, we say whether it contains a swapped operator, point to the
suspected line and fix, and write a one-sentence summary. It is a study and code-review aid,
not a verifier: code is never executed. (`docs/report.md` §1, `src/api/app.py`)

**2. Where does the data come from, and what is its license?**
CodeSearchNet Python on Hugging Face, revision `7e3332b8…`, streamed with seed 42, first 10,000
accepted functions. Its license is listed as "other": each function keeps its GitHub repository's
license, and there is no per-example license field; we use it for course research only.
(`src/data/download_codesearchnet.py`, `docs/dataset-card.md`)

**3. How did you clean the data?**
LF line endings; drop code that `ast.parse()` rejects; keep exactly one top-level function;
remove the docstring by AST coordinates without reformatting; remove exact duplicates; assign
IDs after cleaning. (`src/data/preprocess.py`, README "Cleaning")

**4. How are the labels made? Is a BUGGY function really buggy?**
BUGGY is the original with one AST operator swapped (comparison, arithmetic, and/or); CLEAN is
the original. The label marks a changed operator, not a proven failure, and CLEAN code may
contain real bugs. (`src/data/mutations.py`, `docs/dataset-card.md`)

**5. What is the class balance?**
32.2% BUGGY in every split: test has 1,500 CLEAN and 711 BUGGY. (`data/processed/stratified_v1/manifest.json`)

**6. How did you prevent leakage?**
The 70/15/15 split groups rows by `source_id`, so a function and its mutant never cross splits
(overlap 0). The manifest stores SHA-256 hashes that every trainer checks. All choices use
validation; test is scored once. Summaries that appear verbatim in the code are removed.
(`src/bug_detection/split_dataset.py`, `data/processed/stratified_v1/manifest.json`)

**7. Are there near duplicates between train and test?**
No exact copies; no validation or test function is ≥ 0.95 TF-IDF cosine to train, but 137 of
1,500 test functions are ≥ 0.8. The closest pairs are sibling methods of one class in one
repository: the split is by function, not by repository. This is a measured limitation.
(`scripts/near_duplicates.py`, `results/near_duplicates.json`)

**8. What is the summarization dataset?**
Clean functions without docstrings as input, the cleaned first docstring sentence as target;
same split: 6,524 / 1,384 / 1,400, median reference 8 words. (`src/summarization/build_summary_dataset.py`, `docs/summarization-results.md`)

## Tokenization and architecture

**9. How does the tokenizer turn code into IDs?**
CodeBERT uses RoBERTa byte-level BPE (vocabulary 50,265): the code is split into subword pieces
(a space becomes part of the next token), mapped to IDs, wrapped in `<s> … </s>` and cut at 512.
`scripts/demo_tokenize.py --model codebert` prints the first 40 tokens and their IDs.
(`src/bug_detection/lora_common.py` `tokenize_once`, `scripts/demo_tokenize.py`)

**10. What is the attention mask?**
A 0/1 vector of the same length as `input_ids`: 1 for real tokens, 0 for padding. Functions in a
batch have different lengths, so shorter ones are padded and the mask stops attention to the
padding. For a single function it is all ones. (`DataCollatorWithPadding` in `src/bug_detection/train_lora.py`)

**11. What happens to long functions?**
CodeBERT keeps the first 512 tokens (special tokens kept); 404 of 2,211 test functions are
longer, so a mutation near the end can be lost. CodeT5 reads up to 384 tokens. TF-IDF reads the
whole function. (`src/bug_detection/lora_common.py`, README "By code length")

**12. What does TF-IDF see?**
Character n-grams of 2–4 characters (range chosen on validation), 30,000 features, no
lowercasing, so `!=`, `>=` and ` and ` are features directly. (`src/bug_detection/train_tfidf.py`, `models/tfidf_v2/metrics.json`)

**13. What is the output layer and the loss of each model?**
TF-IDF + LR: one linear unit, sigmoid, log-loss with L2 and balanced class weights. CodeBERT:
`RobertaClassificationHead` on the `<s>` vector (dense + tanh → 2 logits), softmax,
cross-entropy. CodeT5: LM head over 32,100 tokens, token-level cross-entropy with teacher
forcing, beam search (4 beams) at inference. (`docs/pipeline.md`)

**14. Why LoRA?**
The base model stays frozen; LoRA learns low-rank updates on attention `query`/`value` (r=8,
alpha 16) and the classification head is trained fully: 887,042 of 125.5M parameters (0.71%),
an 8.4 MB adapter instead of a full model copy, and training fits a free T4 in about 11 minutes.
(`src/bug_detection/lora_common.py` `add_classifier_lora`)

**15. How does localization work without the original code?**
Every operator is flipped back to each operator it could have been mutated from; all candidate
functions and the original are scored in one batch; the candidate that lowers the bug probability
most is the suspected fix. No execution. (`src/bug_detection/localize.py`)

## Training

**16. Which optimizer, learning rate and schedule?**
CodeBERT: HF Trainer's AdamW, lr 2e-4, linear schedule, 6% warmup, weight decay 0.01, batch 16,
fp16, up to 5 epochs. CodeT5: AdamW, weight decay 0.01, linear schedule with 5% warmup, gradient
clipping 1.0, batch 8, lr 1e-4 (full) / 2e-4 and 5e-4 (LoRA r8 / r16). (`src/bug_detection/train_lora.py`, `src/summarization/train_codet5.py`)

**17. Where is the training configuration?**
In the training scripts above; each CodeBERT run also writes `training_config.json` next to its
adapter (base model, revision, max length, runtime). Hyperparameter table: `docs/report.md` §6.

**18. How was the checkpoint chosen?**
CodeBERT: evaluation each epoch, best validation BUGGY F1 (`metric_for_best_model="bug_f1"`,
`load_best_model_at_end`), early stopping with patience 2. CodeT5: best validation loss,
patience 2. Test is never used for selection. (`src/bug_detection/train_lora.py`)

**19. What do the learning curves show?**
CodeBERT: validation loss falls from 0.41–0.45 to about 0.31 and flattens, train loss ends near
0.32: no overfitting; for 2 of 3 seeds the last epoch is best, so maybe slightly undertrained.
CodeT5 full fine-tuning overfits after epoch 4 and stops at epoch 6. (`results/figures/codebert_learning_curves.png`, `results/learning_curves.json`)

**20. Why is validation loss below train loss at the start?**
The train value averages the last 500 steps with dropout on, while the model is still improving;
validation is measured at the end of the epoch without dropout. (`results/learning_curves.json`)

**21. How stable are the results across seeds?**
Three CodeBERT seeds (42, 13, 7): test BUGGY F1 0.756, 0.767, 0.768, so 0.764 ± 0.007. Precision
and recall move more (seed 13 trades precision for recall). (`results/codebert_seeds.md`)

## Evaluation and error analysis

**22. Why F1 of the BUGGY class and not accuracy?**
Only 32% of test functions are buggy: always answering CLEAN gives 67.8% accuracy and BUGGY
F1 = 0. F1 of the positive class rewards finding bugs and penalizes false alarms. (`src/evaluation/bug_metrics.py`)

**23. Macro or weighted averages?**
We report BUGGY F1 as the main metric and macro F1 (plain mean of both classes, not weighted by
support): TF-IDF 0.670, CodeBERT 0.826 ± 0.006. (`results/tfidf_v2_metrics.json`, `results/codebert_seeds/*/metrics.json`)

**24. What are the main results?**
Test BUGGY F1: TF-IDF 0.585, CodeBERT 0.764 ± 0.007. Localization top-1 on 711 test mutants:
78.2% CodeBERT, 67.8% TF-IDF, 48.8% random (McNemar p = 1.2 × 10⁻⁹). Summaries: full CodeT5
ROUGE-L 0.386, BLEU-4 0.140 on test. (`docs/report.md` §7)

**25. Show a model error and how you would investigate it.**
`pfLon` (test): `asc + sun + moon` instead of `asc + sun - moon`; CodeBERT flags it (0.88), but its
first repair is on line 9, the other branch of the `if`; the exact fix is second. To investigate,
run `scripts/demo_tokenize.py` on the function: tokens, truncation, probability and the ranked
repairs. (`results/localization/codebert_rows.csv`, slide 08)

**26. What does the confusion matrix say?**
CodeBERT seed 42 on test: 1,340 / 160 for CLEAN, 182 / 529 for BUGGY. It misses 26% of buggy
functions and flags 11% of clean ones; TF-IDF flags 449 clean functions (30%). (`results/codebert_seeds/42/metrics.json`)

**27. What do BLEU and ROUGE-L measure, and what not?**
Word overlap with one reference docstring after lowercasing: BLEU-4 is corpus-level 1–4-gram
precision with a brevity penalty; ROUGE-L is the F1 of the longest common subsequence, averaged
over examples. They do not check whether the summary is true to the code and penalize valid
paraphrases; human evaluation and BERTScore were not measured. (`src/summarization/metrics.py`)

**28. Why does retrieval beat CodeT5 on BLEU-4?**
Retrieval copies real docstrings, which match 4-grams exactly when similar functions exist in
train; CodeT5 writes shorter text (about 6.5 words against 9.3), and the brevity penalty lowers
its BLEU. By ROUGE-L, full CodeT5 is best. (`docs/summarization-results.md`)

**29. Where does CodeT5 go wrong?**
Mostly too short and generic summaries (471 of 1,384 on validation for LoRA r8), summaries that
follow the function name, copied comments and identifiers, and reasonable answers scored low
against noisy references. (`docs/summarization-results.md` "Error analysis")

**30. Why is ROC AUC near 0.5 on real bugs?**
The detector learned what our synthetic mutations look like, not what real bugs look like; on
500 real operator fixes it cannot tell the buggy version from the fix (CodeBERT 0.53, TF-IDF
0.50). Localization keeps some signal on fixes of the trained kind: 44.2% top-1 vs 28.5% random.
(`docs/real-bugs.md`, `results/real_bugs/`)

**31. Where is the evaluation code?**
Bug metrics: `src/evaluation/bug_metrics.py`, `src/evaluation/compare_models.py`; localization:
`src/evaluation/localization_eval.py`, `localization_compare.py`; summaries:
`src/summarization/evaluate_model.py`, `metrics.py`.

**32. What would you do next?**
Train on real bug fixes and add mutations for `<=`, `>=` and non-operator bugs, split by
repository, run the per-row protocol (AUC, calibration) for CodeBERT, evaluate summaries with
people, and switch the API to fp16. (`docs/report.md` §12)

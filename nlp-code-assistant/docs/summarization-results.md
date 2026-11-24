# Code summarization results

Task: one-sentence English summary of a Python function (docstring removed from the input).
Status: complete. Validation results are used for all choices; the test split was scored **once** at the end (`--compare --split test --confirm-final-test`, LoRA r16 as the LoRA model). Every number below comes from the saved result files (`results_summarization/`, `checkpoints/codet5_summarizer*/`).

## Data

- Source: CodeSearchNet (Python) functions from the bug-detection pipeline. Code never executes.
- Summary = first sentence of the original docstring (docstrings were removed from the code by the preprocessing step). Docstring noise is cleaned (parameter lists, links, doctests); too-short, too-long, empty and non-English summaries are dropped. Rows whose summary appears verbatim inside the code (comments, log strings) are removed to avoid leakage.
- Only clean (`label=0`) functions are used.
- Split: 70/15/15 by `source_id`, the shared `stratified_v1` split (applied with `apply_shared_split.py`; only the `split` column changes). Train 6,524, validation 1,384; test 1,400.
- Training reads train and validation only.

## Methods

| Method | Description |
|---|---|
| AST template | Fixed sentence built from the function name and whether the function returns a value. No training. |
| TF-IDF retrieval | Returns the summary of the most similar train function (TF-IDF over identifier sub-tokens). Index = train only. |
| BM25 retrieval | Same with BM25 ranking. |
| CodeT5-small full FT | `Salesforce/codet5-small`, all weights trained. lr 1e-4, batch 8, input 384 / target 48 tokens, seed 42, early stopping on validation loss (patience 2). |
| CodeT5-small LoRA (r8) | Same base model with frozen weights and LoRA adapters (`train_codet5_lora.py`): r=8, alpha=16, dropout 0.1, targets q and v, lr 2e-4, up to 8 epochs. Same data, tokenization, batch size, seed and stopping rule as full FT. fp32. |
| CodeT5-small LoRA (r16) | Extra run to check for headroom: r=16, alpha=16, targets q, k, v and o, lr 5e-4, up to 12 epochs. Everything else as r8. |

Metrics (`metrics.py`, the same for every method): corpus BLEU-4 and mean ROUGE-L F1 over lowercase word tokens (own pure-Python implementation, unsmoothed). Latency: one example per call (batch 1), warmed-up model, 100 examples, beam search 4. Artifact size: full model = whole checkpoint folder; LoRA = adapter weights and config only (the tokenizer and base model come from the base checkpoint).

## Comparison (validation, 1,384 examples)

Produced by `python -m src.summarization.evaluate_model --compare` (`results_summarization/comparison_validation.*`). Generation: 4 beams, default length penalty, same for both CodeT5 rows. The retrieval baselines index the train split only.

| Method | BLEU-4 | ROUGE-L F1 | Latency median (ms) | Latency mean (ms) | Artifact (MB) | Trainable params |
|---|---|---|---|---|---|---|
| AST template | 0.0068 | 0.2221 | — | — | — | — |
| TF-IDF retrieval | 0.1934 | 0.3270 | — | — | — | — |
| BM25 retrieval | 0.1804 | 0.3236 | — | — | — | — |
| CodeT5-small full FT | 0.1390 | 0.3796 | 112.2 | 121.8 | 245.12 | 60,492,288 |
| CodeT5-small LoRA (r8) | 0.0780 | 0.3369 | 119.1 | 123.8 | 1.19 | 294,912 |
| CodeT5-small LoRA (r16) † | 0.1066 | 0.3607 | 140.8 | 145.8 | 4.74 | 1,179,648 |

† Extra run, evaluated separately with `evaluate_model --lora-adapter ...` (`checkpoints/codet5_summarizer_lora_r16/metrics_validation.json`): same split, 1,384 examples, same generation settings. It is not yet a row of the `--compare` output files.

## Test results (1,400 examples, scored once)

From `results_summarization/comparison_test.md` and `.json`. Same settings as the validation table (4 beams, default length penalty); the LoRA row is the r16 adapter, chosen on validation. All rows have 1,400 examples.

| Method | BLEU-4 | ROUGE-L F1 | Latency median (ms) | Latency mean (ms) | Artifact (MB) | Trainable params |
|---|---|---|---|---|---|---|
| AST template | 0.0079 | 0.2237 | — | — | — | — |
| TF-IDF retrieval | 0.1840 | 0.3283 | — | — | — | — |
| BM25 retrieval | 0.1770 | 0.3251 | — | — | — | — |
| CodeT5-small full FT | 0.1398 | 0.3862 | 118.5 | 125.4 | 245.12 | 60,492,288 |
| CodeT5-small LoRA (r16) | 0.1066 | 0.3620 | 133.6 | 144.6 | 4.74 | 1,179,648 |

Other test numbers: exact match 0.0550 (full FT) and 0.0400 (LoRA r16); mean words in predictions 6.48 and 6.25 against 9.26 in the references; brevity penalty 0.6512 and 0.6175.

**Test conclusions**

- Full FT has the best ROUGE-L (0.3862) of all methods. LoRA r16 is next among the CodeT5 models: 0.0242 ROUGE-L and 0.0332 BLEU-4 below full FT, at 4.74 MB against 245.12 MB (about 52 times smaller) and 1.91% of the trainable parameters. Both are above the retrieval baselines by ROUGE-L (0.3283 and 0.3251).
- The retrieval baselines are higher by BLEU-4 (TF-IDF 0.1840, BM25 0.1770 against 0.1398 and 0.1066). They return existing training summaries and do not generate text for a new function; see Conclusions for possible reasons.
- Test agrees with validation: the ranking of methods by ROUGE-L is the same, and the numbers are close (full FT ROUGE-L 0.3796 on validation, 0.3862 on test; LoRA r16 0.3607 and 0.3620; full FT BLEU-4 0.1390 and 0.1398; LoRA r16 0.1066 and 0.1066). No sign of overfitting to the validation split.
- **Final model:** full FT stays the default for `summarize()` (auto backend). LoRA r16 is the small-artifact option (the `lora` backend uses it by default when its folder exists, otherwise r8). The test split was run once and was not used for any choice.

More numbers for the CodeT5 models:

| | Full FT | LoRA r8 | LoRA r16 |
|---|---|---|---|
| LoRA rank / alpha / targets | – | 8 / 16 / q,v | 16 / 16 / q,k,v,o |
| Exact match | 0.0549 | 0.0195 | 0.0412 |
| Mean words, prediction / reference | 6.42 / 9.27 | 5.96 / 9.27 | 6.19 / 9.27 |
| Brevity penalty | 0.6414 | 0.5735 | 0.6082 |
| Learning rate | 1e-4 | 2e-4 | 5e-4 |
| Best epoch (epochs run) | 4 (6, early stop) | 8 (8, schedule end) | 11 (12, schedule end) |
| Best validation loss | 2.5604 | 2.8336 | 2.6972 |
| Training time | 37.7 min | 43.0 min | 58.6 min |
| Mean seconds per step | 0.436 | 0.360 | 0.326 |
| Trainable / total parameters | 60,492,288 (full) | 294,912 of 60,787,200 (0.49%) | 1,179,648 of 61,671,936 (1.91%) |
| Adapter size | – | 1.19 MB | 4.74 MB |

All runs: Apple M4 (MPS), fp32, 6,524 train / 1,384 validation examples, batch 8, seed 42, `test_split_used: false`. LoRA epochs took 4.2–8.6 minutes each (varying); full FT 6.3 minutes each. Latency: batch 1, 100 examples, warmed up; the LoRA adapter was not merged for this measurement (merged latency: not measured). The r8 latency medians of two separate runs were 119.1 and 119.8 ms, so differences of a few ms are noise. The r16 median (140.8 ms) is about 20 ms higher, possibly because it has more adapter layers that are applied unmerged; this was not checked.

### Training curves (validation loss)

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Full FT | 2.7385 | 2.6171 | 2.5785 | **2.5604** | 2.5973 | 2.6248 | – | – |
| LoRA | 2.9466 | 2.9008 | 2.8787 | 2.8636 | 2.8486 | 2.8423 | 2.8365 | **2.8336** |

Full FT starts to overfit after epoch 4 (train loss keeps falling, validation loss rises). LoRA r8's validation loss was still falling slowly at epoch 8 (about 0.003 per epoch), but the learning rate also decays linearly to zero at epoch 8, so part of the flattening is the schedule.

LoRA r16 (validation loss per epoch 1–12): 2.8851, 2.8221, 2.7915, 2.7521, 2.7204, 2.7265, 2.7115, 2.7131, 2.7058, 2.7063, **2.6972** (epoch 11), 2.7030. It flattens after epoch 5 and the last epoch is slightly worse than epoch 11, while train loss keeps falling (3.843 to 2.486). Early stopping did not trigger (patience 2). The best r16 loss (2.6972) is 0.1364 below r8 but still 0.1368 above full FT.

### Generation settings (full FT)

Compared on validation only, with the selection rule fixed beforehand (`generation_selection_rule.md`):

| Variant | Beams | Length penalty | Min new tokens | BLEU-4 | ROUGE-L F1 |
|---|---|---|---|---|---|
| v0 | 4 | – | – | 0.1390 | 0.3796 |
| A | 4 | 1.5 | – | 0.1440 | 0.3787 |
| B | 4 | 2.0 | – | 0.1452 | 0.3789 |
| C | 4 | – | 8 | 0.1473 | 0.3710 |
| D | 8 | 1.5 | – | 0.1448 | 0.3791 |

`summarize()` uses variant B (4 beams, length penalty 2.0). The comparison table above uses v0 for both models, so it does not include this setting.

## Conclusions

- **Full FT is the best CodeT5 model here:** it beats LoRA on every quality number (BLEU-4 0.1390 vs 0.0780, ROUGE-L 0.3796 vs 0.3369, validation loss 2.560 vs 2.834) and has the best ROUGE-L of all five methods.
- **By BLEU-4 the retrieval baselines are ahead of both models:** TF-IDF 0.1934 and BM25 0.1804 against 0.1390. By ROUGE-L they are behind full FT (0.3270 and 0.3236 against 0.3796) but only slightly below LoRA (0.3369). So the ranking depends on the metric. Plausible reasons (not tested): retrieval returns real docstrings, which match 4-grams exactly when train and validation contain similar or templated functions (the split is by `source_id`, not by repository); the generative models write shorter text (6.4 and 6.0 words against 9.3 in the references), which lowers corpus BLEU through the brevity penalty (0.64 and 0.57). The length of the retrieval outputs was not measured here.
- **AST template** is far below everything else (BLEU-4 0.0068, ROUGE-L 0.2221), as expected for a fixed sentence.
- **LoRA trades quality for size.** r8: 1.19 MB against 245.12 MB (about 200 times smaller) and 0.49% of the parameters, at a cost of 0.0610 BLEU-4 and 0.0427 ROUGE-L against full FT. Latency is the same within noise (112.2 vs 119.1 ms median, unmerged adapter); training time was not shorter in these runs (43.0 and 58.6 vs 37.7 min, partly because LoRA ran all its epochs).
- **The r8 LoRA was undertrained or too small; r16 closes part of the gap.** The extra run (r16, q/k/v/o, lr 5e-4, 12 epochs) improved on r8 in every quality number: BLEU-4 0.1066 vs 0.0780, ROUGE-L 0.3607 vs 0.3369, validation loss 2.6972 vs 2.8336. It is still below full FT (BLEU-4 −0.0324, ROUGE-L −0.0189, validation loss +0.1368) at 4.74 MB (about 52 times smaller than full FT, 1.91% trainable parameters). The r16 ROUGE-L is above both retrieval baselines (0.3270, 0.3236); its BLEU-4 is still well below theirs (0.1934, 0.1804).
- **The r8/r16 comparison is not controlled.** Five things changed at once: rank (8 to 16), target modules (q,v to q,k,v,o), learning rate (2e-4 to 5e-4), maximum epochs (8 to 12), and alpha stayed 16 so the LoRA scaling alpha/r dropped from 2 to 1. The result shows that LoRA has headroom; it does not say which change helped. The r16 loss flattens from epoch 5 on, so more epochs alone are not the likely cause (not tested).
- **Final LoRA candidate (chosen on validation only): r16.** It is better than r8 on all validation metrics. Full FT remains the most accurate model, so it stays the default for `summarize()` (auto backend); r16 is the option when a small artifact matters.
- The differences come from a single run with one seed; no confidence intervals were computed.

## Error analysis

Based on the saved review files. LoRA (`checkpoints/codet5_summarizer_lora/review_validation.md`): mean ROUGE-L 0.3369, median 0.3077; buckets =0: 107, (0, 0.3): 566, [0.3, 0.6): 538, ≥0.6: 173. Groups: good 147, medium 371, bad 150, too_short 471. Full FT, for comparison: good 230, medium 364, bad 152, too_short 403. LoRA's extra losses are mostly in too_short and good.
Quotes are copied from the review file; code is shortened to the signature.

**1. Too short and generic (the biggest group: 471 of 1,384 for LoRA).** The model names the action but drops the object and conditions.

- `def parse_token(self, token, tags=[WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA]):`
  reference: "Returns the arguments for Sentence.append() from a tagged token representation."
  prediction: "Parse a token."
- `def update(sql, *args, **kwargs):` (asserts an update statement, executes it, returns `cursor.lastrowid`)
  reference: "Updates and commits with an insert sql statement, returns the record, but with a small chance of a race condition"
  prediction: "Update a row in the database."

**2. Summary follows the function name, not the behavior.**

- `def _getSmallestDifference(inputList, targetVal):`
  reference: "Returns the value in inputList that is closest to targetVal"
  prediction: "Get the smallest difference between two values."
- `def _validateParamsFor_validateChoice(choices, blank=False, strip=None, ...):` (a series of checks that raise `PySimpleValidateException`)
  reference: "Raises PySimpleValidateException if the arguments are invalid."
  prediction: "Validate a list of choices."

**3. Copying surface text from the code (identifiers and comments) without the purpose.**

- `def runcommand(cosmology='WMAP5'):` contains the comment "# Return the WMAP5 cosmology concentration predicted for"
  reference: "Example interface commands"
  prediction: "Return the WMAP5 cosmology concentration predicted for WMAP5."
- `def setup(app) -> Dict[str, Any]:` (many `app.add_config_value("uqbar_api_...")` calls)
  reference: "Sets up Sphinx extension."
  prediction: "Setup uqbar API configuration."

**4. Generic repeated outputs.** Most repeated LoRA predictions: "run the command." (3), "escape a string." (3), "main function." (3). First words: returns (160), get (134), return (87), create (64), add (49).

**5. Low score although the prediction is reasonable (reference noise or different wording).**

- reference: "A fingerprint for each vertex" / prediction: "Get the fingerprints of all the vertices and edges." (ROUGE-L 0.00)
- reference: "Creates new organization :rtype: Organization" / prediction: "Create a new organization." (0.44; the reference contains a leftover `:rtype:` tag)
- reference: "Function waitPuppetCatalogToBeApplied Wait for puppet catalog to be applied" / prediction: "Wait for puppet catalog to be applied" (0.88; the reference contains the function name)

Examples from the full-FT review show the same patterns, e.g. "Get state and exit code." for the reference "Returns RUNNING, -1 COMPLETE, 0 or EXECUTOR_ERROR, 255".

## Limitations

- CodeSearchNet docstrings are noisy: some are boilerplate, context rather than behavior, or contain leftovers. They are the only reference, so a good summary can still score low.
- BLEU-4 and ROUGE-L measure word overlap with one reference and are low by nature for this task (median reference: 8 words). Corpus BLEU-4 is unsmoothed. No semantic metric is reported yet.
- The split is by `source_id`, not by repository. Functions from the same repository, templated docstrings and near-duplicates can appear on both sides, which favors retrieval baselines and fine-tuned models.
- CodeT5 was pre-trained on CodeSearchNet, so the base model may have seen validation and test functions or their docstrings. Gains from fine-tuning cannot be separated from this overlap.
- Identical summaries for different functions are kept on purpose (not treated as duplicates).
- LoRA r16 was found by one extra run with five changed settings and chosen on validation; other ranks, learning rates or scalings were not tried, so LoRA may still be under-tuned. Its validation scores come from a separate evaluation call, not from the `--compare` output files.
- Latency was measured on one machine (Apple M4, MPS) and is not representative for other hardware.
- Test split: scored once, after all choices (generation settings, final LoRA) were fixed on validation. Results are from one seed and one run per method.

## Reproduction

See the "Code summarization" section of the README.

## Notes for the report

- Not added (would need new dependencies): METEOR, BERTScore/CodeBERTScore, a human rating of a sample.
- Extra LoRA run (done): `python -m src.summarization.train_codet5_lora --lora-r 16 --lora-targets qkvo --lr 5e-4 --epochs 12 --output checkpoints/codet5_summarizer_lora_r16`.
- `summarize()` with the `lora` backend uses `checkpoints/codet5_summarizer_lora_r16/best` if it exists, otherwise the r8 folder.

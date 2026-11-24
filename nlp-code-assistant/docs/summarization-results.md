# Code summarization results

Task: one-sentence English summary of a Python function (docstring removed from the input).
Status: validation results complete; **the test split has not been scored** (one final run is planned at the very end, **TBD**). Every number below comes from the saved result files (`results_summarization/`, `checkpoints/codet5_summarizer*/`).

## Data

- Source: CodeSearchNet (Python) functions from the bug-detection pipeline. Code never executes.
- Summary = first sentence of the original docstring (docstrings were removed from the code by the preprocessing step). Docstring noise is cleaned (parameter lists, links, doctests); too-short, too-long, empty and non-English summaries are dropped. Rows whose summary appears verbatim inside the code (comments, log strings) are removed to avoid leakage.
- Only clean (`label=0`) functions are used.
- Split: 70/15/15 by `source_id`, the shared `stratified_v1` split (applied with `apply_shared_split.py`; only the `split` column changes). Train 6,524, validation 1,384; test: TBD (not opened).
- Training reads train and validation only.

## Methods

| Method | Description |
|---|---|
| AST template | Fixed sentence built from the function name and whether the function returns a value. No training. |
| TF-IDF retrieval | Returns the summary of the most similar train function (TF-IDF over identifier sub-tokens). Index = train only. |
| BM25 retrieval | Same with BM25 ranking. |
| CodeT5-small full FT | `Salesforce/codet5-small`, all weights trained. lr 1e-4, batch 8, input 384 / target 48 tokens, seed 42, early stopping on validation loss (patience 2). |
| CodeT5-small LoRA | Same base model with frozen weights and LoRA adapters (`train_codet5_lora.py`): r=8, alpha=16, dropout 0.1, targets q and v, lr 2e-4. Same data, tokenization, batch size, seed and stopping rule as full FT. fp32. |

Metrics (`metrics.py`, the same for every method): corpus BLEU-4 and mean ROUGE-L F1 over lowercase word tokens (own pure-Python implementation, unsmoothed). Latency: one example per call (batch 1), warmed-up model, 100 examples, beam search 4. Artifact size: full model = whole checkpoint folder; LoRA = adapter weights and config only (the tokenizer and base model come from the base checkpoint).

## Comparison (validation, 1,384 examples)

Produced by `python -m src.summarization.evaluate_model --compare` (`results_summarization/comparison_validation.*`). Generation: 4 beams, default length penalty, same for both CodeT5 rows. The retrieval baselines index the train split only.

| Method | BLEU-4 | ROUGE-L F1 | Latency median (ms) | Latency mean (ms) | Artifact (MB) | Trainable params |
|---|---|---|---|---|---|---|
| AST template | 0.0068 | 0.2221 | — | — | — | — |
| TF-IDF retrieval | 0.1934 | 0.3270 | — | — | — | — |
| BM25 retrieval | 0.1804 | 0.3236 | — | — | — | — |
| CodeT5-small full FT | 0.1390 | 0.3796 | 112.2 | 121.8 | 245.12 | 60,492,288 |
| CodeT5-small LoRA | 0.0780 | 0.3369 | 119.1 | 123.8 | 1.19 | 294,912 |

Test split: **TBD** (one time, at the very end).

More numbers for the two CodeT5 models:

| | Full FT | LoRA (r=8, q,v) |
|---|---|---|
| Exact match | 0.0549 | 0.0195 |
| Mean words, prediction / reference | 6.42 / 9.27 | 5.96 / 9.27 |
| Brevity penalty | 0.6414 | 0.5735 |
| Learning rate | 1e-4 | 2e-4 |
| Best epoch (epochs run) | 4 (6, early stop) | 8 (8, schedule end) |
| Best validation loss | 2.5604 | 2.8336 |
| Training time | 37.7 min | 43.0 min |
| Mean seconds per step | 0.436 | 0.360 |
| Trainable / total parameters | 60,492,288 (full) | 294,912 of 60,787,200 (0.49%) |

Both runs: Apple M4 (MPS), fp32, 6,524 train / 1,384 validation examples, batch 8, seed 42, `test_split_used: false`. LoRA epochs took 4.2–8.6 minutes each (varying); full FT 6.3 minutes each. Latency: batch 1, 100 examples, warmed up; the LoRA adapter was not merged for this measurement (merged latency: not measured). The LoRA latency medians of two separate runs were 119.1 and 119.8 ms, so differences of a few ms are noise.

### Training curves (validation loss)

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Full FT | 2.7385 | 2.6171 | 2.5785 | **2.5604** | 2.5973 | 2.6248 | – | – |
| LoRA | 2.9466 | 2.9008 | 2.8787 | 2.8636 | 2.8486 | 2.8423 | 2.8365 | **2.8336** |

Full FT starts to overfit after epoch 4 (train loss keeps falling, validation loss rises). LoRA's validation loss was still falling slowly at epoch 8 (about 0.003 per epoch), but the learning rate also decays linearly to zero at epoch 8, so part of the flattening is the schedule.

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
- **LoRA trades quality for size:** 1.19 MB against 245.12 MB (about 200 times smaller) and 0.49% of the parameters, at a cost of 0.0610 BLEU-4 and 0.0427 ROUGE-L. Latency is the same within noise (112.2 vs 119.1 ms median, unmerged adapter), and training time was not shorter in this run (43.0 vs 37.7 min, mostly because LoRA ran all 8 epochs).
- **LoRA may be undertrained:** its best epoch was the last one. One extra run with more capacity and epochs would show whether the gap is real (see the command in the notes at the end of this file). Until then, the LoRA numbers are a lower bound for this method.
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
- Latency was measured on one machine (Apple M4, MPS) and is not representative for other hardware.
- Test split: not evaluated. One final run is planned after all choices are fixed (`--split test --confirm-final-test`).

## Reproduction

See the "Code summarization" section of the README.

## Notes for the report

- Not added (would need new dependencies): METEOR, BERTScore/CodeBERTScore, a human rating of a sample.
- Optional extra LoRA run (not done): `python -m src.summarization.train_codet5_lora --lora-r 16 --lora-targets qkvo --lr 5e-4 --epochs 12 --output checkpoints/codet5_summarizer_lora_r16`.

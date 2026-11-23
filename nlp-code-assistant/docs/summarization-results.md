# Code summarization results

Task: one-sentence English summary of a Python function (docstring removed from the input).
Status: **in progress.** Numbers marked **TBD** have not been measured on the final split yet and are filled in from the saved result files. Nothing has been scored on the test split.

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

Produced by `python -m src.summarization.evaluate_model --compare`. Generation: 4 beams, default length penalty.

| Method | BLEU-4 | ROUGE-L F1 | Latency median (ms) | Latency mean (ms) | Artifact (MB) | Trainable params |
|---|---|---|---|---|---|---|
| AST template | TBD | TBD | — | — | — | — |
| TF-IDF retrieval | TBD | TBD | — | — | — | — |
| BM25 retrieval | TBD | TBD | — | — | — | — |
| CodeT5-small full FT | 0.1390 | 0.3796 | TBD | TBD | TBD | TBD |
| CodeT5-small LoRA | TBD | TBD | TBD | TBD | TBD | TBD |

The full-FT row is the saved result of the earlier run (`metrics_validation_v0.json`; latency, size and parameters were not recorded then). Baseline numbers from before the switch to the shared split are not reused because the split changed.

### Full fine-tuning: training and generation settings

Best epoch 4 of 6 run (early stopping), validation loss 2.560, 37.7 minutes on Apple M4 (MPS), 0.44 s per step.
Generation variants were compared on validation only, with the selection rule fixed beforehand (`generation_selection_rule.md`):

| Variant | Beams | Length penalty | Min new tokens | BLEU-4 | ROUGE-L F1 |
|---|---|---|---|---|---|
| v0 | 4 | – | – | 0.1390 | 0.3796 |
| A | 4 | 1.5 | – | 0.1440 | 0.3787 |
| B | 4 | 2.0 | – | 0.1452 | 0.3789 |
| C | 4 | – | 8 | 0.1473 | 0.3710 |
| D | 8 | 1.5 | – | 0.1448 | 0.3791 |

`summarize()` uses variant B (4 beams, length penalty 2.0). LoRA training time, best epoch and loss: TBD.

## Error analysis

From `review_examples.py` on the full-FT validation predictions (1,384 examples). ROUGE-L buckets: 0 for 96, (0, 0.3) for 521, [0.3, 0.6) for 502, ≥ 0.6 for 265. Mean prediction length 6.4 words against 9.3 for references, so outputs are on the short side. Frequent first words: get, return, returns, create, add.

| Case | Reference | Prediction | Comment |
|---|---|---|---|
| good (0.95) | Function enhance Enhance the object with new item or enhanced items | Function enhance Enhance the object with new or enhanced items | Near copy of a templated docstring. |
| bad (0.00) | Based on example | mptt_before_insert will create a new row from the database. | Reference is uninformative; the model describes a name found in the code. |
| bad (0.00) | Returns RUNNING, -1 COMPLETE, 0 or EXECUTOR_ERROR, 255 | Get state and exit code. | Reference lists return values; the prediction is a plausible paraphrase. |
| bad (0.14) | The inside of run's infinite loop. | Run the thread and reset _current_tups. | Reference is context, not behavior. |
| too short (0.11) | Force use of accuracy score since we don't inherit from ClassifierMixin | Return the score of the model. | Misses the reason that the reference gives. |
| too short (0.26) | Due to XML content that will not conform with the strict JSON schema validation rules, for elife articles only, rewrite the JSON to make it valid | rewrite json for elife articles | Correct but much shorter than a long reference. |

Many low scores are acceptable summaries that differ in wording from the original docstring. LoRA error examples: TBD.

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

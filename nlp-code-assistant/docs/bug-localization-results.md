# Bug localization without reference code

## Method

The classifiers say *whether* a function looks buggy; localization asks *where*. For every
operator in the function, `src/bug_detection/localize.py` proposes each operator that could
have been mutated into it (the mutation tables in `src/data/mutations.py`, read backwards: a
`>` may be a broken `<` or `>=`). Every candidate repair is a full function with one operator
flipped. The original function and all candidates are scored by the classifier in one batch,
and candidates are ranked by how much they lower the predicted bug probability (`drop`). The
first candidate is the suspected bug; its line and its fix are the explanation. No reference
code and no execution are involved.

`src/evaluation/localization_eval.py` runs this on every buggy function of the shared test
split (`stratified_v1`) whose original is also in the split (the original is the true fix),
and records where the true fix lands in the ranking.

- **top-1 / top-3:** the true fix is the first / among the first three candidates.
- **Correct line first:** the first candidate edits the line that was mutated (even if it
  proposes a different operator).
- **Random baseline:** expected top-1 / top-3 when a candidate is picked uniformly at random,
  averaged over functions (1/n and min(3, n)/n for n candidates).

## Results (711 test mutants)

CodeBERT is the seed-42 model of the new training configuration (batch 16, fp16, 5 epochs).
TF-IDF is `models/tfidf_v2`. Numbers: `results/localization/{codebert,tfidf}_summary.json`
(CodeBERT from a Colab T4 run).

| | CodeBERT | TF-IDF | Random choice |
|---|---:|---:|---:|
| top-1, exact fix | **78.2%** | 67.8% | 48.8% |
| top-3 | **92.3%** | 86.9% | 78.6% |
| Correct line ranked first | **86.2%** | 78.1% | — |
| True fix missing from candidates | 0 | 0 | — |

By mutation type (top-1):

| Mutation type | n | CodeBERT | TF-IDF | Random |
|---|---:|---:|---:|---:|
| Comparison (`==`, `<`, ...) | 303 | **76.9%** | 67.0% | 48.0% |
| Arithmetic (`+`, `-`, `*`, `/`) | 277 | **77.6%** | 68.2% | 46.6% |
| Boolean (`and`, `or`) | 131 | **82.4%** | 68.7% | 55.3% |

Split by whether the classifier itself flagged the function as BUGGY (threshold 0.5):

| | n | top-1 | Correct line first | Random top-1 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| CodeBERT, flagged BUGGY | 529 | 86.2% | 90.7% | — | 4.0 |
| CodeBERT, missed | 182 | 54.9% | — | 34.1% | 7.2 |
| TF-IDF, flagged BUGGY | 480 | 62.1% | 71.5% | 39.9% | 6.1 |
| TF-IDF, missed | 231 | 79.7% | 91.8% | 67.5% | 2.2 |

## CodeBERT vs TF-IDF: McNemar test

Both models are scored on the same 711 mutants, so the comparison is paired
(`src/evaluation/localization_compare.py`, output `results/localization/comparison.json`).
Only the mutants where exactly one model is right count:

| Metric | CodeBERT | TF-IDF | Only CodeBERT right | Only TF-IDF right | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| top-1 | 78.2% | 67.8% | 112 | 38 | 1.2 × 10⁻⁹ |
| top-3 | 92.3% | 86.9% | 59 | 21 | 2.5 × 10⁻⁵ |
| Correct line first | 86.2% | 78.1% | 89 | 31 | 1.1 × 10⁻⁷ |

The top-1 difference is not noise: CodeBERT alone finds the fix on 112 mutants, TF-IDF alone on
38 (both right on 444, both wrong on 117). top-1 is the pre-planned test; top-3 and line are
shown for reference.

## Conclusions

- Both classifiers localize far better than chance; CodeBERT is about 10 points ahead of
  TF-IDF in top-1 (78.2% vs 67.8%, McNemar p = 1.2 × 10⁻⁹) and correct-line accuracy
  (86.2% vs 78.1%), and ahead for every mutation type.
- When CodeBERT flags a function, its top repair is the exact fix 86.2% of the time and is on
  the right line 90.7% of the time, so the suggested fix is most useful exactly when it is shown.
- Even on functions it missed, CodeBERT's ranking is informative (54.9% vs 34.1% random).
- The two models fail on different functions: CodeBERT misses mostly long functions (7.2
  candidates on average against 4.0 for the ones it flags), TF-IDF misses mostly short ones
  (2.2 candidates), where localization is easy for any ranking.

## Limitations

- **Synthetic bugs only.** Every bug is one operator from the mutation tables, and the
  candidate generator inverts exactly those tables, so the true fix is always among the
  candidates (0 missing). Real bugs can change other tokens, several tokens, or none of
  these operators; this evaluation does not measure that.
- **No candidates for `<=` and `>=`.** Nothing in the tables mutates *into* `<=` or `>=`,
  so an operator like that in the code never gets a repair candidate (a real `<=` bug would
  go unnoticed).
- A changed operator does not prove a behavioural bug; the "fix" is the original CodeSearchNet
  code, which may itself be wrong.
- CodeBERT was pretrained on CodeSearchNet and may have seen the test functions.
- One CodeBERT seed (42) was used for localization; the classification seeds differ by about
  ±0.007 F1, localization variance across seeds was not measured.

## Reproduce

```bash
# TF-IDF, on the laptop (fast)
OMP_NUM_THREADS=2 python -m src.evaluation.localization_eval --model tfidf
# CodeBERT, in Colab (T4)
python -m src.evaluation.localization_eval --model codebert --model-dir models/codebert_lora_seed42 --device cuda
```

Outputs: `results/localization/<model>_rows.csv` (one row per mutant) and
`results/localization/<model>_summary.json`. Then, on the laptop:

```bash
OMP_NUM_THREADS=2 python -m src.evaluation.localization_compare   # -> comparison.json
```

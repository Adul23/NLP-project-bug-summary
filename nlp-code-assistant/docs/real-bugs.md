# Real bugs: evaluation on CTSSB-1M operator fixes

Evaluation only; nothing here is used for training.

## Data

[CTSSB-1M](https://github.com/cedricrupb/TSSB3M) (CC BY 4.0) lists about 1M single-statement
bug fixes from Python projects on GitHub. Each record has `project_url`, `commit_sha` (the
fix), `parent_sha` (the buggy version), `file_path`, the changed statement (`before`,
`after`), `in_function`, `comodified` and `sstub_pattern`. The code itself is not included.

`scripts/collect_real_bugs.py` turns records with the pattern `CHANGE_BINARY_OPERATOR` into
buggy/fixed function pairs:

1. keep records inside a function and with only one changed statement in the file
   (`in_function`, not `comodified`), and whose statement differs in exactly one token, both
   tokens from our operators (`== != < > <= >= + - * / and or`);
2. download both file versions from raw.githubusercontent.com (0.5 s pause, disk cache);
3. find the one function (or method) that differs and require that the only AST difference is
   one operator from that set;
4. drop duplicates, functions already in our CodeSearchNet dataset, and more than 5 pairs per
   repository; stop at 500 pairs.

Output: `data/real_bugs/ctssb_binop.csv` (`sample_id, source_id, code, label, mutation_type,
repo, in_tables`), label 1 before the fix and 0 after it, plus `ctssb_binop.stats.json` with
how many records each filter dropped. The CSV holds third-party code and is not committed.

`in_tables` marks pairs where fixed → buggy is one of the training mutations (for example
`<=` → `<`). Other pairs (for example `<` → `<=`, or a real `<=` bug) never occur in the
synthetic data, and localization cannot propose their fix: there are no repair candidates
for `<=` and `>=`. Results are reported for all pairs and for each group.

## Results

500 pairs from 488 repositories (1,000 functions): 344 pairs whose fix → bug change is a
training mutation (`in_tables`), 156 that are not (152 comparison, 4 arithmetic). The most
common of these: a correct `>`/`<` turned into `>=`/`<=` (53 pairs; the tables only mutate the
other way) and `==` swapped with an ordering comparison (35 pairs). Collection funnel
(`ctssb_binop.stats.json`): 9,193 `CHANGE_BINARY_OPERATOR` records → 5,640 single-statement
candidates with our operators → 500 pairs (188 files did not parse, 44 downloads failed,
9 duplicates, 0 overlaps with our CodeSearchNet data). CodeBERT is the same seed-42 model as
on the synthetic test; TF-IDF is `models/tfidf_v2`. Nothing was retrained.

### Classification: does the model tell the buggy version from the fix?

Each pair contributes both versions (label 1 before the fix, 0 after), threshold 0.5.
95% bootstrap intervals in brackets.

| | ROC AUC | Accuracy | F1 (BUGGY) | MCC |
|---|---:|---:|---:|---:|
| CodeBERT, all 1,000 | 0.530 [0.493, 0.566] | 0.521 | 0.462 | 0.043 |
| CodeBERT, `in_tables` (688) | 0.556 [0.515, 0.602] | 0.541 | 0.509 | 0.082 |
| CodeBERT, not `in_tables` (312) | 0.472 [0.410, 0.542] | 0.478 | 0.340 | −0.049 |
| TF-IDF, all 1,000 | 0.503 [0.465, 0.538] | 0.504 | 0.585 | 0.009 |
| TF-IDF, `in_tables` (688) | 0.509 [0.464, 0.550] | 0.509 | 0.595 | 0.019 |

For comparison, on the synthetic test split CodeBERT reaches BUGGY F1 0.76.

- On real bugs neither classifier separates the buggy version from the fix: TF-IDF is at
  chance (AUC 0.50), CodeBERT barely above it, and only on the pairs that look like training
  mutations (AUC 0.556, interval just above 0.5).
- TF-IDF's F1 of 0.585 is not a signal: it flags 70% of all functions as BUGGY whether or not
  they are (accuracy 0.50); always answering BUGGY would score F1 0.667.
- CodeBERT is badly calibrated here (ECE 0.17): functions it gives over 90% bug probability
  are buggy only 59% of the time.

### Localization: given the buggy version, is the true fix ranked first?

| | CodeBERT | TF-IDF | Random choice |
|---|---:|---:|---:|
| `in_tables` (344): top-1 | **44.2%** | 37.8% | 28.5% |
| `in_tables`: top-3 | **78.2%** | 69.5% | 61.3% |
| `in_tables`: correct line first | 61.6% | 58.4% | — |
| not `in_tables` (156): top-1 / top-3 | 0% | 0% | — (fix never a candidate) |
| All 500: top-1 | 30.4% | 26.0% | — |
| Synthetic test (711): top-1 | 78.2% | 67.8% | 48.8% |

McNemar on the same 500 pairs (`results/real_bugs/comparison.json`): CodeBERT finds the exact
fix first on 60 pairs that TF-IDF misses, TF-IDF on 38 that CodeBERT misses (exact p = 0.033);
top-3: 52 vs 22 (p = 0.00064); correct line: 56 vs 44 (p = 0.27, not significant).

- Localization still carries signal on real bugs of the trained kind: CodeBERT ranks the
  exact fix first in 44% of cases against 28.5% for a random choice, and significantly better
  than TF-IDF. But this is far from the 78% on synthetic mutants.
- 31% of real single-operator fixes (156 of 500) are changes the candidate generator cannot
  propose at all, because the mutation tables have no source for `<=` and `>=`.
- The random baseline is only meaningful where the fix is among the candidates; over all 500
  pairs it is 19.6% (28.5% × 344/500), so the 32.1% printed in the summary for `all` overstates it.
- Three `in_tables` pairs have no matching candidate (rendering differences); they count as
  misses.

### Caveats

- One model per type, one seed; 500 pairs. Intervals are wide (about ±0.04 AUC).
- CTSSB-1M labels a commit as a fix heuristically (`likely_bug` was not required), so some buggy versions may be intentional changes.
- Functions are long (median 945 characters); CodeBERT sees only the first 512 tokens, and the
  changed operator can fall outside them.

## Reproduce (Colab)

```bash
wget -q -O /content/ctssb_data_1M.zip https://tssb3m.s3.eu-west-1.amazonaws.com/ctssb_data_1M.zip
python scripts/collect_real_bugs.py --input /content/ctssb_data_1M.zip --list-patterns
python scripts/collect_real_bugs.py --input /content/ctssb_data_1M.zip --limit 500
python -m src.evaluation.localization_eval --model tfidf --data data/real_bugs/ctssb_binop.csv
python -m src.evaluation.localization_eval --model codebert --model-dir models/codebert_lora_seed42 \
    --device cuda --data data/real_bugs/ctssb_binop.csv
```

Outputs in `results/real_bugs/`: `<model>_rows.csv` and `<model>_summary.json` (localization)
and `<model>_classifier.json` (classification metrics, overall and by `in_tables`).

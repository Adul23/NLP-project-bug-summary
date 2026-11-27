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

To be filled in after the Colab run (`results/real_bugs/`).

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

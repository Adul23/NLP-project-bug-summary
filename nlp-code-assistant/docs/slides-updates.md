# Slide updates

Ready-to-paste texts for the presentation (the deck is not in this repository). Each block
says what it replaces. Sources: `results/codebert_seeds.md`, `docs/bug-localization-results.md`,
`docs/summarization-results.md`, `results/evaluation_report.md`.

## 1. Classification results

Replace "CodeBERT + LoRA: F1 0.754" with:

> **CodeBERT + LoRA: BUGGY F1 0.764 ± 0.007 (3 seeds)**, accuracy 0.848 ± 0.007
> TF-IDF + LR: BUGGY F1 0.585 (one deterministic model)

Optional seed table:

| Seed | F1 | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| 42 | 0.756 | 0.845 | 0.768 | 0.744 |
| 13 | 0.767 | 0.843 | 0.731 | 0.807 |
| 7 | 0.768 | 0.857 | 0.800 | 0.738 |
| Mean ± std | 0.764 ± 0.007 | 0.848 ± 0.007 | 0.766 ± 0.035 | 0.763 ± 0.038 |

Speaker note: the old single run (F1 0.754) is inside the seed spread, so the new training
setup did not change quality. Precision and recall vary more between seeds than F1.

## 2. Training

Replace the old training line ("batch 1, fp32, 66 minutes") with:

> **11 minutes per run on a Colab T4** (was 66): batch 16, fp16 mixed precision,
> length-grouped batches, 5 epochs with early stopping. **6× faster, same F1.**

## 3. New slide: bug localization

Title: **Where is the bug? Localization without reference code**

> For each operator, try the operators it could have been mutated from, score all
> candidates in one batch, and pick the one that lowers the bug probability the most.

| 711 test mutants | CodeBERT | TF-IDF | Random |
|---|---:|---:|---:|
| Exact fix ranked 1st | **78.2%** | 67.8% | 48.8% |
| Exact fix in top 3 | **92.3%** | 86.9% | 78.6% |
| Correct line ranked 1st | **86.2%** | 78.1% | — |

Optional line: *When CodeBERT flags a function as buggy, its first repair is the exact fix
86.2% of the time (correct line 90.7%).*

Under the table: *CodeBERT vs TF-IDF top-1: McNemar p = 1.2 × 10⁻⁹ (CodeBERT alone right on
112 mutants, TF-IDF alone on 38).*

## 4. Product / API

Replace the API slide text with:

> **`POST /analyze`**: BUGGY/CLEAN label and bug probability from TF-IDF or CodeBERT, plus an
> explanation:
> - with the original code: a **verified** explanation of the exact operator change;
> - without it (bug probability ≥ 0.40): the **suspected line and fix** from localization,
>   e.g. *"Line 2: `<` may be `>`"*, marked **not verified**; the UI highlights the line.
>
> Optional `summarize: true` adds a one-sentence CodeT5 summary.
> **`POST /summarize`**: one-sentence summary from the fully fine-tuned CodeT5-small.
> **Feedback loop**: users mark predictions right/wrong; CodeBERT retrains on feedback
> in the background and is swapped in only if validation F1 drops by at most 0.01.

Speaker note: the localized fix is shown only if flipping that operator lowers the bug
probability; on the test mutants the first repair is the exact fix 78.2% of the time
(86.2% when CodeBERT flags the function).

Not in the product yet (do not claim): generated explanations from the CodeT5 LoRA
"explainer" (it is not trained; `explanation: "lora"` returns 503).

## 5. Limitations

Remove: "one seed".

Add:

> - Trained on synthetic single-operator mutations only. On 500 real operator fixes from GitHub
>   (CTSSB-1M) the classifiers are near chance: CodeBERT ROC AUC 0.53, TF-IDF 0.50.
> - Localization transfers partly: on real fixes of the trained kind CodeBERT ranks the exact fix
>   first 44% of the time (random 28.5%, synthetic 78%); 31% of real fixes cannot be proposed.
> - CodeBERT and CodeT5 were pretrained on CodeSearchNet and may have seen the test functions.
> - Latencies were measured on different hardware (CodeBERT and TF-IDF on a laptop CPU,
>   CodeT5 on an Apple M4) and are not directly comparable.
> - Localization only proposes operators from the mutation tables; `<=` and `>=` never get a
>   candidate.

## 6. New slide: real bugs

Title: **Does it work on real bugs?**

> 500 real single-operator fixes from 488 GitHub projects (CTSSB-1M), evaluation only.

| | CodeBERT | TF-IDF | Random |
|---|---:|---:|---:|
| Buggy vs fixed, ROC AUC | 0.53 | 0.50 | 0.50 |
| Exact fix ranked 1st (fixable pairs) | **44%** | 38% | 28.5% |
| Exact fix in top 3 (fixable pairs) | **78%** | 70% | 61% |

Speaker note: synthetic F1 0.76 does not carry over. The detector learned what our
mutations look like, not what real bugs look like. Localization is the part that still helps,
and CodeBERT is significantly better than TF-IDF there (McNemar p = 0.033).

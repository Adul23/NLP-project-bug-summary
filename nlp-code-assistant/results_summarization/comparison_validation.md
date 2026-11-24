Split: validation (1384 examples). Same dataset, split and metrics for every row.

| Method | BLEU-4 | ROUGE-L F1 | Latency median (ms) | Latency mean (ms) | Artifact (MB) | Trainable params |
|---|---|---|---|---|---|---|
| AST template | 0.0068 | 0.2221 | — | — | — | — |
| TF-IDF retrieval | 0.1934 | 0.3270 | — | — | — | — |
| BM25 retrieval | 0.1804 | 0.3236 | — | — | — | — |
| CodeT5-small full FT | 0.1390 | 0.3796 | 112.2 | 121.8 | 245.12 | 60,492,288 |
| CodeT5-small LoRA | 0.0780 | 0.3369 | 119.1 | 123.8 | 1.19 | 294,912 |

Latency: one example per call (batch 1), warmed-up model, beam search as set for the run. '—' = not measured or not applicable (baselines have no neural model); 'TBD' = checkpoint not available yet.

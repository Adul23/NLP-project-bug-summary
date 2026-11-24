Split: test (1400 examples). Same dataset, split and metrics for every row.

| Method | BLEU-4 | ROUGE-L F1 | Latency median (ms) | Latency mean (ms) | Artifact (MB) | Trainable params |
|---|---|---|---|---|---|---|
| AST template | 0.0079 | 0.2237 | — | — | — | — |
| TF-IDF retrieval | 0.1840 | 0.3283 | — | — | — | — |
| BM25 retrieval | 0.1770 | 0.3251 | — | — | — | — |
| CodeT5-small full FT | 0.1398 | 0.3862 | 118.5 | 125.4 | 245.12 | 60,492,288 |
| CodeT5-small LoRA | 0.1066 | 0.3620 | 133.6 | 144.6 | 4.74 | 1,179,648 |

Latency: one example per call (batch 1), warmed-up model, beam search as set for the run. '—' = not measured or not applicable (baselines have no neural model); 'TBD' = checkpoint not available yet.

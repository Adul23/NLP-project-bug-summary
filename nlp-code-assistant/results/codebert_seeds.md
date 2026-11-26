# CodeBERT + LoRA: three seeds, new training configuration

Configuration: batch 16, fp16 on a Colab T4, 5 epochs, length-grouped batches; other
settings assumed at the `train_lora.py` defaults (learning rate 2e-4 with 6% linear warm-up,
early stopping with patience 2, LoRA r=8 on query/value). About 11 minutes per run, against
66 minutes for the old configuration (batch 1 with 8-step accumulation, fp32, 3 epochs).

Test split (`stratified_v1`, 2,211 rows), BUGGY class, threshold 0.5:

| Seed | F1 (BUGGY) | Accuracy | Precision | Recall | Best epoch |
|---|---:|---:|---:|---:|---:|
| 42 | 0.756 | 0.845 | 0.768 | 0.744 | 5 |
| 13 | 0.767 | 0.843 | 0.731 | 0.807 | 4 |
| 7 | 0.768 | 0.857 | 0.800 | 0.738 | 5 |
| **Mean ± std** | **0.764 ± 0.007** | 0.848 ± 0.007 | 0.766 ± 0.035 | 0.763 ± 0.038 | |

std is the sample standard deviation (n − 1). Old run for reference (seed 42, old
configuration): F1 0.754, accuracy 0.845.

Conclusion: quality is unchanged (the old run is within the spread between seeds), and
training is about 6 times faster. Precision and recall move more between seeds than F1:
seed 13 trades precision for recall.

Source: the three `metrics.json` files from the Colab runs. They are not in the repository
yet; once copied to `results/codebert_seeds/<seed>/metrics.json`, this table should be
regenerated from them. Recomputed from the rounded values above, the accuracy std is 0.008,
so the 0.007 should be checked against the unrounded numbers.

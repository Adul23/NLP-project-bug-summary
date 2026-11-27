# Python Bug Detection and Code Summarization

Course NLP project: detect single-operator bugs in Python functions, point to the suspected
line and fix, and summarize what a function does. Source code is never executed.

Everything lives in [`nlp-code-assistant/`](nlp-code-assistant/); its
[README](nlp-code-assistant/README.md) is the full documentation (setup, training, API,
web interface, metrics, limitations).

## Status (2026-11-27)

- **Bug detection.** TF-IDF + Logistic Regression (test BUGGY F1 0.585) and CodeBERT + LoRA
  (test BUGGY F1 0.764 ± 0.007 over three seeds) on synthetic operator mutations from
  CodeSearchNet.
- **Bug localization.** Ranking single-operator repairs by the drop in bug probability puts the
  exact fix first for 78.2% of synthetic test mutants with CodeBERT (TF-IDF 67.8%, random 48.8%).
- **Real bugs.** On 500 real operator fixes from GitHub (CTSSB-1M) the classifiers are near
  chance (CodeBERT ROC AUC 0.53, TF-IDF 0.50); localization keeps some signal (exact fix first
  in 44% of fixable cases vs 28.5% at random). See
  [docs/real-bugs.md](nlp-code-assistant/docs/real-bugs.md).
- **Code summarization.** CodeT5-small, fully fine-tuned and LoRA-tuned; the full fine-tune
  serves `/summarize`. See
  [docs/summarization-results.md](nlp-code-assistant/docs/summarization-results.md).
- **Inference on GPU.** fp16 autocast makes CodeBERT about 4x faster at batch 16 on a T4 with
  identical labels on 200 validation functions.
- **Product.** FastAPI backend and React frontend with feedback-driven CodeBERT retraining.

## Quick start

```bash
cd nlp-code-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.api.app:app --port 8000
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Model artifacts, data preparation and Colab training commands are described in
[nlp-code-assistant/README.md](nlp-code-assistant/README.md).

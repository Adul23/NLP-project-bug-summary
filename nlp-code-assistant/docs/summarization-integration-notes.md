# Summarization: integration notes for the API (plan only)

Nothing in the teammate's files was changed. This document lists what is needed to add `POST /summarize`.

## What exists today

- `src/api/app.py` (uncommitted local edits in the working tree, not mine) adds a `_LocalSummarizer` marked "LOCAL TEST ONLY". For the `summarizer` model name it calls `src.summarization.summarize.summarize(code)` and returns it as `explanation` with `explanation_source: "lora_generated_unverified"`. It is used only by `/predict` with `explanation == "lora"`.
- Without that edit, `/predict` loads `LoraPredictor(models/codet5_lora, summarizer=True)` (from `src/inference/models.py`), which generates a **mutation explanation** from a prompt in `train_lora.py`. That is a different task (explain a bug), with a different model and folder.
- `summarize(code)` keeps its interface `summarize(code: str) -> str`. It raises `ValueError` for empty, non-string or too long input (> 20,000 characters) and `FileNotFoundError` if no checkpoint exists.

## Minimal change: new endpoint

Add to `src/api/app.py` (the schema goes next to the other request models, the route inside `create_app`):

```python
class SummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=1, max_length=20_000)
    backend: Literal["auto", "full", "lora"] = "auto"


    # inside create_app(), next to the other routes
    @api.post("/summarize")
    def summarize_code(request: SummarizeRequest):
        from src.summarization.summarize import summarize

        if not request.code.strip():
            raise HTTPException(422, "code is empty")
        try:
            summary = summarize(request.code, backend=request.backend)
        except FileNotFoundError as exc:
            raise HTTPException(
                503, "summarizer unavailable: train a CodeT5 checkpoint; see README"
            ) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (ImportError, RuntimeError, OSError) as exc:
            raise HTTPException(503, "summarizer unavailable") from exc
        return {"summary": summary, "backend": request.backend, "verified": False}
```

Notes:
- `summarize()` loads the model lazily once per checkpoint and is thread-safe for loading; generation itself is not batched. Run it in a plain `def` route (as above) so FastAPI uses its thread pool.
- The first request is slow (model load). A `/health` entry for the summarizer could report `loaded` without loading.
- Submitted code is only tokenized, never executed.

## Optional: replace the temporary `_LocalSummarizer`

If the `/predict` explanation should be a plain summary for non-mutation inputs, `_LocalSummarizer.generate` can stay as it is; it already calls `summarize(code)`. To pick a backend there, pass `backend=` to `summarize`. Remove the "LOCAL TEST ONLY" marker only after agreeing the behaviour below.

## To agree with the teammate

1. Two different "summarizers" exist: the mutation-explanation LoRA (`models/codet5_lora`, `train_lora.py`) and the code-summary model (`checkpoints/codet5_summarizer*`). Use different names in the API (`explanation` vs `summary`) so they are not mixed up.
2. Where checkpoints live: `checkpoints/` (untracked, my default) or `models/` (his convention, `BUG_MODEL_ROOT`). The env var `CODET5_SUMMARIZER_CHECKPOINT` already lets the API point elsewhere.
3. Which backend is the default in the API (`auto` = full if present, else LoRA) once results are in; this depends on the comparison table.
4. Dependencies: `peft` and `transformers` from `requirements-transformers.txt` are needed for the LoRA backend; the API currently treats them as optional.
5. Whether `/predict` should also return a summary for every request, and whether a `buggy` result should be summarized (the model describes the function as written, not the bug).
6. Caveat text for the UI: summaries are generated and unverified.

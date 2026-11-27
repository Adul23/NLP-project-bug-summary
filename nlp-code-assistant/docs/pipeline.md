# Pipeline and input representation

How a Python function becomes a prediction, for the three models. Code is parsed and
tokenized, never executed. Every setting below is taken from the code named in brackets.

## End-to-end flow

```mermaid
flowchart TD
    A["Raw text: CodeSearchNet Python functions<br/>(HF, pinned revision, streamed, seed 42)"]
    B["Cleaning (src/data)<br/>LF line ends, ast.parse, one top-level function,<br/>docstring removed by AST coordinates, exact duplicates dropped"]
    C["Labels and split<br/>one AST operator mutated: CLEAN 0 / BUGGY 1<br/>stratified_v1: 70/15/15 by source_id"]
    A --> B --> C

    C --> T1
    C --> B1
    C --> S1

    subgraph TFIDF["TF-IDF + Logistic Regression (baseline)"]
        T1["Vectorization<br/>character n-grams (2,4), 30,000 features,<br/>no lowercasing, raw TF"]
        T2["Model input: sparse TF-IDF vector"]
        T3["Forward pass: w·x + b, sigmoid"]
        T4["Loss: log-loss with L2 (sklearn default C=1.0),<br/>class_weight=balanced; fitted on train only"]
        T5["Evaluation: BUGGY F1 on validation picks the n-gram range;<br/>test scored once"]
        T6["Inference output: label + bug probability"]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6
    end

    subgraph CODEBERT["CodeBERT + LoRA (classifier)"]
        B1["Tokenization: RoBERTa byte-level BPE,<br/>vocab 50,265, s ... /s, max 512, truncation"]
        B2["Model input: input_ids + attention_mask,<br/>dynamic padding per batch"]
        B3["Forward pass: 12-layer encoder with LoRA (r=8) on query/value,<br/>classification head on the s token"]
        B4["Prediction / loss: 2 logits, softmax, cross-entropy"]
        B5["Training: HF Trainer, AdamW, lr 2e-4, linear schedule,<br/>batch 16, fp16, up to 5 epochs, early stopping 2"]
        B6["Evaluation: best validation BUGGY F1 checkpoint;<br/>test scored once"]
        B7["Inference output: label + bug probability,<br/>localized fix (operator flip ranking)"]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7
    end

    subgraph CODET5["CodeT5-small (summarizer, part 2)"]
        S1["Tokenization: byte-level BPE, vocab 32,100;<br/>code max 384 tokens, summary max 48"]
        S2["Model input: code ids + mask (encoder),<br/>summary ids as labels, padding = -100"]
        S3["Forward pass: 6+6-layer encoder-decoder,<br/>teacher forcing"]
        S4["Loss: token cross-entropy over the summary"]
        S5["Training: own loop, AdamW, warmup 5% + linear,<br/>batch 8, grad clip 1.0, patience 2 on val loss"]
        S6["Evaluation: BLEU-4, ROUGE-L on validation;<br/>test scored once"]
        S7["Inference output: one-sentence summary<br/>(beam search, 4 beams)"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    end
```

## Input representation

| Model | Unit | Vocabulary | Length | What the model receives |
|---|---|---|---|---|
| TF-IDF + LR | character n-grams, 2–4 characters (chosen on validation from (2,4), (3,5), (3,6)) | 30,000 most frequent n-grams of train | whole function | a sparse vector of TF-IDF weights (`lowercase=False`, `sublinear_tf=False`), so operators such as `>=`, `!=` and ` and ` are features |
| CodeBERT | RoBERTa byte-level BPE subwords | 50,265 (`microsoft/codebert-base` config) | 512 tokens including `<s>` and `</s>`; longer functions are cut at the end, special tokens kept | `input_ids` and `attention_mask` (1 = real token, 0 = padding); batches are padded dynamically to their longest row (`DataCollatorWithPadding`, to a multiple of 8 under fp16) and grouped by length |
| CodeT5-small | byte-level BPE subwords | 32,100 (checkpoint config) | code up to 384 tokens, summary up to 48 tokens, both truncated | encoder: code ids and mask, raw code with no prompt; decoder: summary ids as labels, padding replaced by -100 so it is ignored by the loss |

Sources: `src/bug_detection/train_tfidf.py`, `train_comparison_tfidf.py`,
`models/tfidf_v2/metrics.json`; `src/bug_detection/train_lora.py`, `lora_common.py`
(`tokenize_once`), `models/codebert_lora/training_config.json`;
`src/summarization/train_codet5.py` (`make_batches`).

`scripts/demo_tokenize.py` prints these representations for one function.

## Outputs and loss functions

| Model | Output layer | Prediction | Training loss |
|---|---|---|---|
| TF-IDF + LR | one linear unit over the TF-IDF vector | sigmoid → bug probability; BUGGY if ≥ 0.5 | binary log-loss with L2 regularization (scikit-learn default `C=1.0`), `class_weight="balanced"`, `max_iter=2000` |
| CodeBERT + LoRA | `RobertaClassificationHead` on the `<s>` vector: dropout → dense 768→768 → tanh → dropout → `out_proj` 768→2 | softmax over 2 logits → bug probability; BUGGY if ≥ 0.5 (no tuned threshold is saved) | cross-entropy over the two classes (unweighted by default; `--class-weights` is optional and off) |
| CodeT5-small | encoder-decoder with a language-model head over the 32,100-token vocabulary | beam search: 4 beams, up to 48 new tokens | token-level cross-entropy over the summary with teacher forcing; padded label positions (-100) are ignored |

Generation settings for CodeT5: the test table in `docs/summarization-results.md` used 4 beams
and the default length penalty (1.0); so does the API on `main`
(`src/inference/models.py`). `src/summarization/summarize.py` uses 4 beams, length penalty
2.0, `max_new_tokens=48` and `early_stopping=True`; the API switches to these settings only
if the `feature/unify-summarizer` branch is merged.

LoRA in CodeBERT: the base weights are frozen; trainable are low-rank matrices on the
attention `query` and `value` layers (r=8, alpha 16, dropout 0.1) and the whole
classification head (`modules_to_save=["classifier"]`), 887,042 of 125,534,212 parameters
(0.71%).

## Inference path in the app

`POST /analyze` (`src/api/app.py`): the code is tokenized or vectorized as above, the model
returns the bug probability, and for a probability of at least 0.40 without reference code
`localize()` (`src/bug_detection/localize.py`) flips each operator back to every operator it
could have been mutated from, scores all candidates in one batch, and reports the candidate
that lowers the bug probability the most as the suspected line and fix, marked not verified.
With reference code, the exact AST difference is reported instead and marked verified.

# Python Bug Detection and Synthetic Bug Explanation

This project reuses the CodeSearchNet preprocessing and operator-mutation pipeline
and adds two classification paths, evidence-based explanations, optional LoRA
summarization, and a small FastAPI backend. Source code is never executed.

**Current status (2026-10-11):** TF-IDF trained on the complete local dataset;
verified synthetic explanation data prepared; API tested with real TF-IDF artifacts.
CodeBERT and CodeT5 training pipelines are implemented and tested with tiny random
local transformer fixtures. **The pretrained CodeBERT classifier and CodeT5
summarizer have not been fine-tuned here.** They have no measured quality or
latency results. This environment is CPU-only; full pretrained training is deferred
because of the runtime/resource cost, not because the fixture scores represent it.

## Setup and CPU baseline

Python 3.10+ (tested with 3.12.3). From the repository root:

```bash
cd nlp-code-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Existing dataset: data/processed/bug_detection_dataset.csv
python -m src.bug_detection.experiment_data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.bug_detection.train_comparison_tfidf
python -m src.summarization.prepare
python -m src.evaluation.compare_models
```

The split/preparation/training commands require an empty destination and refuse to
overwrite completed runs. If their artifacts already exist, use them directly or
choose new `--output` directories. Pass the same `--split-dir` to both classifiers
and explanation preparation. Keep `--data-dir` aligned for summarization.
The original dataset and historical `train.csv`, `validation.csv`, `test.csv`,
models and reports remain unchanged.

New shared splits live in `data/processed/stratified_v1/`. They stratify source groups
by their label composition (clean-only versus clean+buggy) while keeping each
`source_id` in exactly one partition. This preserves class balance without leaking
originals across splits. `manifest.json` records dataset and split SHA-256 hashes,
class/mutation counts, seed, and zero-overlap checks. All new trainers verify hashes.
This does not guarantee repository-level separation or remove semantic clones.

Character TF-IDF preserves punctuation, operators and local syntax. The new run uses
`analyzer="char"`, `lowercase=False`, `max_features=30000`, `sublinear_tf=False`.
It tries `(3,5)`, `(2,4)`, `(3,6)` and retains the model with the best validation
BUGGY F1. Logistic Regression uses 2000 maximum iterations, balanced class weights,
and seed 42. Test is evaluated once after selection. Reloaded predictions and
probabilities are checked on validation samples.

## Architecture

```text
CodeSearchNet Python functions
              |
              v
    Preprocessing and AST validation
              |
              v
    Synthetic operator mutation
              |
              v
       Labeled dataset
              |
              v
   Stratified source-group train/val/test
              |
       +------+------+
       |             |
       v             v
   TF-IDF        CodeBERT
      +             +
 Logistic Reg.     LoRA
       |             |
       +------+------+
              |
              v
     Evaluation / detection
              |
              v
  Explanation: verified code pair
  or optional CodeT5 LoRA generation
              |
              v
           FastAPI
```

| Component | Entry point |
|---|---|
| Existing preprocessing/mutations | `src/data/{preprocess,mutations,build_dataset}.py` |
| Frozen shared split protocol | `src/bug_detection/experiment_data.py` |
| New TF-IDF experiment | `src/bug_detection/train_comparison_tfidf.py` |
| CodeBERT LoRA | `src/bug_detection/train_lora.py` |
| Mutation verification and synthetic supervision | `src/summarization/prepare.py` |
| CodeT5 LoRA | `src/summarization/train_lora.py` |
| Local inference and CLI | `src/inference/{models,cli}.py` |
| Recorded metrics and warm latency comparison | `src/evaluation/compare_models.py` |
| HTTP backend | `src/api/app.py` |

## Optional transformer installation and training

The baseline/API do not require torch, transformers, GPU access, or model downloads.
Install the optional dependencies only to train/use adapters:

```bash
# CPU PyTorch; choose the appropriate official PyTorch wheel for your CUDA system.
python -m pip install 'torch>=2.6,<2.7' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-transformers.txt
export HF_HOME="$PWD/.cache/huggingface"
python -m src.bug_detection.train_lora --device cpu --batch-size 1 --accumulation 8 --gradient-checkpointing
python -m src.summarization.train_lora --device cpu --batch-size 1 --accumulation 8 --gradient-checkpointing
```

For GPU training use a CUDA torch build and `--device cuda`. `--device auto` is the
default. BF16 is enabled only on supporting CUDA hardware; otherwise FP32 is used.
No FP16 is enabled on CPU. Start at 512 tokens, use dynamic padding and gradient
accumulation, and lower `--max-length` only as a separately documented experiment.
CUDA out-of-memory errors include reduction options. Long functions are truncated;
the classifier records truncation fractions and the changed token can be lost.

LoRA freezes the base weights and learns low-rank updates. The CodeBERT path checks
for RoBERTa architecture, discovers attention `query`/`value` linear layers, uses
`TaskType.SEQ_CLS`, `r=8`, alpha 16, dropout 0.1, and explicitly trains/saves the
`classifier` head. Validation BUGGY F1 selects the checkpoint; patience is 2 epochs.
CodeT5 uses T5 attention `q`/`v` layers and `TaskType.SEQ_2_SEQ_LM`; validation loss
selects its checkpoint. Both save adapter/tokenizer, training metadata, base model
revision, and metrics. Seeds are 42; different devices may still differ numerically.
Use `--revision <commit>` to pin a model download; the resolved revision is recorded.

Classification predicts a label, while summarization generates tokens. They are
separate models and objectives, not interchangeable uses of a classification head.
CodeT5-small is chosen as a smaller code-oriented encoder-decoder with an
[Apache-2.0 model card](https://huggingface.co/Salesforce/codet5-small).
CodeBERT's [configuration](https://huggingface.co/microsoft/codebert-base/raw/main/config.json)
identifies RoBERTa attention; the official [CodeBERT repository](https://github.com/microsoft/CodeBERT/blob/master/LICENSE)
uses MIT. Architecture-specific adapter saving follows the
[PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora).
Both models were pretrained on CodeSearchNet, so prior exposure is a further
limitation when interpreting this downstream evaluation.

## Explanation supervision and interpretation

`mutation_type` alone is not a human-written explanation. Preparation pairs each
buggy function with its original in the **same split**, verifies exactly one
supported AST operator change and one changed source token, and rejects conflicts.
It reuses the existing mutation maps and AST checker. Each JSONL record contains:

```json
{
  "source_id": "000001",
  "sample_id": 2,
  "buggy_code": "def f(x): return x - 1",
  "original_code": "def f(x): return x + 1",
  "explanation": "Synthetic mutation: arithmetic operator changed from `+` to `-` on line 1. This may alter behavior; an actual failure is not established.",
  "supervision": "verified_synthetic_pair",
  "change": {"mutation_type": "arithmetic_operator", "old_operator": "+", "new_operator": "-", "line": 1}
}
```

The saved records also include a null suggested fix and the verified change's
explanation/type. Files are hash-checked and pair verification runs again before
training. There are **3320 / 712 / 711** training/validation/test summaries.
These template targets are synthetic, not human annotations. The generator sees
only buggy code; it cannot always identify the original operator without evidence.
It may hallucinate, and token loss on templates does not measure real bug reasoning.

Training exports 20 fixed test examples with generated/reference text and blank
human-review fields. Rate correctness and specificity 0 (wrong/absent), 1 (partial),
2 (supported and precise), and mark unsupported claims true/false. Inspect against
both original and mutated functions, including operator and line. Report counts,
rater agreement and examples; do not fill ratings automatically. No human quality
scores or trained pretrained summarizer are claimed in this checkout.

## Inference and API

```bash
printf 'def f(a,b): return a != b\n' | python -m src.inference.cli --model tfidf
python -m src.inference.cli --model codebert --file function.py
python -m src.inference.cli --model summarizer --file function.py
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Use `--original-file original.py` with the CLI for verified pair evidence.
The API caches models after their first successful load. Defaults are
`models/tfidf_v2`, `models/codebert_lora`, and `models/codet5_lora`.
`BUG_MODEL_ROOT` can change the parent directory. Inference never downloads models;
transformer base weights must already be in the local cache or at the saved path.

- `GET /health`: service status and each model's unavailable/artifacts-present/loaded/load-failed state.
- `POST /predict`: `{"code":"def f(a,b): return a != b", "model":"tfidf"}`.
  Returns `label`, `bug_probability`, predicted-label `confidence`, and `model`.
- `POST /analyze`: same fields, with optional `original_code` for a verified change
  or `"explanation":"lora"` for explicitly unverified generated text.
  Returns classification plus `is_buggy`, `bug_type`, `explanation`,
  `suggested_fix`, `explanation_source`, and `verified`.

Without an original or trained summarizer, explanation/type/fix are null. A verified
pair can provide evidence even if the classifier predicts CLEAN; the classifier
label is not silently replaced. Verification proves a supported code difference,
not semantic incorrectness or the trustworthiness of a user-supplied original.
Generated explanations always have `verified=false`, unknown `bug_type`, and no
suggested fix. Classifier confidence is not explanation confidence or calibration.

Requests have a 20,000-character code limit (also for originals) and 128,000-byte
body limit, including streamed bodies. Blank/non-string inputs and unknown models
return 422; missing model artifacts/dependencies return 503; oversized bodies return
413. No `eval`, `exec`, subprocess execution of source, or precise repair guessing.

## Current measured experiment

This is a **new protocol** (stratified groups and ordinary TF). It is not directly
comparable to the historical sublinear-TF run below or the student notebook.

| Model | BUGGY precision | BUGGY recall | BUGGY F1 | Accuracy | Inference latency |
|---|---:|---:|---:|---:|---|
| TF-IDF + LR | 0.5167 | 0.6751 | 0.5854 | 0.6924 | 1.59 ms median |
| CodeBERT + LoRA | Not trained | Not trained | Not trained | Not trained | Not measured |

Latency: 30 validation functions, warm CPU, batch 1, includes vectorization and
prediction, excludes loading. Adapter-only size is not total model size; when
CodeBERT is trained, comparison also records loaded base-plus-adapter parameter bytes.
The TF-IDF artifacts total 1,246,638 bytes. No transformer speed/size advantage is claimed.

New split row counts: **10,320 / 2,212 / 2,211**, with **3,320 / 712 / 711** buggy rows.
All pairwise source-ID overlaps are zero. Validation selected **(2,4)** with F1
**0.5761**; (3,5) scored 0.5664 and (3,6) scored 0.5650. Final test confusion matrix:

```text
[[1051, 449],
 [ 231, 480]]
```

Test mutation recall: comparison 214/303 = 0.7063; arithmetic 194/277 = 0.7004;
boolean 72/131 = 0.5496. Comparison was easiest and boolean hardest in this run.
See `results/model_comparison.json` and `results/tfidf_v2_metrics.json` for measured
values, provenance and per-class reports. Models and generated data remain ignored.

## Tests and limitations

Verified on 2026-10-11: **63 tests and 31 subtests passed**. `pip check` found no
broken requirements. Three dependency/fixture warnings remain (Starlette's deprecated
AnyIO portal alias and PEFT config warnings for two in-memory random model fixtures).
A real Uvicorn localhost smoke test returned HTTP 200 for health, prediction and
verified-pair analysis; its output is saved in `results/api_smoke.json`.
The complete test log is `results/extended_pytest.log`.

```bash
pytest -v
python -m pip check
```

Transformer tests skip if optional dependencies are absent; with them installed,
`tests/test_lora_local.py` performs real training, gradient checks, adapter/head
save/reload and generation using tiny random local RoBERTa/T5 fixtures. Those
fixture scores are not model comparison results. Training never starts at import.

The CPU baseline is functional. Full pretrained CodeBERT/CodeT5 fine-tuning and
human explanation review remain outstanding. Synthetic operator replacements may
not introduce a behavioral error; original code may already contain bugs. Dataset
sampling, repository overlap, semantic clones and CodeSearchNet pretraining exposure
limit generalization. See the original preparation documentation below.

---

## Historical data preparation and baseline (2026-09-28)

Конвейер собирает **10 000 уникальных Python-функций CodeSearchNet** и создаёт
не более одной синтетической версии каждой функции. `label=0` — исходная
функция, `label=1` — функция с одной заменой оператора. Метка `clean` означает
исходный код: CodeSearchNet не гарантирует отсутствие реальных ошибок.

Получено **10 000 clean + 4 743 buggy = 14 743 строки**. Реализован baseline
**Character TF-IDF + Logistic Regression**: разбиение без пересечения `source_id`,
выбор конфигурации на validation, оценка на test и предсказание для нового кода.
Подробности и проблемы разбора — в [отчёте запуска](nlp-code-assistant/docs/data-preparation-results.md).

## Запуск

Python **3.10+**. Из каталога `nlp-code-assistant`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/data/download_codesearchnet.py
python src/data/preprocess.py
python src/data/build_dataset.py
```

Первый шаг требует интернет. Остальные работают с локальными файлами.
Пути по умолчанию вычисляются относительно проекта. Параметры доступны через
`--help`. Для небольшого прогона укажите одинаковый `--target 100` в первых
двух командах. `download_codesearchnet.py --input local.jsonl` принимает
локальный JSONL с полем `code`; порядок такого файла сохраняется.

## Файлы

| Файл | Назначение |
|---|---|
| `src/data/download_codesearchnet.py` | Потоковое чтение до 10 000 принятых функций |
| `src/data/preprocess.py` | Очистка, AST-проверка, дедупликация, `source_id` |
| `src/data/mutations.py` | Одна мутация AST с сохранением оформления |
| `src/data/build_dataset.py` | Сборка, проверки качества, статистика и примеры |
| `src/bug_detection/split_dataset.py` | Разбиение 70/15/15 по `source_id` |
| `src/bug_detection/train_tfidf.py` | Обучение, выбор конфигурации, оценка и сохранение моделей |
| `src/bug_detection/predict_tfidf.py` | `predict_bug(code)` для нового Python-кода |
| `data/raw/codesearchnet/python_functions.jsonl` | Обработанные исходные записи и происхождение |
| `data/raw/codesearchnet/python_functions.manifest.json` | Ревизия, URL, seed, версии библиотек и счётчики |
| `data/processed/clean_functions.csv` | `source_id,code,label,mutation_type` |
| `data/processed/bug_detection_dataset.csv` | `sample_id,source_id,code,label,mutation_type` |
| `data/processed/*.stats.json` | Счётчики очистки и распределения классов/мутаций |
| `data/processed/bug_detection_dataset.examples.txt` | По пять полных пар на категорию |

Данные, модели, кэш и виртуальное окружение исключены через `.gitignore`.
TF-IDF baseline реализован; остальные заготовки моделей и приложения не реализованы.

## Источник и воспроизводимость

Используется [официальный CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net),
Python, существующий upstream-раздел `train`. Это выбор источника, а не
разбиение нового датасета. Parquet закреплён на ревизии
`7e3332b8032ff895377dba3e57f0152fcb32ff7d`.

Загрузчик использует [потоковый режим Hugging Face Datasets](https://huggingface.co/docs/datasets/en/stream)
и читает только столбцы кода и происхождения. Seed **42**, буфер перемешивания
**1000**; это воспроизводимая выборка из начала перемешанного потока,
**не равномерная выборка всего корпуса**. Чтение прекращается после 10 000
принятых функций; библиотека может заранее прочитать буфер и группы строк
Parquet. Ограничение просмотра — 100 000 записей (`--max-scan`).
Весь архив CodeSearchNet загружать не требуется.

Parquet читается последовательными пакетами по 256 записей через PyArrow,
перемешивание выполняет `datasets.IterableDataset`. Это позволяет остановить
поток без фонового многопоточного сканера, который в проверенном окружении
падал при завершении Python после ранней остановки.

Для повторения используйте сохранённый JSONL и версии из manifest: порядок
потока зависит от `datasets`, а AST — от Python. В raw JSONL принятые записи
имеют `source_id`, совпадающий с clean CSV; отброшенные — `source_id=null`.
Сохранены репозиторий, путь, имя функции и URL исходника. Отдельные поля
документации не загружаются.

## Очистка

- Нормализуются CRLF/CR в LF, удаляются внешние пробелы и пустые примеры.
- `ast.parse()` отбрасывает синтаксически некорректные примеры.
- Допускается ровно одна верхнеуровневая `FunctionDef` или `AsyncFunctionDef`,
  включая декораторы; случайные фрагменты и целые классы отбрасываются.
- Начальные строки документации удаляются по координатам AST без
  форматирования остального тела. Если тело состояло только из них,
  остаётся `pass`.
- Точное совпадение очищенного кода считается дубликатом.
- ID назначается после очистки, начиная с `000001`.

`Valid Python` включает синтаксически корректные дубликаты и отсеянные
нефункциональные фрагменты. `Invalid Python` — ошибки разбора/обработки;
первые пять сообщений сохранены в статистике. Пустые примеры учитываются
отдельно. При недостатке данных команда завершается ошибкой и не публикует
неполный CSV как успешный результат.

## Мутации

| Категория | Замены |
|---|---|
| `comparison_operator` | `== → !=`, `!= → ==`, `> → <`, `< → >`, `>= → >`, `<= → <` |
| `arithmetic_operator` | `+ → -`, `- → +`, `* → /`, `/ → *` |
| `boolean_operator` | `and → or`, `or → and` |

Случайно выбираются применимая категория и узел. Меняется **один оператор
AST**, результат преобразуется через `ast.unparse()` и проверяется повторным
разбором. Для итогового CSV заменяется только токен этого оператора в
исходном тексте. Полученное дерево обязано совпасть с мутированным AST:
глобальной замены строк нет. Сохраняются строки, комментарии и оформление
в обоих классах.

Операторы в декораторах, аннотациях и значениях аргументов по умолчанию
не изменяются. В цепочках сравнений меняется одно сравнение. Для булевых
операций используются узлы с двумя операндами: плоское `a and b and c`
требовало бы замены двух токенов при прямой замене `BoolOp.op` и пропускается.
Вложенные бинарные операции допускаются.

Невалидные или меняющие структуру приоритета кандидаты пропускаются;
проверяются оставшиеся. Проверяется совпадение со **всеми** чистыми примерами
и принятыми мутациями. Если подходящего уникального варианта нет, сохраняется
только чистая функция. `Invalid mutations skipped` считает попытки, а не
функции; отдельно выводятся коллизии и число функций без мутации.

Замена оператора создаёт синтетическую метку, но не доказывает поведенческую
ошибку на реальных входах. Код датасета никогда не исполняется.

## Проверки и чтение CSV

Перед записью проверяются разбор всего кода, глобальная уникальность,
допустимые метки, категории, ID, наличие чистого оригинала для каждой мутации
и ровно одна разрешённая замена в AST каждой пары.

```python
import pandas as pd

df = pd.read_csv(
    "data/processed/bug_detection_dataset.csv",
    dtype={"source_id": str},  # сохранить ведущие нули
    keep_default_na=False,
)
```

```bash
pytest -v
```

## TF-IDF + Logistic Regression Baseline

Character-level TF-IDF represents Python source code using character n-grams.
It captures local syntax and operators such as `==`, `!=`, `>=`, `+`, and `-`.
Logistic Regression performs binary classification: **CLEAN (0)** vs **BUGGY (1)**.

The dataset is split **70% / 15% / 15% by unique `source_id`, seed 42** to avoid
leakage between original and mutated versions of the same function. These
percentages apply to source IDs; row counts vary because not every function
has a mutation. All three partitions must contain both classes.

TF-IDF is fitted only on training code, with `analyzer="char"`,
`max_features=30000`, `lowercase=False`, and `sublinear_tf=True`.
The initial baseline uses `(3, 5)`; the experiment compares `(2, 4)`, `(3, 5)`,
and `(3, 6)`. Logistic Regression uses `max_iter=2000`, `class_weight="balanced"`,
and `random_state=42`. The highest **validation Buggy F1** selects the winner;
ties prefer the first evaluated configuration: `(3, 5)`, `(2, 4)`, `(3, 6)`.
The selected trained model is retained, with no refit on validation, and
evaluated **once on test**. The test set does not select features, models, or thresholds.

From `nlp-code-assistant/`, with the virtual environment activated:

```bash
python -m pip install -r requirements.txt
# Optional: create/inspect the splits without training.
python src/bug_detection/split_dataset.py
# Recreates the same deterministic splits, trains, evaluates, and saves artifacts.
python src/bug_detection/train_tfidf.py
pytest -v
```

All default paths are resolved relative to the project. `--help` lists custom
input and output paths. Training prints dataset/split statistics, the leakage
check, configuration comparison, both class metrics, confusion matrices,
mutation recalls, and up to five examples each of TP/TN/FP/FN with full code.

```python
from src.bug_detection.predict_tfidf import predict_bug

result = predict_bug("""
def calculate_total(price, tax):
    return price - tax
""")
print(result)  # {"label": "clean" or "buggy", "bug_probability": float}
```

Artifacts under `nlp-code-assistant/`:

| Path | Contents |
|---|---|
| `data/processed/{train,validation,test}.csv` | Grouped splits, preserving zero-padded IDs |
| `models/tfidf_vectorizer.joblib` | Selected fitted TF-IDF vectorizer |
| `models/logistic_regression.joblib` | Selected fitted classifier |
| `results/tfidf_logistic_regression_metrics.json` | Validation comparison, final test metrics, per-class reports, mutation recall, examples, dataset hash, and library versions |
| `results/tfidf_error_examples.txt` | Up to five complete examples per prediction outcome |

For mutation analysis, recall is computed only over BUGGY rows. An absent
category has zero samples and `null` recall. Metrics JSON top-level accuracy,
Buggy precision/recall/F1, and confusion matrix refer to the final **test** set.
Probabilities are model estimates, not calibrated guarantees of semantic correctness.
The synthetic labels and CodeSearchNet sampling limitations described above still apply.

### Результаты запуска

Разбиение по `source_id` с seed 42:

| Выборка | Функции (`source_id`) | Строки | CLEAN | BUGGY |
|---|---:|---:|---:|---:|
| train | 7000 | 10315 | 7000 | 3315 |
| validation | 1500 | 2206 | 1500 | 706 |
| test | 1500 | 2222 | 1500 | 722 |

Пересечение `source_id` между каждой парой выборок: **0**.

Сравнение конфигураций на validation; precision, recall и F1 относятся к BUGGY:

| N-grams | Precision | Recall | F1 |
|---|---:|---:|---:|
| `(2, 4)` | 0.5541 | 0.6813 | 0.6112 |
| `(3, 5)` | 0.5418 | 0.6884 | 0.6064 |
| `(3, 6)` | 0.5297 | 0.6813 | 0.5960 |

Выбрана конфигурация **`(2, 4)`**. Итоговые метрики выбранной модели:

| Выборка | Accuracy | BUGGY precision | BUGGY recall | BUGGY F1 |
|---|---:|---:|---:|---:|
| validation | 0.7226 | 0.5541 | 0.6813 | 0.6112 |
| test | 0.7142 | 0.5462 | 0.7119 | 0.6182 |

Матрица ошибок на test, порядок `[[TN, FP], [FN, TP]]`:

```text
[[1073, 427],
 [ 208, 514]]
```

Обнаружение мутаций среди BUGGY-примеров test:

| Тип мутации | Примеров | Обнаружено | Recall |
|---|---:|---:|---:|
| `arithmetic_operator` | 269 | 188 | 0.6989 |
| `boolean_operator` | 142 | 77 | 0.5423 |
| `comparison_operator` | 311 | 249 | 0.8006 |

Проверка завершённого baseline: **45 тестов и 31 subtest прошли**.
Модель обнаружила 514 из 722 синтетических ошибок, но ошибочно пометила
427 чистых примеров. Эти результаты характеризуют синтетический датасет,
а не гарантируют обнаружение ошибок в произвольном Python-коде.

Run results: [TF-IDF baseline report](nlp-code-assistant/docs/tfidf-baseline-results.md).

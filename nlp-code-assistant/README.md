# NLP Code Assistant — подготовка данных

Конвейер собирает **10 000 уникальных Python-функций CodeSearchNet** и создаёт
не более одной синтетической версии каждой функции. `label=0` — исходная
функция, `label=1` — функция с одной заменой оператора. Метка `clean` означает
исходный код: CodeSearchNet не гарантирует отсутствие реальных ошибок.

Previous run: **10,000 clean + 4,743 buggy = 14,743 rows**.
Current local run: **10,000 clean + 5,304 buggy = 15,304 rows**; see the baseline section below.
Подробности и проблемы разбора — в [отчёте запуска](docs/data-preparation-results.md).

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
| `data/raw/codesearchnet/python_functions.jsonl` | Обработанные исходные записи и происхождение |
| `data/raw/codesearchnet/python_functions.manifest.json` | Ревизия, URL, seed, версии библиотек и счётчики |
| `data/processed/clean_functions.csv` | `source_id,code,label,mutation_type` |
| `data/processed/bug_detection_dataset.csv` | `sample_id,source_id,code,label,mutation_type` |
| `data/processed/*.stats.json` | Счётчики очистки и распределения классов/мутаций |
| `data/processed/bug_detection_dataset.examples.txt` | По пять полных пар на категорию |

Данные, модели, кэш и виртуальное окружение исключены через `.gitignore`.
The TF-IDF + Logistic Regression baseline is implemented; other models and the app remain placeholders.

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
python -m unittest discover -s tests -v
```

Source-ID splitting, training, and evaluation are implemented in the baseline below.
## TF-IDF + Logistic Regression Baseline

Character-level TF-IDF represents Python source code using character n-grams.
These features capture local syntax and operators such as `==`, `!=`, `>=`, `+`,
and `-`. Logistic Regression predicts CLEAN (0) versus BUGGY (1).

The baseline validates all required fields, removes exact duplicate rows, and
splits unique `source_id` values into 70% train, 15% validation, and 15% test
with seed 42. Original and mutated versions stay together to prevent source-ID
leakage. Row proportions can differ slightly because groups have different sizes.
All three pairwise source-ID intersections are checked explicitly.

Only training code is used to fit TF-IDF and Logistic Regression. The experiment
compares `(2, 4)`, `(3, 5)`, and `(3, 6)` character n-grams with `max_features=30000`,
`lowercase=False`, and `sublinear_tf=True`. Logistic Regression uses
`max_iter=2000`, `class_weight="balanced"`, and `random_state=42`.
The largest validation Buggy F1 wins; ties keep the first configuration in that
order. The selected train-only model is retained without refitting and evaluated
once on test data. Running the training command again repeats that experiment.

From this directory, in PowerShell with standard CPython installed:

```powershell
py -3.12 -m venv .venv-baseline
.\.venv-baseline\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-baseline\Scripts\python.exe src/bug_detection/train_tfidf.py
.\.venv-baseline\Scripts\python.exe -m pytest -v
```

On other platforms, activate your Python environment and use `python` for the
same commands. The separate split command is
`python src/bug_detection/split_dataset.py`. Both scripts support `--help` and
resolve default paths relative to this project, independent of the working directory.

This checkout originally lacked the processed CSV. The current dataset was
regenerated from the first 10,000 valid unique functions in the local training
shard `data/raw/python/final/jsonl/train/python_train_0.jsonl`. This is a new local
subset, not a reproduction of the older Hugging Face subset described above.
To reproduce its preparation using the existing scripts:

```powershell
.\.venv-baseline\Scripts\python.exe src/data/download_codesearchnet.py --input data/raw/python/final/jsonl/train/python_train_0.jsonl
.\.venv-baseline\Scripts\python.exe src/data/preprocess.py
.\.venv-baseline\Scripts\python.exe src/data/build_dataset.py
```

The preparation manifest is saved beside the raw subset. The baseline metrics
include the processed CSV SHA-256 and package versions for reproducibility.
No collected Python source is executed. Existing AST-based mutation generation
is used solely to prepare labels; the model uses only character TF-IDF features.

Artifacts:

- `data/processed/train.csv`, `validation.csv`, `test.csv`: grouped splits.
- `models/tfidf_vectorizer.joblib`, `models/logistic_regression.joblib`: selected model.
- `results/tfidf_logistic_regression_metrics.json`: dataset/split statistics,
  validation comparison, both class reports, confusion matrices, mutation recall,
  and up to five examples in each prediction category.
- `results/tfidf_error_examples.txt`: full readable TP/TN/FP/FN code examples.
- `results/tfidf_test_predictions.csv`: test predictions and bug probabilities.
- `docs/tfidf-baseline-results.md`: results from the completed local run.

```python
from src.bug_detection.predict_tfidf import predict_bug

result = predict_bug("def calculate_total(price, tax):\n    return price - tax")
print(result)  # {"label": "clean" or "buggy", "bug_probability": a float in [0, 1]}
```

Mutation-type recall is `null` when a category has no test samples. Undefined
classification precision/recall is reported as zero. Probabilities are model
estimates, not proof that a function is correct or buggy. Synthetic operator
mutations and source-ID separation do not establish performance on real bugs or
on entirely unseen repositories.

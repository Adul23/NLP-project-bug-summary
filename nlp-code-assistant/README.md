# NLP Code Assistant — подготовка данных и TF-IDF baseline

Конвейер собирает **10 000 уникальных Python-функций CodeSearchNet** и создаёт
не более одной синтетической версии каждой функции. `label=0` — исходная
функция, `label=1` — функция с одной заменой оператора. Метка `clean` означает
исходный код: CodeSearchNet не гарантирует отсутствие реальных ошибок.

Получено **10 000 clean + 4 743 buggy = 14 743 строки**. Реализован baseline
**Character TF-IDF + Logistic Regression**: разбиение без пересечения `source_id`,
выбор конфигурации на validation, оценка на test и предсказание для нового кода.
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

Run results: [TF-IDF baseline report](docs/tfidf-baseline-results.md).

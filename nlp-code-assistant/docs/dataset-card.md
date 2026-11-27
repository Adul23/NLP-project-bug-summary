# Dataset card

Two datasets built from the same CodeSearchNet Python functions and the same split:
bug detection (part 1) and code summarization (part 2). Code is parsed, never executed.

## Source

- [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) on
  Hugging Face, Python, upstream `train` partition, Parquet files pinned to revision
  `7e3332b8032ff895377dba3e57f0152fcb32ff7d` (`src/data/download_codesearchnet.py`).
- Streamed with Hugging Face Datasets, seed 42, shuffle buffer 1000; reading stops after
  10,000 accepted functions (at most 100,000 records scanned). This is a reproducible sample
  from the start of the shuffled stream, not a uniform sample of the corpus.
- Kept per function: code, repository, path, function name, source URL
  (`data/raw/codesearchnet/python_functions.manifest.json` records revision, URL, seed,
  library versions and counters).

## License

The Hugging Face page lists the CodeSearchNet license as "other": every function comes from a
public GitHub repository under that repository's own license, and the dataset carries no
per-example license information. We use the functions for course research only. The processed
data is kept in this course repository (`data/processed.zip`) so that the experiments can be
reproduced; it is not published as a separate dataset.

## Cleaning (`src/data`, README "Cleaning")

1. CR/LF line endings normalized to LF; surrounding whitespace and empty samples removed.
2. `ast.parse()` drops syntactically invalid samples.
3. Exactly one top-level `def` or `async def` (decorators allowed); fragments and whole
   classes are dropped.
4. Leading docstrings removed by AST coordinates, without reformatting the rest of the body
   (`pass` remains if the body was only a docstring).
5. Exact duplicates of the cleaned code removed.
6. IDs (`source_id`) assigned after cleaning, from `000001`.

## Bug detection dataset

### Labels

- `label = 0` (CLEAN): the original function. It is not guaranteed bug-free: CodeSearchNet
  does not check behaviour.
- `label = 1` (BUGGY): the same function with exactly one AST operator replaced. The label is
  synthetic: it marks a change of the operator, not a proven behavioural error.

### Mutations

| Type | Replacements |
|---|---|
| `comparison_operator` | `== → !=`, `!= → ==`, `> → <`, `< → >`, `>= → >`, `<= → <` |
| `arithmetic_operator` | `+ → -`, `- → +`, `* → /`, `/ → *` |
| `boolean_operator` | `and → or`, `or → and` |

One applicable node is chosen at random and only that operator's token is replaced in the
source text; the result must parse and match the mutated AST. Strings, comments and formatting
are unchanged. Operators in decorators, annotations and default values are not mutated;
mutations that change operator precedence are skipped. Every mutation is checked against all
clean samples and accepted mutations, so no buggy row duplicates another row. At most one
buggy version per function.

### Size and split

The `stratified_v1` split used by all experiments covers 14,743 rows: 10,000 clean + 4,743
buggy (`dataset_sha256` `0734fa6e…` in `data/processed/stratified_v1/manifest.json`). The README
also mentions a later local rebuild with 15,304 rows; it was not used for training or
evaluation.

| Split | Rows | Functions (`source_id`) | CLEAN | BUGGY | BUGGY share | comparison | arithmetic | boolean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10,320 | 7,000 | 7,000 | 3,320 | 32.2% | 1,354 | 1,284 | 682 |
| validation | 2,212 | 1,500 | 1,500 | 712 | 32.2% | 299 | 277 | 136 |
| test | 2,211 | 1,500 | 1,500 | 711 | 32.2% | 303 | 277 | 131 |

Source: `split_manifest` in `results/codebert_seeds/42/metrics.json` (identical in
`data/processed/stratified_v1/manifest.json`).

### Leakage protection

- 70/15/15 split by `source_id`: a function and its mutant are always in the same split.
  Pairwise `source_id` overlaps are 0.
- Groups are stratified by label composition (clean only vs clean + buggy), which keeps the
  BUGGY share equal across splits.
- `manifest.json` stores SHA-256 hashes of the dataset and each split; every trainer checks
  them before training.
- Model and threshold choices use validation only; test is scored once.

### Near duplicates

`scripts/near_duplicates.py` → `results/near_duplicates.json`: for every clean validation and
test function, the maximum cosine similarity (TF-IDF over code tokens, 1–3 grams) to any clean
train function.

| Split | Functions | ≥ 0.8 | ≥ 0.9 | ≥ 0.95 | Median |
|---|---:|---:|---:|---:|---:|
| validation | 1,500 | 112 | 17 | 0 | 0.33 |
| test | 1,500 | 137 | 14 | 0 | 0.34 |

No exact copies remain, but some near copies do: the closest pairs (0.945 and 0.940) are
sibling methods of one class in one repository, with the same body apart from one call
(`Path.is_char_device` / `Path.is_block_device` in `click-contrib/click-configfile`,
`Region.symmetric_difference` / `Region.without` in `PaulHancock/Aegean`; repository names
from `data/raw/codesearchnet/python_functions.jsonl`). The split separates functions, not
repositories.

## Summarization dataset (part 2)

- Input: the clean function code without its docstring; target: the cleaned first sentence of
  the original docstring (`src/summarization/build_summary_dataset.py`).
- Only clean (`label = 0`) functions. Docstring noise (parameter lists, links, doctests) is
  removed; too short, too long, empty and non-English summaries are dropped; rows whose summary
  appears verbatim in the code are removed to avoid leakage.
- Same `stratified_v1` split by `source_id`: train 6,524, validation 1,384, test 1,400
  (9,308 pairs). Median reference length: 8 words.

Source: `docs/summarization-results.md`.

## Known limitations

- Bugs are synthetic single-operator replacements. On 500 real operator fixes from GitHub
  (CTSSB-1M) the classifiers are near chance (`docs/real-bugs.md`).
- CLEAN originals may contain real bugs.
- The split does not separate repositories; near copies exist (see above).
- CodeBERT and CodeT5 were pretrained on CodeSearchNet and may have seen these functions.
- The sample is the start of a shuffled stream, not uniform over CodeSearchNet.
- Docstring summaries are noisy references: a reasonable summary can score low.

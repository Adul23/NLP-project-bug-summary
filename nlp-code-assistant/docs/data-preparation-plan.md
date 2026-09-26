# Data preparation implementation plan

The supplied request is the specification. Implement only preparation of
10,000 unique Python functions and at most one synthetic mutation per source.
Use Python 3.10+, seed 42, pandas, datasets and the standard-library AST.

1. Test preprocessing with empty inputs, CRLF, duplicate functions, invalid
   Python, async functions and fragments. Implement `src/data/preprocess.py`:
   normalize without reformatting, optionally remove standalone docstrings,
   require a top-level function, assign zero-padded IDs, write clean CSV.
2. Test operator changes against hand-written expectations, string safety,
   chained comparisons, reproducibility and functions without candidates.
   Implement `src/data/mutations.py`: choose a supported category and AST node,
   change one operator, unparse and validate. Preserve original formatting
   using the AST-local token span when it reconstructs the same mutated AST.
3. Test the stream cutoff and a local end-to-end CLI run. Implement
   `src/data/download_codesearchnet.py`: stream the pinned official Python
   training Parquet, stop when preprocessing accepts 10,000 functions, save
   only consumed rows to JSONL and record revision and processing statistics.
4. Test pair linkage, global code deduplication, CSV IDs and quality failures.
   Implement `src/data/build_dataset.py`: create the combined CSV, validate
   before writing, report class/category counts and five pairs per category.
5. Install dependencies in `.venv`, execute all three CLIs on CodeSearchNet,
   run the full test suite and independently audit the generated CSVs.
   Document commands, actual statistics, parsing limitations and provenance.

No model, split, retrieval, evaluation or UI implementation. Data and caches
are ignored by git. No source code from the dataset is executed.

Review focus: insufficient input must fail without publishing a short dataset;
source IDs must retain zeros on CSV read; mutation collisions must be skipped;
AST shared operator objects must never be modified in place; download failures
must leave no apparently complete raw output. Boolean chains must not lead to
multiple source-operator edits: only binary BoolOp nodes are eligible.

Completed: all four modules, documentation, 23 passing offline tests, real
CodeSearchNet download and both CSVs. Independent CSV/token audit passed;
all 10,000 clean source IDs verified against raw provenance. Reviewer Unicode
line-position finding fixed with a failing-then-passing test. Real-data
consecutive-docstring finding likewise fixed and tested. Native Arrow scanner
shutdown failure resolved using synchronous ParquetFile batches with
datasets.IterableDataset shuffle; 100-row and 10,000-row remote runs exited 0.
Final counts and examples are in data-preparation-results.md.

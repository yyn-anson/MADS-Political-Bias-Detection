# Data folders

- `articles/` is the user input folder. It is intentionally empty in Git.
- `sample_articles/` contains five small, source-linked fixtures for `--demo`.

Add JSON, JSONL, CSV, or TXT files to `articles/`, then run `python3 run.py` from
the repository root. See `docs/DATA_FORMAT.md` for the schema and validation rules.

The samples are original paraphrases based on real public-affairs releases. They are
not verbatim copies of news articles. Their labels are manual expectations selected
to exercise all three output classes, not research-grade annotations.

# Data folders

- `articles/` is the user input folder. It is intentionally empty in Git.
- `sample_articles/` contains five small, source-linked fixtures for `--demo`.
- `custom_dataset/` documents and converts the project's historical custom corpus.
- `article_outlet_labels.csv` maps 473,989 article IDs to AllSides 2025 outlet ratings.
- `allsides/AllSides_Rating.csv` is the corresponding outlet-rating reference.

Add JSON, JSONL, CSV, or TXT files to `articles/`, then run `python3 run.py` from
the repository root. See `docs/DATA_FORMAT.md` for the schema and validation rules.

The samples are original paraphrases based on real public-affairs releases. They are
not verbatim copies of news articles. Their labels are manual expectations selected
to exercise all three output classes, not research-grade annotations.

Custom-dataset labels follow a different rule: each article inherits its media
outlet's AllSides 2025 rating. See `custom_dataset/README.md` for the exact mapping,
conversion command, and limitation of this outlet-level ground-truth assumption.

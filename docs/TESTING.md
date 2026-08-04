# Testing and verification

## Deterministic suite

```bash
python3 -m unittest discover -s tests -v
```

The suite uses a fake model backend and covers score boundaries, probability entropy,
data validation, custom-dataset conversion, five-to-three AllSides label mapping,
unanimous early exit, 2-vs-1 debate, all-different divergent-pair selection, stage
routing, prediction-token log-probabilities, article and outlet metrics, valid SVG
figures, and all report artifacts. It is deterministic and has no network, model, or
third-party dependency.

## Local Ollama contract check

```bash
python3 run.py --check
```

This confirms that the configured OpenAI-compatible endpoint is reachable and every
configured model is installed.

## End-to-end smoke test

```bash
python3 run.py --demo --embedding hashing
```

This makes real inference calls to the installed local model, runs the complete
three-agent router, and writes:

- `report.html` for human review;
- `results.json` with complete agent analyses, probabilities, debate transcripts,
  termination conditions, and configuration;
- `results.jsonl` with one result per line;
- `summary.csv` for analysis in a spreadsheet;
- `evaluation.json` with article-level and outlet-level metrics;
- `outlet_summary.csv` with mean-score classifications by outlet;
- `score_distributions.svg` and `outlet_comparison.svg`, corresponding to the two
  custom-dataset figure types in the paper.

The five fixture labels are deliberately obvious Left/Center/Right examples used to
verify plumbing. Five samples cannot measure model quality and their accuracy must not
be compared with the paper's benchmark results.

## Custom-dataset run

After the historical article bodies have been added:

```bash
python3 tools/convert_custom_dataset.py
python3 run.py --validate-only --input data/custom_dataset/articles.jsonl
python3 run.py --input data/custom_dataset/articles.jsonl --limit 100 --embedding hashing
```

The 100-article run is a pipeline and report check. It does not reproduce the paper's
1,300-article custom experiment. Accuracy in this run measures agreement with
AllSides 2025 outlet-derived proxy labels.

## Paper-exact embedding check

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[paper,dev]'
pytest
python run.py --demo --embedding minilm
```

The generated manifest must name
`sentence-transformers/all-MiniLM-L6-v2`, not the hashing fallback.

# Testing and verification

## Deterministic suite

```bash
python3 -m unittest discover -s tests -v
```

The suite uses a fake model backend and covers score boundaries, probability entropy,
data validation, unanimous early exit, 2-vs-1 debate, all-different divergent-pair
selection, stage routing, prediction-token log-probabilities, metrics, and all report
artifacts. It is deterministic and has no network, model, or third-party dependency.

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
- `summary.csv` for analysis in a spreadsheet.

The five fixture labels are deliberately obvious Left/Center/Right examples used to
verify plumbing. Five samples cannot measure model quality and their accuracy must not
be compared with the paper's benchmark results.

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

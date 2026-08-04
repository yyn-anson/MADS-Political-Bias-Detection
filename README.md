# MADS Political Bias Detection

Project repository: **https://github.com/yyn-anson/MADS-Political-Bias-Detection**

MADS classifies the political framing of English-language U.S. news articles as
**Left**, **Center**, or **Right**. Three language-model agents first analyze the
article independently. The pipeline exits early when all three agree; otherwise it
runs the conditional 2-vs-1 or 1-vs-1-vs-1 debate described in the paper.

This rewrite is designed for ordinary users rather than private benchmark owners:
put articles in `data/articles/`, run one command, and open the generated HTML report.
No training data is required and the base installation has no Python dependencies.

> Bias labels are subjective model judgments, not fact-checks. Read the evidence and
> debate trace before using a result in research or editorial decisions.

## Quick start on a Mac

Requirements: Python 3.11 or newer, [Ollama](https://ollama.com), and a local model.
The tested Mac configuration uses `qwen2.5vl:3b` for all three agents.

```bash
# One-time model download, if it is not already installed
ollama pull qwen2.5vl:3b

# Verify the configured local model
python3 run.py --check

# Analyze the five bundled real-world, source-linked samples
python3 run.py --demo
```

Open the `report.html` path printed at the end. The same run directory contains the
complete JSON/JSONL audit trail, CSV summaries, article and outlet evaluation metrics,
and paper-style SVG figures.

For your own data, copy JSON, JSONL, CSV, or TXT files into `data/articles/` and run:

```bash
python3 run.py
```

The minimum JSON format is deliberately small:

```json
{
  "id": "article-001",
  "title": "Article title",
  "text": "The complete article text goes here."
}
```

Optional fields are `source`, `url`, `published_at`, and `label`. A label must be
`Left`, `Center`, or `Right`; when labels are present the report also calculates
accuracy, macro-F1, per-class metrics, and a confusion matrix. See
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for every supported format.

### Existing custom dataset

The committed lookup table preserves AllSides 2025 outlet labels for all 473,989
historical article IDs. Once the corresponding article bodies are placed in
`data/custom_dataset/raw_articles/`, convert and analyze them with:

```bash
python3 tools/convert_custom_dataset.py
python3 run.py --input data/custom_dataset/articles.jsonl --limit 100
```

For this custom evaluation, an article's ground-truth label is assumed to be its
outlet's AllSides 2025 rating. Left and Lean Left become `Left`; Lean Right and Right
become `Right`; Center is unchanged. This is an outlet-derived proxy rather than an
independent article annotation. See
[data/custom_dataset/README.md](data/custom_dataset/README.md) for details.

## What is implemented from the paper

1. Three independent content-only analyses produce scores in `[-3, 3]`.
2. Scores map to Left at `<= -1`, Right at `>= +1`, and Center otherwise.
3. Unanimous panels exit early and return the mean score.
4. A 2-vs-1 split selects the lower-entropy majority agent to debate the dissenter.
5. A three-way split embeds the reasoning and debates the least-similar pair first,
   followed by the remaining agent.
6. Each round alternates challenger and target, preserves shared history, permits
   both agents to revise, and stops on pair consensus, semantic stagnation, or the
   round cap.
7. Unresolved debates use the lower prediction entropy as the tiebreaker.
8. Panel unanimity returns the mean current score; otherwise the winner's score is
   returned.

Ollama's OpenAI-compatible API supplies token log-probabilities, so the entropy
tiebreaker uses the model's Left/Center/Right prediction-token distribution instead
of score magnitude. The exact paper embedder,
`sentence-transformers/all-MiniLM-L6-v2`, is supported as an optional extra:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[paper]'
python run.py --embedding minilm --demo
```

Without that optional package, `auto` mode reports and uses a deterministic
384-dimensional hashing fallback. It exercises the same routing and stagnation
logic fully offline, but production research runs should install MiniLM.

See [docs/METHOD.md](docs/METHOD.md) for the algorithm-to-code mapping and explicit
differences between the paper's diverse-model experiment and the one-model Mac smoke
test.

## Configuration

Edit `mads.toml` or override common settings at the command line:

```bash
python3 run.py --model llama3.2:3b --embedding hashing --max-rounds 2
python3 run.py --input path/to/articles.jsonl --output path/to/reports
python3 run.py --validate-only --input data/articles
```

The default config intentionally gives the three instances different random seeds
and low temperatures. That creates independent test runs while reusing one installed
model. For real evaluation, set three distinct model names in `mads.toml` to recover
the architectural diversity assumed by the paper.

## Repository structure

```text
data/articles/          user-owned input files (empty by default)
data/sample_articles/   five source-linked, paraphrased test fixtures
data/custom_dataset/    custom-corpus conversion instructions and generated data
src/mads/               one reusable implementation of the complete method
tests/                  deterministic unit and integration tests
reports/                generated reports (ignored by Git)
mads.toml               model and method configuration
run.py                  zero-install entry point
tools/                  custom dataset conversion entry point
```

## Testing

The deterministic suite needs no model or network:

```bash
python3 -m unittest discover -s tests -v
```

The real local-model smoke test is:

```bash
python3 run.py --demo --embedding hashing
```

Testing details and expected artifacts are in [docs/TESTING.md](docs/TESTING.md).

## Privacy and reproducibility

- Article metadata is saved in reports but never sent to the model; only `text` is
  used for classification, matching the paper's content-only setup.
- Articles are never truncated silently. Inputs beyond the configured context safety
  limit fail validation with a clear message.
- Every report records model names, seeds, temperatures, configuration, confidence
  source, complete state transitions, termination reasons, and timings.
- The five samples are concise original paraphrases of linked public-affairs releases,
  not copies of proprietary news articles. Their labels are human test-fixture
  judgments, not benchmark ground truth.

## License

See [LICENSE](LICENSE).

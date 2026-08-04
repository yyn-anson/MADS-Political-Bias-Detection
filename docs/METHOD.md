# Method and implementation map

This implementation follows the paper's supplementary Algorithms 1 and 2. The core
orchestration is in `src/mads/engine.py`; prompts and schemas are isolated in
`src/mads/prompts.py`, and model transport is isolated in `src/mads/llm.py`.

## Phase I: independent analysis

Exactly three configured agents receive the same complete article text and no source
metadata. Each returns an understanding paragraph, separate bias reasoning, short
textual evidence, a `[-3, 3]` score, and a final categorical prediction.

The score determines direction using the paper's threshold `tau = 1`. If a model's
text label conflicts with its numeric score, the numeric score is authoritative and
the derived label is recorded. Scores outside the range are clamped.

The model response also carries token log-probabilities. The Ollama adapter locates
the final categorical prediction token, collects Left/Center/Right alternatives,
normalizes them, and calculates Shannon entropy. A model-reported probability vector
is retained only as a compatibility fallback when an endpoint omits log-probabilities.

## Phase II: routing

- **Unanimous:** return the mean of the three scores immediately.
- **Majority (2-vs-1):** choose the lower-entropy agent from the majority pair and
  debate the dissenter.
- **All different:** embed the three reasoning texts, select the pair with minimum
  cosine similarity, debate them, and then debate the winner against the remaining
  agent unless panel unanimity already exists.

`sentence-transformers/all-MiniLM-L6-v2` is the paper-exact embedder. The dependency-
free hashing embedder is an explicit offline fallback and is always named as such in
the report.

## Phase III: pairwise debate

The pair alternates challenger and target. A challenge must acknowledge valid points,
cite article evidence, and reconsider the challenger's own score. The response may
defend or revise the target's score. Both updated states are committed after each
exchange, and the complete shared history is supplied to later exchanges.

Termination is checked in this order:

1. the pair now has the same direction;
2. current and previous exchange embeddings have cosine similarity above `0.90`;
3. the configured round cap is reached.

The lower-entropy agent wins the pair whether the final directions agree or differ.
If all three current states agree, the final score is their mean. Otherwise the final
pair winner's score and direction are returned.

## Paper experiment versus local smoke test

The paper evaluates three architecturally different LLMs. The default `mads.toml`
reuses `qwen2.5vl:3b` three times because that is a practical functional test on a
single Mac. Distinct seeds and temperatures make the calls independent, but this does
not create true model diversity and must not be presented as reproducing the paper's
reported accuracy.

For research replication, configure three different model names and install the
`paper` optional dependency for MiniLM. The routing, debate, confidence, and reporting
code does not change.

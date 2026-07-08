# Adding or Swapping a Model

Every labeler talks to its model through an OpenAI-compatible HTTP endpoint
(vLLM, or anything else that speaks the same protocol). Swapping a model
therefore never requires touching the ensemble logic.

There are two levels of customization:

| You want to… | Effort |
|--------------|--------|
| Serve a different model in one of the three existing slots | Environment variables only — no code |
| Add a model that needs its own prompts, parameters, or output post-processing | Copy one template file |

---

## Option A — Swap a model with environment variables (no code)

Each of the three ensemble slots reads its server URL and model ID from
`config.py`, and every value can be overridden with an environment variable:

| Slot (small ensemble) | URL variable | Model variable |
|-----------------------|--------------|----------------|
| llama32 | `VLLM_LLAMA_URL` | `VLLM_LLAMA_MODEL` |
| qwen3 | `VLLM_QWEN_URL` | `VLLM_QWEN_MODEL` (+ `VLLM_QWEN_THINKING=0` to disable thinking) |
| mistral | `VLLM_MISTRAL_URL` | `VLLM_MISTRAL_MODEL` |

| Slot (regular ensemble) | URL variable | Model variable |
|-------------------------|--------------|----------------|
| qwen3 | `VLLM_QWEN14B_URL` | `VLLM_QWEN14B_MODEL` (+ `VLLM_QWEN14B_THINKING=0`) |
| gptoss | `VLLM_GPTOSS_URL` | `VLLM_GPTOSS_MODEL` |
| mistral | `VLLM_MISTRAL22B_URL` | `VLLM_MISTRAL22B_MODEL` |

Example — replace Mistral-7B with Phi-4 in the small ensemble:

```bash
# 1. Serve the replacement model
vllm serve microsoft/phi-4 --port 8003 --api-key token-abc123

# 2. Point the slot at it
export VLLM_MISTRAL_MODEL="microsoft/phi-4"     # PowerShell: $env:VLLM_MISTRAL_MODEL="microsoft/phi-4"

# 3. Run as usual
python run_batches.py --model small --dataset baly --total 10
```

The slot keeps its internal name (`mistral`) in logs and result files, but all
requests go to your model. The shared prompts and the `RobustJSONExtractor`
output parser work with any instruction-tuned model that can emit JSON.

---

## Option B — Add a custom labeler (one file)

Use this when your model needs different generation parameters, extra
`extra_body` flags, or output cleanup (e.g. stripping `<think>` blocks).

### Step 1 — Copy the template

```bash
cp src/models/custom_labeler_template.py src/models/my_model_labeler.py
```

Rename the class inside (`CustomLabeler` → e.g. `Phi4Labeler`).

### Step 2 — Fill in the TODO sections

The template marks exactly four places:

1. **Prompts** — the defaults match the other labelers; usually keep them.
2. **Generation parameters** — set `temperature` / `max_tokens` / `top_p`
   to your model's recommended values in `__init__`.
3. **`extra_body`** — add server-side flags if your model needs them, e.g.
   `{"chat_template_kwargs": {"enable_thinking": True}}` for Qwen3.
4. **Output cleanup** — strip wrapper tokens before JSON parsing if your
   model emits them (see `_strip_thinking` in `qwen3_labeler.py` for a
   reference).

The output contract for `predict()`:

```python
{
    "lean":      int,   # integer in [-3, 3]
    "direction": str,   # "Left" | "Center" | "Right"
    "reason":    str,   # brief explanation
}
```

Parsing failures must raise (`ValueError`) rather than return a default —
the ensemble counts and reports them as model errors.

### Step 3 — Register the model in config.py

```python
'small_ensemble': {
    ...
    'my_model': {
        'base_url': os.environ.get('VLLM_MYMODEL_URL', 'http://localhost:8004/v1'),
        'model_id': os.environ.get('VLLM_MYMODEL_MODEL', 'your-org/your-model'),
    },
},
```

### Step 4 — Wire it into the ensemble

In `src/ensemble/ensemble_small.py` (or `ensemble_regular.py`), replace one
of the three labeler constructions in `EnsembleMultiModelDetector.__init__`:

```python
from src.models.my_model_labeler import Phi4Labeler

self.mistral_labeler = Phi4Labeler(
    base_url=small_cfg['my_model']['base_url'],
    model_id=small_cfg['my_model']['model_id'],
    api_key=api_key,
)
```

The ensemble accepts any `BaseLabeler` subclass. Discussion support
(`generate_discussion_challenge` / `generate_discussion_response`) is already
implemented in the template, so the two-stage debate works out of the box.

### Step 5 — Run a quick test

```bash
# Verify the import works
python -c "from src.models.my_model_labeler import Phi4Labeler; print('OK')"

# Run 10 articles end-to-end
python run_batches.py --model small --dataset baly --total 10
```

Results land in `ensemble_outputs_small/session_TIMESTAMP/`.

---

## Discussion support

The two-stage collaborative discussion requires each labeler to implement:

1. `generate_discussion_challenge()` — challenge another model's analysis
2. `generate_discussion_response()` — respond to a challenge, possibly
   revising its own score via `final_lean`

Both are already implemented in the template and all shipped labelers using
the shared JSON challenge/response formats. If a labeler does not implement
them, `BaseLabeler` raises `NotImplementedError` the first time that model is
drawn into a discussion — models without discussion support cannot silently
participate.

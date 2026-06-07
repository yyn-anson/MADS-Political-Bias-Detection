# Adding a Custom Model

This walkthrough shows you how to plug any HuggingFace causal-language model into the ensemble in four steps.

---

## Prerequisites

- A HuggingFace model ID (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) or a local path to model weights
- The model must support text generation (causal LM)
- Enough VRAM to run the model (see [MODELS.md](MODELS.md) for guidance)

---

## Step 1 — Copy the template

```bash
cp src/models/custom_labeler_template.py src/models/my_model_labeler.py
```

Open `src/models/my_model_labeler.py` and rename the class:

```python
# Before
class CustomModelLabeler(BaseLabeler):

# After
class MyModelLabeler(BaseLabeler):
```

---

## Step 2 — Fill in the three TODO sections

### TODO 1: Set your model ID and load the model

```python
def __init__(
    self,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",  # ← your model ID
    cache_dir: str = "models/",
    batch_size: int = 1,
    config: Dict = None,
):
    super().__init__(model_name=model_name, cache_dir=cache_dir, batch_size=batch_size)
```

`load_model()` already has a working default using `AutoModelForCausalLM` and `AutoTokenizer`.
Adjust `torch_dtype`, `device_map`, or quantization if needed:

```python
# For 8-bit quantization (reduces VRAM by ~50%):
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

self.model = AutoModelForCausalLM.from_pretrained(
    self.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=self.cache_dir,
)
```

### TODO 2: Format prompts for your model

Most modern chat models expose a `apply_chat_template` method. Use it if available:

```python
def _build_prompt(self, article_text: str) -> str:
    user_content = USER_TEMPLATE.format(article=article_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
```

If your model uses a custom template format (e.g. `[INST]...[/INST]`), build the string directly:

```python
return f"[INST] {SYSTEM_PROMPT}\n\n{user_content} [/INST]"
```

### TODO 3: Parse the output

`predict()` is already wired to `RobustJSONExtractor`, which handles messy JSON embedded in free-form text.
The output contract you must satisfy is:

```python
{
    "lean":      int,   # integer in [-3, 3]
    "direction": str,   # "Left" | "Center" | "Right"
    "reason":    str,   # brief explanation
    "error":     bool,  # True only when inference failed
}
```

The template's default implementation is sufficient for most models. Only override if your model outputs in a non-JSON format.

---

## Step 3 — Wire your labeler into the ensemble

Open `src/ensemble/ensemble_small.py` (for the 3-model small ensemble) or `ensemble_regular.py` (for the regular ensemble) and replace one of the existing labelers:

```python
# Add import at top of file
from src.models.my_model_labeler import MyModelLabeler

# Inside run_ensemble() or wherever labelers are constructed:
labelers = [
    QwenLabeler(model_name="Qwen/Qwen3-4B-Instruct", ...),
    MyModelLabeler(model_name="mistralai/Mistral-7B-Instruct-v0.3", ...),
    MistralLabeler(model_name="mistralai/Mistral-Small-Instruct-2409", ...),
]
```

The ensemble accepts any list of `BaseLabeler` subclasses — no other changes are needed.

---

## Step 4 — Run a quick test

```bash
cd multi_agent_bias_detection

# Verify the import works
python -c "from src.models.my_model_labeler import MyModelLabeler; print('OK')"

# Run 10 articles to confirm end-to-end
python run_batches.py --model small --dataset baly --total 10
```

Expected console output:
```
[INFO] Loading mistralai/Mistral-7B-Instruct-v0.3 ...
[INFO] mistralai/Mistral-7B-Instruct-v0.3 loaded.
[INFO] Processing batch 0 (articles 0-2) ...
...
[INFO] Session complete. Results saved to outputs/ensemble_outputs_small/session_TIMESTAMP/
```

---

## Enabling discussion support (optional)

The two-stage collaborative discussion requires each labeler to be able to:
1. **Challenge** another model's analysis
2. **Respond** to a challenge against its own analysis

By default, `generate_discussion_challenge()` and `generate_discussion_response()` raise `NotImplementedError`. To enable discussion, override them in your labeler.

Look at `src/models/qwen3_labeler.py` for a complete reference implementation — specifically `generate_discussion_challenge()` and `generate_discussion_response()` in the `PoliticalLeaningLabeler` class.

The key pattern:

```python
def generate_discussion_challenge(
    self,
    article_content: str,
    conversation_history: str,
    own_analysis: Dict[str, Any],
    target_analysis: Dict[str, Any],
) -> Tuple[str, str]:
    # Build a prompt asking the model to challenge target_analysis
    # given its own_analysis and the conversation so far
    prompt = ...
    raw = self.pipe(prompt)[0]["generated_text"]
    return prompt, raw

def generate_discussion_response(
    self,
    article_content: str,
    conversation_history: str,
    challenge: str,
    own_analysis: Dict[str, Any],
    challenger_analysis: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    # Build a prompt asking the model to respond to the challenge
    # Returns (prompt_str, updated_analysis_dict)
    prompt = ...
    raw = self.pipe(prompt)[0]["generated_text"]
    parsed = RobustJSONExtractor.extract_json(raw) or {}
    return prompt, parsed
```

If discussion support is not implemented, the ensemble falls back to simple majority voting when models disagree.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ImportError: cannot import name 'MyModelLabeler'` | Wrong module path | Check the file is in `src/models/` and the class name matches |
| `NotImplementedError: ... does not support collaborative discussion` | Discussion methods not overridden | Either override them or confirm the ensemble is set to skip discussion |
| JSON parse failures, `lean` always 0 | Model not following JSON output format | Add an explicit example in `USER_TEMPLATE` or set `temperature=0` |
| CUDA OOM | Model too large for available VRAM | Use 8-bit quantization or a smaller model |
| Very slow inference | `batch_size=1` with a large dataset | Increase `batch_size` in `__init__` and override `label_articles_batch()` |

For more, see [MODELS.md](MODELS.md) for VRAM requirements and [REPRODUCTION.md](REPRODUCTION.md) for end-to-end run instructions.

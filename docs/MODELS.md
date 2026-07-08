# Model Specifications

All models are served by [vLLM](https://github.com/vllm-project/vllm) and accessed via the OpenAI-compatible HTTP API. The Python client only makes HTTP calls — no GPU drivers or local model weights are required on the client machine.

---

## Ensembles

### Small Ensemble (~28 GB VRAM total)

Suited for a single A100-40 GB or two consumer GPUs.

| Role | Model | Size | VRAM (BF16) | Port (default) |
|------|-------|------|-------------|----------------|
| Labeler 1 | `meta-llama/Llama-3.2-3B-Instruct` | 3.2 B | ~6 GB | 8001 |
| Labeler 2 | `Qwen/Qwen3-4B` | 4 B | ~8 GB | 8002 |
| Labeler 3 | `mistralai/Mistral-7B-Instruct-v0.3` | 7 B | ~14 GB | 8003 |

#### Starting the small ensemble servers

Open **three separate terminals** and run one command per terminal:

```bash
# Terminal 1 — Llama-3.2-3B
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --port 8001 \
    --api-key token-abc123 \
    --max-model-len 16384

# Terminal 2 — Qwen3-4B  (thinking mode is enabled in the prompt, no extra flag needed)
vllm serve Qwen/Qwen3-4B \
    --port 8002 \
    --api-key token-abc123 \
    --max-model-len 16384

# Terminal 3 — Mistral-7B
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --port 8003 \
    --api-key token-abc123 \
    --max-model-len 16384
```

Wait until each server prints `INFO:     Application startup complete` before running the ensemble.

---

### Regular Ensemble (~112 GB VRAM total)

Requires multi-GPU (e.g., 4× A100-80 GB or equivalent).

| Role | Model | Size | VRAM (BF16) | Port (default) |
|------|-------|------|-------------|----------------|
| Labeler 1 | `Qwen/Qwen3-14B` | 14.7 B | ~28 GB | 8001 |
| Labeler 2 | `openai/gpt-oss-20b` | 20 B | ~40 GB | 8002 |
| Labeler 3 | `mistralai/Mistral-Small-Instruct-2409` | 22 B | ~44 GB | 8003 |

#### Starting the regular ensemble servers

```bash
# Terminal 1 — Qwen3-14B
vllm serve Qwen/Qwen3-14B \
    --port 8001 \
    --api-key token-abc123 \
    --max-model-len 32768

# Terminal 2 — GPT-OSS-20B
vllm serve openai/gpt-oss-20b \
    --port 8002 \
    --api-key token-abc123 \
    --max-model-len 8192

# Terminal 3 — Mistral-Small-22B
vllm serve mistralai/Mistral-Small-Instruct-2409 \
    --port 8003 \
    --api-key token-abc123 \
    --max-model-len 8192
```

---

## Useful vLLM flags

| Flag | When to use |
|------|-------------|
| `--tensor-parallel-size N` | Spread one model across N GPUs |
| `--gpu-memory-utilization 0.90` | Allow up to 90 % VRAM (default 0.90) |
| `--dtype bfloat16` | Explicit BF16 (default on Ampere+) |
| `--max-model-len 8192` | Cap context to reduce KV-cache memory |
| `--served-model-name my-alias` | Let clients address the model by an alias |

Example for a model that needs two GPUs:

```bash
vllm serve Qwen/Qwen3-14B \
    --port 8001 \
    --api-key token-abc123 \
    --tensor-parallel-size 2
```

---

## Installing vLLM

Install vLLM on the **GPU server** (not the client):

```bash
pip install vllm
```

The client machine only needs `openai`:

```bash
pip install openai
```

See [vllm.ai](https://docs.vllm.ai) for full installation options (CUDA, ROCm, CPU-only).

---

## Verifying a server is ready

```bash
curl http://localhost:8001/v1/models \
  -H "Authorization: Bearer token-abc123"
```

Expected output:

```json
{"object": "list", "data": [{"id": "meta-llama/Llama-3.2-3B-Instruct", ...}]}
```

---

## Changing endpoints

All URLs are controlled by environment variables (see `config.py`):

```bash
# PowerShell
$env:VLLM_LLAMA_URL   = "http://gpu-server:8001/v1"
$env:VLLM_QWEN_URL    = "http://gpu-server:8002/v1"
$env:VLLM_MISTRAL_URL = "http://gpu-server:8003/v1"
$env:VLLM_API_KEY     = "your-secret-key"

# Linux / macOS
export VLLM_LLAMA_URL="http://gpu-server:8001/v1"
export VLLM_QWEN_URL="http://gpu-server:8002/v1"
export VLLM_MISTRAL_URL="http://gpu-server:8003/v1"
export VLLM_API_KEY="your-secret-key"
```

---

## Model performance

For reported per-model and ensemble performance on the 3-class task
(Left / Center / Right), refer to the original paper.

Qualitative notes from development runs:

| Model | Notes |
|-------|-------|
| Qwen3-14B | Strongest with thinking mode |
| GPT-OSS-20B | High nuance |
| Mistral-Small-22B | Best instruction following |
| Llama-3.2-3B | Fastest inference |
| Qwen3-4B | Good balance of speed/accuracy |

---

For implementation details see `src/models/`. To add a custom model, see `docs/ADDING_A_MODEL.md`.

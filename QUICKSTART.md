# Quick Start Guide

Get the Multi-Agent Bias Detection system running in under 10 minutes.

---

## Prerequisites

| Component | Where it runs | Requirement |
|-----------|--------------|-------------|
| vLLM servers | **GPU machine** | Python 3.10+, CUDA 11.8+, NVIDIA GPU (see VRAM table below) |
| Python client | **Your machine** | Python 3.10+ |

> The GPU machine and your machine can be the same computer, or separate machines on the same network.

### VRAM requirements

| Ensemble | Models | VRAM needed |
|----------|--------|-------------|
| Small | Llama-3.2-3B + Qwen3-4B + Mistral-7B | ~28 GB total |
| Regular | Qwen3-14B + GPT-OSS-20B + Mistral-Small-22B | ~112 GB total |

---

## Step 1: Start vLLM servers (on the GPU machine)

Install vLLM once on the GPU machine:

```bash
pip install vllm
```

Then open **three separate terminals** and run one command each.

### Small ensemble

```bash
# Terminal 1
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --port 8001 --api-key token-abc123 --max-model-len 16384

# Terminal 2
vllm serve Qwen/Qwen3-4B \
    --port 8002 --api-key token-abc123 --max-model-len 16384

# Terminal 3
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --port 8003 --api-key token-abc123 --max-model-len 16384
```

### Regular ensemble

```bash
# Terminal 1
vllm serve Qwen/Qwen3-14B \
    --port 8001 --api-key token-abc123 --max-model-len 32768

# Terminal 2
vllm serve openai/gpt-oss-20b \
    --port 8002 --api-key token-abc123 --max-model-len 8192

# Terminal 3
vllm serve mistralai/Mistral-Small-Instruct-2409 \
    --port 8003 --api-key token-abc123 --max-model-len 8192
```

Wait until each terminal prints:
```
INFO:     Application startup complete.
```

**If the GPU machine is not localhost**, set these env vars on your client machine before step 3:

```bash
# Linux / macOS
export VLLM_LLAMA_URL="http://gpu-server-ip:8001/v1"
export VLLM_QWEN_URL="http://gpu-server-ip:8002/v1"
export VLLM_MISTRAL_URL="http://gpu-server-ip:8003/v1"
export VLLM_API_KEY="token-abc123"

# PowerShell
$env:VLLM_LLAMA_URL   = "http://gpu-server-ip:8001/v1"
$env:VLLM_QWEN_URL    = "http://gpu-server-ip:8002/v1"
$env:VLLM_MISTRAL_URL = "http://gpu-server-ip:8003/v1"
$env:VLLM_API_KEY     = "token-abc123"
```

---

## Step 2: Install client dependencies

On the machine you will run the Python scripts from:

```bash
git clone https://github.com/yyn-anson/MADS-Political-Bias-Detection.git
cd MADS-Political-Bias-Detection
pip install -r requirements.txt
```

---

## Step 3: Verify servers are reachable

```bash
curl http://localhost:8001/v1/models -H "Authorization: Bearer token-abc123"
# Expected: {"object":"list","data":[{"id":"meta-llama/Llama-3.2-3B-Instruct",...}]}
```

---

## Step 4: Run your first evaluation

```bash
# Small ensemble (servers on ports 8001-8003)
python run_batches.py --model small --dataset baly --total 10

# Regular ensemble
python run_batches.py --model regular --dataset baly --total 10
```

Expected console output:
```
Running small ensemble model on baly dataset
Total articles to process: 10

Batch 1: Processing articles 0-8
[OK] Batch 0-8 completed successfully

AGGREGATING BATCH RESULTS
Accuracy: 0.7500 (75.00%)
Macro F1:  0.7234
```

Results are written to `outputs/ensemble_outputs_small/session_TIMESTAMP/`.

---

## Common commands

```bash
# Process 1000 articles
python run_batches.py --model small --dataset baly --total 1000

# Resume an interrupted batch
python run_batches.py --resume outputs/ensemble_outputs_small/session_20260101_120000

# Outlet-level evaluation
python run_outlet_evaluation.py small

# Different datasets
python run_batches.py --model small --dataset budak --total 100
python run_batches.py --model small --dataset ad_fontes --total 100
```

---

## Troubleshooting

**`RuntimeError: vLLM server at http://localhost:8001/v1 is not reachable`**
The server on that port is not running or not yet ready. Check that `Application startup complete` appeared in that terminal, and that the port/URL env vars match.

**Server starts but model is wrong**
Each port must serve the exact model ID listed above. Check `vllm serve` was given the right `meta-llama/Llama-3.2-3B-Instruct` / `Qwen/Qwen3-4B` / `mistralai/Mistral-7B-Instruct-v0.3` argument.

**Out of VRAM**
Add `--max-model-len 8192` to reduce KV-cache size, or use the small ensemble instead of regular.

**Multi-GPU setup**
Add `--tensor-parallel-size 2` (or higher) to the `vllm serve` command to shard a single model across multiple GPUs.

---

For full documentation see [docs/MODELS.md](docs/MODELS.md) and [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

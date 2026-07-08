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

Expected console output (metric values depend on your models - refer to the
original paper for reported performance):

```
Running small ensemble model on baly dataset
Total articles to process: 10
Articles per run: 8
Output directory: ensemble_outputs_small\session_20260101_120000
============================================================

Batch 1: Processing articles 0-8
----------------------------------------
[OK] Batch 0-8 completed successfully

Batch 2: Processing articles 8-10
----------------------------------------
[OK] Batch 8-10 completed successfully

============================================================
AGGREGATING BATCH RESULTS
============================================================
...
4. OVERALL ENSEMBLE PERFORMANCE
============================================================
Total articles processed: 10

Accuracy: 0.XXXX (XX.XX%)
Macro F1: 0.XXXX
...
7. ABLATION STUDY: Impact of Collaborative Discussion
============================================================
...
Aggregated results saved to: ensemble_outputs_small\session_20260101_120000\aggregated_results.json
```

During each batch the ensemble phases appear in the logs:
`PHASE 1: Individual Model Analysis` -> `PHASE 2: Collaborative Discussion for
Disagreements` -> (when all three models disagree) `Article N: Triggering
collaborative discussion` with round-by-round score adjustments.

---

## Step 5: Read the results

Everything is written to `ensemble_outputs_small/session_TIMESTAMP/`
(or `ensemble_outputs/` for the regular ensemble), relative to the project root:

```
ensemble_outputs_small/session_TIMESTAMP/
├── aggregated_results.json          # All metrics: per-model, consensus-only,
│                                    #   overall ensemble, ablation study
├── batch_0_8_results.json           # Raw per-batch results + statistics
├── article_list.json                # Which articles were processed
├── article_0000/                    # Per-article decisions
│   ├── llama32_response.json        #   each model's score + reasoning
│   ├── qwen3_response.json
│   ├── mistral_response.json
│   └── final_decision.json          #   consensus outcome (example below)
├── individual_models/               # Per-model accuracy vs ground truth
└── collaborative_discussions/       # Full debate transcripts
    └── article_0003/
        ├── discussion_summary.json  #   initial -> final positions per agent
        └── round_*_..._prompt/_response.txt   # every prompt & raw reply
```

Example `final_decision.json` - all three models agreed, so the final score is
their average and no discussion was needed:

```json
{
  "article_id": 0,
  "filename": "2996_fmioocJFHjjkfHXU.json",
  "models_used": ["llama32", "qwen3", "mistral"],
  "individual_scores": {
    "llama32": { "score": -2, "direction": "Left" },
    "qwen3":   { "score": -2, "direction": "Left" },
    "mistral": { "score": -3, "direction": "Left" }
  },
  "consensus_type": "unanimous",
  "final_score": -2.33,
  "final_direction": "Left"
}
```

When the three models split three ways (`"consensus_type": "all_different"`),
the result additionally records the debate outcome: `stage1_winner`,
`stage2_winner`, `final_agent_states`, and `convergence_type`.

---

## Common commands

```bash
# Process 1000 articles
python run_batches.py --model small --dataset baly --total 1000

# Resume an interrupted batch (auto-detects dataset, model type, batch size,
# and continues from the first unprocessed article)
python run_batches.py --resume ensemble_outputs_small/session_20260101_120000

# Outlet-level evaluation (per-outlet report + violin/beeswarm plots)
python run_outlet_evaluation.py small

# Different datasets
python run_batches.py --model small --dataset budak --total 100
python run_batches.py --model small --dataset ad_fontes --total 100
```

---

## Swapping a model (no code changes)

Any of the three slots can serve a different model - set the slot's env vars
and run as usual:

```bash
# Example: replace Mistral-7B with Phi-4
vllm serve microsoft/phi-4 --port 8003 --api-key token-abc123

export VLLM_MISTRAL_MODEL="microsoft/phi-4"     # PowerShell: $env:VLLM_MISTRAL_MODEL="microsoft/phi-4"
python run_batches.py --model small --dataset baly --total 10
```

Slot variables: `VLLM_LLAMA_URL/_MODEL`, `VLLM_QWEN_URL/_MODEL`
(+ `VLLM_QWEN_THINKING=0` to disable thinking mode), `VLLM_MISTRAL_URL/_MODEL`;
regular ensemble: `VLLM_QWEN14B_*`, `VLLM_GPTOSS_*`, `VLLM_MISTRAL22B_*`.
See [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) for custom labelers.

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

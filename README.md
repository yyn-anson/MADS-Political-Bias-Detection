# Multi-Agent Political Bias Detection System

A production-ready system for detecting political bias in news articles using collaborative multi-agent LLMs with two-stage discussion framework.

---

## Overview

This system employs multiple large language models (LLMs) in a collaborative framework to detect political bias in news articles. When models disagree, they engage in structured two-stage discussions to reach consensus, significantly improving accuracy over individual models or simple averaging.

---

## System Requirements

| Component | Where it runs | Requirement |
|-----------|--------------|-------------|
| vLLM servers | **GPU machine** | Python 3.10+, CUDA 11.8+, NVIDIA GPU |
| Python client | **Your machine** | Python 3.10+, no GPU needed |

**VRAM**: ~28 GB for the small ensemble, ~112 GB for the regular ensemble.
The GPU machine and your machine can be the same computer or separate hosts.

---

## Quick Start

### 1. Start vLLM servers (on the GPU machine)

Install vLLM once on the GPU machine:
```bash
pip install vllm
```

Open **three terminals** and start one model per terminal (small ensemble shown):

```bash
# Terminal 1
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8001 --api-key token-abc123 --max-model-len 16384

# Terminal 2
vllm serve Qwen/Qwen3-4B --port 8002 --api-key token-abc123 --max-model-len 16384

# Terminal 3
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8003 --api-key token-abc123 --max-model-len 16384
```

Wait until each terminal prints `INFO: Application startup complete.`

See **[docs/MODELS.md](docs/MODELS.md)** for regular ensemble commands and multi-GPU flags.

### 2. Install client dependencies

```bash
git clone https://github.com/yyn-anson/MADS-Political-Bias-Detection.git
cd MADS-Political-Bias-Detection
pip install -r requirements.txt
```

If the GPU machine is on a different host, point the client at it:
```bash
export VLLM_LLAMA_URL="http://gpu-server:8001/v1"
export VLLM_QWEN_URL="http://gpu-server:8002/v1"
export VLLM_MISTRAL_URL="http://gpu-server:8003/v1"
export VLLM_API_KEY="token-abc123"
```

### 3. Run Evaluation

```bash
# Small models (faster, less VRAM)
python run_batches.py --model small --dataset baly --total 100

# Regular models (higher accuracy)
python run_batches.py --model regular --dataset baly --total 100

# Outlet-level evaluation
python run_outlet_evaluation.py small
```

### 4. View Results

Results are saved in `ensemble_outputs/` or `ensemble_outputs_small/` (relative to the project root):
```
ensemble_outputs_small/session_TIMESTAMP/
├── aggregated_results.json        # Overall metrics
├── batch_0_8_results.json         # Individual batch results
├── individual_models/              # Per-model performance
└── collaborative_discussions/      # Discussion transcripts
```

---

## Environment Setup

### Client machine

```bash
pip install -r requirements.txt
```

Key packages installed: `openai` (vLLM HTTP client), `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `openpyxl`.

### GPU server (vLLM)

```bash
pip install vllm
# Optional: HuggingFace CLI for downloading gated models
pip install huggingface_hub
huggingface-cli login   # paste your HF_TOKEN when prompted
```

See [docs/MODELS.md](docs/MODELS.md) for per-model `vllm serve` commands.

---

## Model Configurations

### Regular Ensemble (Higher Accuracy)
| Model | Port | VRAM |
|-------|------|------|
| Qwen3-14B (thinking mode) | 8001 | ~28 GB |
| GPT-OSS-20B | 8002 | ~40 GB |
| Mistral-Small-22B | 8003 | ~44 GB |

**Batch Size**: 3 articles | **Total VRAM**: ~112 GB

### Small Ensemble (Faster, Efficient)
| Model | Port | VRAM |
|-------|------|------|
| Llama-3.2-3B | 8001 | ~6 GB |
| Qwen3-4B | 8002 | ~8 GB |
| Mistral-7B | 8003 | ~14 GB |

**Batch Size**: 8 articles | **Total VRAM**: ~28 GB

### Scoring scale

All models rate each article on a **-3 to +3 integer scale**:

| Score | Meaning |
|-------|---------|
| -3 | Strong left |
| -2 | Moderate left |
| -1 | Slight left |
| 0 | Neutral / balanced |
| +1 | Slight right |
| +2 | Moderate right |
| +3 | Strong right |

For **accuracy / F1 metrics** the score is collapsed to 3 classes: score <= -1 -> Left, -1 < score < 1 -> Center, score >= 1 -> Right.
For **outlet-level violin plots** the raw -3..+3 scores are used directly, showing the full distribution of individual article predictions per outlet.

---

## Project Structure

```
multi_agent_bias_detection/
├── README.md                    # This file
├── QUICKSTART.md                # 5-minute quick start guide
├── requirements.txt             # Python dependencies
├── config.py                    # System configuration
├── run_batches.py               # Batch processing runner
├── run_outlet_evaluation.py     # Complete evaluation workflow
│
├── src/                         # Source code
│   ├── ensemble/                # Ensemble implementations
│   │   ├── ensemble_regular.py  # Regular (14B/20B/22B) ensemble
│   │   └── ensemble_small.py    # Small (3B/4B) ensemble
│   ├── models/                  # Model wrappers
│   │   ├── base_labeler.py      # Abstract interface (extend this for custom models)
│   │   ├── custom_labeler_template.py  # Starter template for your own model
│   │   ├── qwen3_labeler.py
│   │   ├── llama32_labeler.py
│   │   ├── gptoss_labeler.py
│   │   └── mistral_labeler.py
│   ├── utils/                   # Utility modules
│   └── evaluation/              # Evaluation tools
│
├── tools/                       # Dataset preparation tools
│   ├── create_balanced_dataset.py
│   └── label_articles_by_outlet.py  # Label raw articles by AllSides outlet ratings
│
├── data/                        # Data directory (user-provided)
│   └── balanced_datasets/       # Balanced datasets by bias label
│
├── models/                      # Model cache (auto-populated)
├── outputs/                     # Results (auto-created)
└── docs/                        # Detailed documentation
    ├── ADDING_A_MODEL.md        # Custom model integration walkthrough
    ├── ARCHITECTURE.md          # Pipeline and consensus logic
    ├── MODELS.md                # VRAM requirements table
    ├── DATASETS.md              # Dataset formats
    └── REPRODUCTION.md          # Step-by-step reproduction commands
```

---

## Usage Examples

### Basic Batch Processing

```bash
# Process 100 articles with small models
python run_batches.py --model small --dataset baly --total 100

# Resume interrupted batch
python run_batches.py --resume ensemble_outputs_small/session_20250120_143022

# Custom batch size
python run_batches.py --model small --dataset baly --batch-size 10
```

### Outlet-Level Evaluation

```bash
# Complete workflow: ensemble + evaluation
python run_outlet_evaluation.py small

# Use existing ensemble results
python run_outlet_evaluation.py small --skip-ensemble --ensemble-dir ensemble_outputs_small/session_20250120_143022
```

### Dataset Preparation

```bash
# Create balanced Baly dataset
python tools/create_balanced_dataset.py --dataset baly --n-samples 1000

# Create balanced Budak dataset
python tools/create_balanced_dataset.py --dataset budak --n-samples 500

# Create custom outlet dataset
python tools/create_balanced_dataset.py --dataset custom --samples-per-outlet 100
```

---

## Using Your Own Model

Any model served by vLLM (or any OpenAI-compatible API) can be plugged in without changing the ensemble logic.

1. Start your model's vLLM server:
   ```bash
   vllm serve your-org/your-model --port 8004 --api-key token-abc123
   ```

2. Copy the labeler template:
   ```bash
   cp src/models/custom_labeler_template.py src/models/my_model_labeler.py
   ```

3. Fill in the `TODO` sections: constructor params, any `extra_body` flags, and thinking-token stripping if needed.

4. Wire it into the ensemble `__init__` and run:
   ```bash
   python run_batches.py --model small --dataset baly --total 10
   ```

See **[docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md)** for a full walkthrough.

---

## Documentation

- **[docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md)**: Step-by-step guide to plugging in a custom model
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Two-stage discussion system architecture
- **[docs/MODELS.md](docs/MODELS.md)**: Model specifications and parameters
- **[docs/DATASETS.md](docs/DATASETS.md)**: Dataset formats and sources
- **[docs/REPRODUCTION.md](docs/REPRODUCTION.md)**: Complete reproduction guide
- **[data/README.md](data/README.md)**: Data preparation instructions

---

## Methodology

### Two-Stage Collaborative Discussion

When all three models disagree on bias direction (Left/Center/Right):

**Stage 1**: All-model debate until majority consensus (2/3 agree)
- Models challenge each other's analyses
- Can adjust positions based on arguments
- Continues until majority or max rounds (8)

**Stage 2**: Winner vs. Minority debate
- Representative of majority debates minority
- Final winner determined by convergence or conviction
- All models adopt winning position

This structured approach significantly improves accuracy over simple averaging or voting.

---

## Performance

For reported benchmark performance (Baly, Budak, Ad Fontes), please refer to the original paper.

Numbers vary with model versions, prompts, sampling parameters, and hardware. To measure
performance on your own setup, run the pipeline on a balanced dataset - per-model,
consensus-only, overall-ensemble, and ablation metrics are produced automatically in
`aggregated_results.json`:

```bash
python run_batches.py --model small --dataset baly
```

---

## Troubleshooting

**`RuntimeError: vLLM server at http://localhost:8001/v1 is not reachable`**
The vLLM server on that port is not running or not yet ready. Confirm `Application startup complete` appeared in that terminal and that port/URL env vars match.

**Server on a remote host not reachable**
Set `VLLM_LLAMA_URL` / `VLLM_QWEN_URL` / `VLLM_MISTRAL_URL` to point at the remote IP. Make sure ports 8001-8003 are open in the firewall.

**Out of VRAM on the GPU server**
Add `--max-model-len 8192` to the `vllm serve` command to shrink KV-cache memory, or add `--tensor-parallel-size 2` to shard across two GPUs.

**Dataset Not Found**
```bash
ls data/balanced_datasets/balanced_baly/
```
The balanced datasets are included in this repo. If the directory is empty, re-clone or check the gitignore.

For more troubleshooting, see [docs/REPRODUCTION.md](docs/REPRODUCTION.md#troubleshooting)

---

## Citation

If you use this system in your research, please cite the original paper.

---

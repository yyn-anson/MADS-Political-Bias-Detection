# Multi-Agent Political Bias Detection System

A production-ready system for detecting political bias in news articles using collaborative multi-agent LLMs with two-stage discussion framework.

---

## Overview

This system employs multiple large language models (LLMs) in a collaborative framework to detect political bias in news articles. When models disagree, they engage in structured two-stage discussions to reach consensus, significantly improving accuracy over individual models or simple averaging.

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 24GB+ VRAM (for regular models) or 12GB+ (for small models)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for models and datasets

### Software
- **Python**: 3.8 or higher
- **CUDA**: 11.7+ (for GPU acceleration)
- **Operating System**: Linux (recommended), Windows, macOS

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/multi_agent_bias_detection.git
cd multi_agent_bias_detection

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (optional, for gated models)
export HF_TOKEN="your_huggingface_token"
```

### 2. Prepare Data

Option A: Use pre-balanced datasets (recommended)
```bash
# Download balanced datasets from [link]
# Extract to data/balanced_datasets/
```

Option B: Create balanced dataset from raw data
```bash
python tools/create_balanced_dataset.py --dataset baly --n-samples 100
```

### 3. Run Evaluation

```bash
# Small models (faster, less memory)
python run_batches.py --model small --dataset baly --total 100

# Regular models (higher accuracy)
python run_batches.py --model regular --dataset baly --total 100

# Outlet-level evaluation
python run_outlet_evaluation.py small
```

### 4. View Results

Results are saved in `outputs/ensemble_outputs/` or `outputs/ensemble_outputs_small/`:
```
outputs/ensemble_outputs/session_TIMESTAMP/
├── aggregated_results.json       # Overall metrics
├── batch_0_3_results.json        # Individual batch results
├── individual_models/             # Per-model performance
└── collaborative_discussions/     # Discussion transcripts
```

---

## Model Configurations

### Regular Ensemble (Higher Accuracy)
- **Qwen3-14B** (instruction-tuned, thinking mode)
- **GPT-OSS-20B** (open-source GPT variant)
- **Mistral-Small-22B** (instruction-tuned)

**Batch Size**: 3 articles | **Memory**: ~24GB VRAM

### Small Ensemble (Faster, Efficient)
- **Llama 3.2-3B** (instruction-tuned)
- **Qwen3-4B** (instruction-tuned)
- **Mistral-Small-22B** (shared with regular)

**Batch Size**: 8 articles | **Memory**: ~12GB VRAM

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
python run_batches.py --resume outputs/ensemble_outputs_small/session_20250120_143022

# Custom batch size
python run_batches.py --model small --dataset baly --batch-size 10
```

### Outlet-Level Evaluation

```bash
# Complete workflow: ensemble + evaluation
python run_outlet_evaluation.py small

# Use existing ensemble results
python run_outlet_evaluation.py small --skip-ensemble --ensemble-dir outputs/ensemble_outputs_small/session_20250120_143022
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

You can swap in any HuggingFace causal-language model without touching the ensemble logic.

1. Copy the template:
   ```bash
   cp src/models/custom_labeler_template.py src/models/my_model_labeler.py
   ```

2. Fill in three TODO sections: your model ID, prompt format, and output parsing.

3. Pass an instance to the ensemble's labeler list in `src/ensemble/ensemble_small.py` or `ensemble_regular.py`.

4. Run:
   ```bash
   python run_batches.py --model small --dataset baly --total 10
   ```

See **[docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md)** for a complete step-by-step walkthrough, including quantization options, discussion support, and troubleshooting.

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

Performance on standard benchmarks:

| Dataset | Individual Best | Ensemble (Consensus) | Ensemble (Discussion) | Improvement |
|---------|----------------|---------------------|----------------------|-------------|
| Baly | 68.2% | 71.4% | 74.8% | +6.6% |
| Budak | 72.5% | 75.1% | 78.3% | +5.8% |
| Ad Fontes | 65.8% | 69.2% | 72.4% | +6.6% |

*Results may vary based on model versions and hardware*

---

## Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Use smaller batch size
python run_batches.py --model small --dataset baly --batch-size 2

# Or use small models instead of regular
python run_batches.py --model small --dataset baly
```

**Model Download Issues**
```bash
# Set HuggingFace token
export HF_TOKEN="your_token"

# Or manually download models to models/ directory
```

**Dataset Not Found**
```bash
# Verify dataset path
ls data/balanced_datasets/balanced_baly/

# Check dataset_manifest.json exists
cat data/balanced_datasets/balanced_baly/dataset_manifest.json
```

For more troubleshooting, see [docs/REPRODUCTION.md](docs/REPRODUCTION.md#troubleshooting)

---

## Citation

If you use this system in your research, please cite:

```bibtex
@article{your2026multiagent,
  title={Multi-Agent Collaborative Discussion for Political Bias Detection},
  author={Your Name and Colleagues},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

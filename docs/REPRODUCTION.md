# Complete Reproduction Guide

Step-by-step instructions for reproducing all experiments in the Multi-Agent Bias Detection System.

---

## 🎯 Prerequisites

### Hardware Requirements

**Minimum (Small Ensemble)**:
- GPU: NVIDIA GPU with 12GB+ VRAM
- RAM: 16GB system memory
- Storage: 80GB free space
- CPU: 8+ cores recommended

**Recommended (Regular Ensemble)**:
- GPU: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090, RTX 4090, A100)
- RAM: 32GB system memory
- Storage: 150GB free space
- CPU: 16+ cores

### Software Requirements

```bash
# Operating System
- Linux (Ubuntu 20.04+) - Recommended
- Windows 10/11 with WSL2
- macOS (CPU only, slower)

# Python
- Python 3.8 or higher
- pip or conda package manager

# CUDA (for GPU acceleration)
- CUDA 11.7 or higher
- cuDNN 8.0 or higher
```

---

## 📥 Step 1: Environment Setup

### Clone Repository

```bash
git clone https://github.com/your-username/multi_agent_bias_detection.git
cd multi_agent_bias_detection
```

### Create Virtual Environment

Option A: Using conda (recommended)
```bash
conda create -n bias_detection python=3.9
conda activate bias_detection
```

Option B: Using venv
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Set HuggingFace Token

```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"

# Or add to config.py
# config['huggingface']['token'] = 'hf_your_token_here'
```

---

## 📊 Step 2: Dataset Preparation

### Download Pre-Balanced Datasets (Recommended)

```bash
# Download from [link to be added]
# Extract to data/balanced_datasets/

# Verify structure
ls data/balanced_datasets/balanced_baly/
# Should see: article_*.json files + dataset_manifest.json
```

### OR Create Balanced Datasets from Raw Data

#### Baly Dataset

```bash
# Place raw Baly dataset in data/raw/baly/
python tools/create_balanced_dataset.py --dataset baly --n-samples 1000

# Verify creation
cat data/balanced_datasets/balanced_baly/dataset_manifest.json
```

#### Budak Dataset

```bash
python tools/create_balanced_dataset.py --dataset budak --n-samples 600
```

#### Ad Fontes Dataset

```bash
python tools/create_balanced_dataset.py --dataset ad_fontes --n-samples 900
```

#### Custom Outlet Dataset

```bash
python tools/create_balanced_dataset.py --dataset custom --samples-per-outlet 100
```

---

## 🧪 Step 3: Run Experiments

### Experiment 1: Small Ensemble on Baly Dataset

**Purpose**: Baseline evaluation with memory-efficient models

```bash
# Run 100 articles
python run_batches.py --model small --dataset baly --total 100

# Expected runtime: ~2 hours
# Expected accuracy: ~72%
```

**Output**:
```
outputs/ensemble_outputs_small/session_TIMESTAMP/
├── aggregated_results.json
├── batch_*_results.json
└── individual_models/
```

### Experiment 2: Regular Ensemble on Baly Dataset

**Purpose**: High-accuracy evaluation with larger models

```bash
# Run 100 articles
python run_batches.py --model regular --dataset baly --total 100

# Expected runtime: ~4 hours
# Expected accuracy: ~75%
```

### Experiment 3: Ablation Study (Discussion Impact)

**Purpose**: Measure improvement from collaborative discussion

```bash
# Run both ensembles
python run_batches.py --model small --dataset baly --total 1000
python run_batches.py --model regular --dataset baly --total 1000

# Compare results
cat outputs/ensemble_outputs_small/session_*/aggregated_results.json | \
  jq '.ablation_study'
```

**Expected Results**:
- Consensus-only accuracy: ~71%
- With discussion accuracy: ~75%
- Improvement: +4-6%

### Experiment 4: Cross-Dataset Evaluation

**Purpose**: Test generalization across datasets

```bash
# Baly
python run_batches.py --model small --dataset baly --total 500

# Budak
python run_batches.py --model small --dataset budak --total 500

# Ad Fontes
python run_batches.py --model small --dataset ad_fontes --total 500

# Compare metrics across datasets
```

### Experiment 5: Outlet-Level Analysis

**Purpose**: Characterize media outlet bias patterns

```bash
# Complete workflow: ensemble + evaluation + visualization
python run_outlet_evaluation.py small

# Outputs
outputs/ensemble_outputs_small/session_*/outlet_evaluation_*/
├── outlet_evaluation_report.json
└── visualizations/
    ├── violin_plot_raw_scores.pdf
    ├── confusion_matrix.pdf
    ├── per_class_performance.pdf
    └── outlet_comparison.pdf
```

---

## 📈 Step 4: Analyze Results

### View Aggregated Metrics

```bash
# Find latest session
SESSION=$(ls -t outputs/ensemble_outputs_small/ | head -1)

# View overall metrics
cat outputs/ensemble_outputs_small/$SESSION/aggregated_results.json | \
  jq '.overall_ensemble_metrics'

# Output:
# {
#   "accuracy": 0.75,
#   "macro_f1": 0.7234,
#   "weighted_f1": 0.7456
# }
```

### Compare Individual Models

```bash
# Qwen performance
cat outputs/ensemble_outputs_small/$SESSION/individual_models/qwen_results.json | \
  jq '.accuracy'

# GPT-OSS performance
cat outputs/ensemble_outputs_small/$SESSION/individual_models/gptoss_results.json | \
  jq '.accuracy'

# Mistral performance
cat outputs/ensemble_outputs_small/$SESSION/individual_models/mistral_results.json | \
  jq '.accuracy'
```

### Discussion Analysis

```bash
# Count articles that triggered discussion
ls outputs/ensemble_outputs_small/$SESSION/collaborative_discussions/ | wc -l

# View a discussion transcript
cat outputs/ensemble_outputs_small/$SESSION/collaborative_discussions/article_0005/discussion_summary.json | jq '.'

# Statistics
cat outputs/ensemble_outputs_small/$SESSION/aggregated_results.json | \
  jq '.discussion_breakdown'
```

---

## 🔬 Step 5: Advanced Experiments

### Vary Batch Size

```bash
# Test different batch sizes
python run_batches.py --model small --dataset baly --batch-size 4
python run_batches.py --model small --dataset baly --batch-size 8
python run_batches.py --model small --dataset baly --batch-size 12
```

### Partial Dataset Processing

```bash
# Process specific range
python run_batches.py --model small --dataset baly --total 1000

# Resume if interrupted
python run_batches.py --resume outputs/ensemble_outputs_small/session_TIMESTAMP
```

### Ensemble Comparison

```bash
# Run both ensembles on same dataset
python run_batches.py --model small --dataset baly --total 500
python run_batches.py --model regular --dataset baly --total 500

# Compare accuracy
SMALL=$(cat outputs/ensemble_outputs_small/session_*/aggregated_results.json | jq '.overall_ensemble_metrics.accuracy')
REGULAR=$(cat outputs/ensemble_outputs/session_*/aggregated_results.json | jq '.overall_ensemble_metrics.accuracy')

echo "Small ensemble: $SMALL"
echo "Regular ensemble: $REGULAR"
```

---

## 📊 Step 6: Generate Publication Figures

### Outlet Comparison Plot

```bash
python run_outlet_evaluation.py regular

# Output: outputs/ensemble_outputs/session_*/outlet_evaluation_*/visualizations/outlet_comparison.pdf
```

### Confusion Matrix

```bash
# Confusion matrix is auto-generated in outlet evaluation
# Location: visualizations/confusion_matrix.pdf
```

### Performance Comparison Table

```python
import json
import pandas as pd

# Load results
results_files = [
    'outputs/ensemble_outputs_small/session_*/aggregated_results.json',
    'outputs/ensemble_outputs/session_*/aggregated_results.json'
]

# Create comparison table
data = []
for file in results_files:
    with open(file) as f:
        results = json.load(f)
        data.append({
            'Ensemble': 'Small' if 'small' in file else 'Regular',
            'Accuracy': results['overall_ensemble_metrics']['accuracy'],
            'Macro F1': results['overall_ensemble_metrics']['macro_f1'],
            'Improvement': results['ablation_study']['accuracy_improvement']
        })

df = pd.DataFrame(data)
print(df.to_latex())  # For LaTeX papers
```

---

## 🐛 Troubleshooting

### Out of Memory Errors

```bash
# Reduce batch size
python run_batches.py --model small --dataset baly --batch-size 2

# OR use smaller ensemble
python run_batches.py --model small --dataset baly
```

### Model Download Failures

```bash
# Check token
echo $HF_TOKEN

# Manually download model
huggingface-cli login
huggingface-cli download Qwen/Qwen3-14B --local-dir models/models--Qwen--Qwen3-14B
```

### CUDA Errors

```bash
# Verify CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Dataset Loading Errors

```bash
# Verify dataset structure
python -c "
from config import get_config
import json
config = get_config()
manifest = json.load(open('data/balanced_datasets/balanced_baly/dataset_manifest.json'))
print(f'Total articles: {manifest[\"total_articles\"]}')
"
```

### Slow Processing

```bash
# Check GPU usage
nvidia-smi

# Verify not running on CPU
python -c "
import torch
from transformers import AutoModel
model = AutoModel.from_pretrained('Qwen/Qwen3-4B-Instruct')
print(f'Model device: {next(model.parameters()).device}')
"
```

---

## ✅ Validation Checklist

Before publishing results:

- [ ] All datasets balanced (verify with manifest files)
- [ ] Models loaded successfully (check logs for errors)
- [ ] GPU utilization confirmed (nvidia-smi)
- [ ] Results reproducible (same random seed, same dataset)
- [ ] Metrics match expected ranges (accuracy 60-80%)
- [ ] Discussion triggered for ~15-20% of articles
- [ ] Outlet evaluation generated all plots
- [ ] No articles skipped due to errors (check stats)

---

## 📚 Expected Timeline

| Task | Small Ensemble | Regular Ensemble |
|------|----------------|------------------|
| Setup (Step 1-2) | 30 min | 30 min |
| 100 articles | 2 hours | 4 hours |
| 500 articles | 10 hours | 20 hours |
| 1000 articles | 20 hours | 40 hours |
| Outlet eval (1300) | 26 hours | 52 hours |

*Times are approximate, vary by GPU*

---

## 📞 Getting Help

If you encounter issues:

1. **Check documentation**: README.md, QUICKSTART.md
2. **Search issues**: GitHub issues page
3. **Create issue**: Provide error logs, system info
4. **Email**: your.email@institution.edu

---

## 🎓 Citation

When using this system in research:

```bibtex
@article{your2025multiagent,
  title={Multi-Agent Collaborative Discussion for Political Bias Detection},
  author={Your Name and Colleagues},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

**Good luck with your experiments! 🚀**

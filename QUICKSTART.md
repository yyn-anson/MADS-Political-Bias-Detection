# Quick Start Guide

Get the Multi-Agent Bias Detection system running in 5 minutes!

---

## Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd multi_agent_bias_detection

# Install Python packages
pip install -r requirements.txt

# (Optional) Set HuggingFace token for gated models
export HF_TOKEN="your_huggingface_token"
```

**Requirements**: Python 3.8+, CUDA 11.7+, 12GB+ GPU RAM

---

## Step 2: Prepare Data (3 minutes)

### Option A: Download Pre-Balanced Dataset (Recommended)

```bash
# Download from [link to be added]
# Extract to data/balanced_datasets/

# Verify structure
ls data/balanced_datasets/balanced_baly/
# Should see: article files + dataset_manifest.json
```

### Option B: Create from Raw Data

```bash
# Create balanced Baly dataset (100 articles)
python tools/create_balanced_dataset.py --dataset baly --n-samples 100

# Verify creation
ls data/balanced_datasets/balanced_baly/
```

---

## Step 3: Run Your First Evaluation (1 minute)

```bash
# Small models (12GB GPU, faster)
python run_batches.py --model small --dataset baly --total 10

# Regular models (24GB GPU, more accurate)
python run_batches.py --model regular --dataset baly --total 10
```

**Expected output**:
```
Running small ensemble model on baly dataset
Total articles to process: 10
Articles per run: 8

Batch 1: Processing articles 0-8
[OK] Batch 0-8 completed successfully

AGGREGATING BATCH RESULTS
Accuracy: 0.7500 (75.00%)
Macro F1: 0.7234
```

---

## Step 4: View Results (30 seconds)

```bash
# Results are in outputs/ensemble_outputs_small/session_TIMESTAMP/

# View aggregated metrics
cat outputs/ensemble_outputs_small/session_*/aggregated_results.json

# View individual model performance
cat outputs/ensemble_outputs_small/session_*/individual_models/qwen3_results.json
```

---

## 🎉 Success!

You've successfully:
- ✅ Installed the system
- ✅ Prepared a balanced dataset
- ✅ Run multi-agent bias detection
- ✅ Generated evaluation metrics

---

## Next Steps

### Run Outlet-Level Evaluation

```bash
# Complete workflow: ensemble + visualization
python run_outlet_evaluation.py small

# View plots in outputs/ensemble_outputs_small/session_*/outlet_evaluation_*/visualizations/
```

### Process Larger Dataset

```bash
# Process 1000 articles in batches
python run_batches.py --model small --dataset baly --total 1000

# Monitor progress in real-time
```

### Try Different Datasets

```bash
# Budak dataset
python run_batches.py --model small --dataset budak

# Ad Fontes dataset
python run_batches.py --model small --dataset ad_fontes

# Custom outlet dataset
python run_batches.py --model small --dataset custom
```

---

## Common Commands Cheat Sheet

```bash
# Basic ensemble run
python run_batches.py --model small --dataset baly

# Resume interrupted batch
python run_batches.py --resume outputs/ensemble_outputs_small/session_20250120_143022

# Custom batch size (reduce if out of memory)
python run_batches.py --model small --dataset baly --batch-size 4

# Outlet evaluation (custom dataset)
python run_outlet_evaluation.py small --dataset-path data/balanced_datasets/custom_100_per_outlet

# Create balanced dataset
python tools/create_balanced_dataset.py --dataset baly --n-samples 1000
```

---

## Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
python run_batches.py --model small --dataset baly --batch-size 2
```

### Models not downloading?
```bash
# Check HuggingFace token
echo $HF_TOKEN

# Set token if missing
export HF_TOKEN="hf_..."
```

### Dataset not found?
```bash
# Verify path exists
ls data/balanced_datasets/balanced_baly/dataset_manifest.json

# Re-create if missing
python tools/create_balanced_dataset.py --dataset baly --n-samples 100
```

---

## Need More Help?

- **Detailed Guide**: See [docs/REPRODUCTION.md](docs/REPRODUCTION.md)
- **Architecture**: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Dataset Info**: See [data/README.md](data/README.md)
- **Full Documentation**: See [README.md](README.md)

---

**Happy bias detecting! 🚀**

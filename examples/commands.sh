#!/bin/bash
# Example Commands for Multi-Agent Bias Detection System
# Copy and paste these commands to run the system

# =============================================================================
# BASIC USAGE
# =============================================================================

# Run small ensemble on Baly dataset (100 articles)
python run_batches.py --model small --dataset baly --total 100

# Run regular ensemble on Baly dataset (100 articles)
python run_batches.py --model regular --dataset baly --total 100

# =============================================================================
# DIFFERENT DATASETS
# =============================================================================

# Budak dataset
python run_batches.py --model small --dataset budak --total 100

# Ad Fontes dataset
python run_batches.py --model small --dataset ad_fontes --total 100

# Custom outlet dataset (no ground truth, outlet-level analysis)
python run_batches.py --model small --dataset custom

# =============================================================================
# BATCH SIZE CONTROL
# =============================================================================

# Smaller batch size (if running out of memory)
python run_batches.py --model small --dataset baly --batch-size 4

# Larger batch size (if you have more GPU memory)
python run_batches.py --model small --dataset baly --batch-size 12

# =============================================================================
# RESUMING INTERRUPTED RUNS
# =============================================================================

# Resume from specific session
python run_batches.py --resume outputs/ensemble_outputs_small/session_20250120_143022

# =============================================================================
# OUTLET-LEVEL EVALUATION
# =============================================================================

# Complete workflow: run ensemble + outlet evaluation
python run_outlet_evaluation.py small

# Use regular models for outlet evaluation
python run_outlet_evaluation.py regular

# Custom dataset path
python run_outlet_evaluation.py small --dataset-path data/balanced_datasets/custom_100_per_outlet

# Skip ensemble, only run evaluation (use existing results)
python run_outlet_evaluation.py small --skip-ensemble --ensemble-dir outputs/ensemble_outputs_small/session_20250120_143022

# =============================================================================
# DATASET PREPARATION
# =============================================================================

# Create balanced Baly dataset (1000 articles)
python tools/create_balanced_dataset.py --dataset baly --n-samples 1000

# Create balanced Budak dataset (500 articles)
python tools/create_balanced_dataset.py --dataset budak --n-samples 500

# Create balanced Ad Fontes dataset
python tools/create_balanced_dataset.py --dataset ad_fontes --n-samples 800

# Create custom outlet dataset (100 articles per outlet)
python tools/create_balanced_dataset.py --dataset custom --samples-per-outlet 100

# =============================================================================
# TESTING & DEBUGGING
# =============================================================================

# Test with just 10 articles (quick sanity check)
python run_batches.py --model small --dataset baly --total 10

# Test dataset loading
python -c "
import json
from pathlib import Path
manifest = json.load(open('data/balanced_datasets/balanced_baly/dataset_manifest.json'))
print(f'Dataset: {manifest[\"dataset_type\"]}')
print(f'Total: {manifest[\"total_articles\"]}')
"

# =============================================================================
# PRODUCTION RUNS
# =============================================================================

# Large-scale evaluation (1000 articles, small models)
python run_batches.py --model small --dataset baly --total 1000

# High-accuracy evaluation (1000 articles, regular models)
python run_batches.py --model regular --dataset baly --total 1000

# Complete pipeline with visualization
python run_outlet_evaluation.py regular --dataset-path data/balanced_datasets/custom_100_per_outlet

# =============================================================================
# VIEWING RESULTS
# =============================================================================

# Find latest session
ls -lt outputs/ensemble_outputs_small/ | head -n 2

# View aggregated metrics
cat outputs/ensemble_outputs_small/session_*/aggregated_results.json | jq .overall_ensemble_metrics

# View individual model performance
cat outputs/ensemble_outputs_small/session_*/individual_models/qwen_results.json | jq .accuracy

# Count discussion articles
ls outputs/ensemble_outputs_small/session_*/collaborative_discussions/ | wc -l

# =============================================================================
# CLEANUP
# =============================================================================

# Remove specific session
rm -rf outputs/ensemble_outputs_small/session_20250120_143022/

# Keep only latest 3 sessions
cd outputs/ensemble_outputs_small/
ls -t | tail -n +4 | xargs rm -rf
cd ../..

# Clear model cache (will re-download)
rm -rf models/*

#!/usr/bin/env python
"""
Batch runner for ensemble models - processes articles in chunks to avoid memory issues.

Usage:
    python run_batches.py --model small --dataset baly
    python run_batches.py --model regular --dataset ad_fontes --total 500
"""

import subprocess
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Add this directory to path so src.* imports work
sys.path.insert(0, str(Path(__file__).parent))
from config import get_config

from src.utils.ground_truth import get_ground_truth_labels


def get_ground_truth_text(article_data: dict, dataset_type: str) -> tuple:
    """Extract ground truth as text label (left/center/right)."""
    # Skip ground truth for custom datasets
    if dataset_type == 'custom':
        return None, False
    
    true_bias, is_valid = get_ground_truth_labels(article_data, dataset_type)
    if not is_valid:
        return None, False
    
    # Convert numeric to text: 0=Left, 1=Center, 2=Right
    mapping = {0: "left", 1: "center", 2: "right"}
    ground_truth_text = mapping.get(true_bias)
    return ground_truth_text, True


def load_dataset_with_ground_truth(dataset_type: str) -> List[Tuple[Dict, str]]:
    """
    Load the dataset and extract ground truth labels.
    
    Returns:
        List of (article_data, filename) tuples
    """
    config = get_config()
    
    # Handle custom dataset specifically
    if dataset_type == 'custom':
        balanced_dir = Path(config['dirs']['balanced_datasets']) / 'custom_100_per_outlet'
    else:
        balanced_dir = Path(config['dirs']['balanced_datasets']) / f'balanced_{dataset_type}'
    
    if balanced_dir.exists() and (balanced_dir / 'dataset_manifest.json').exists():
        # Load balanced dataset
        print(f"Loading dataset from {balanced_dir}")
        
        with open(balanced_dir / 'dataset_manifest.json', 'r') as f:
            manifest = json.load(f)
        
        articles = []
        
        # Load articles directly from manifest (flat structure)
        for article_info in manifest.get('articles', []):
            filename = article_info['filename']
            article_file = balanced_dir / filename
            if article_file.exists():
                with open(article_file, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    articles.append((article_data, filename))
        
        print(f"Loaded {len(articles)} articles from dataset")
        return articles
    else:
        raise FileNotFoundError(f"Dataset for '{dataset_type}' not found at {balanced_dir}")


def aggregate_batch_results(output_dir: Path, dataset_type: str = 'baly') -> Dict:
    """
    Aggregate results from all batch files and calculate comprehensive metrics.
    Including individual model performance, consensus performance, and overall metrics.
    
    Args:
        output_dir: Directory containing batch result files
        dataset_type: Type of dataset for ground truth extraction
        
    Returns:
        Dictionary with aggregated results and comprehensive metrics
    """
    # Find all batch result files
    batch_files = sorted(output_dir.glob("batch_*_*_results.json"))
    
    if not batch_files:
        print("No batch result files found!")
        return None
    
    print(f"\n{'='*60}")
    print("AGGREGATING BATCH RESULTS")
    print(f"{'='*60}")
    print(f"Found {len(batch_files)} batch files")
    
    # Load the actual dataset to get ground truth
    try:
        articles = load_dataset_with_ground_truth(dataset_type)
    except FileNotFoundError as e:
        print(f"Warning: Could not load dataset for ground truth: {e}")
        articles = []
    
    # Create filename to article data mapping for ground truth extraction
    filename_to_article = {}
    filename_to_index = {}
    for idx, (article_data, filename) in enumerate(articles):
        filename_to_article[filename] = article_data
        filename_to_index[filename] = idx
    
    # Collect all results and individual model results
    all_results = []
    all_stats = {
        'total_articles': 0,
        'consensus_unanimous': 0,
        'consensus_majority': 0,
        'discussion_triggered': 0,
        'discussion_converged': 0,
        'articles_skipped': 0,
        'model_errors': {'llama32': 0, 'qwen3': 0, 'mistral': 0}
    }
    
    # Aggregate individual model results across all batches
    # We'll extract from batch results since individual files get overwritten
    # Initialize for all possible model names (both small and regular ensembles)
    individual_model_results = {
        'llama32': [],  # Small ensemble
        'qwen3': [],    # Small ensemble
        'mistral': [],  # Both ensembles
        'qwen': [],     # Regular ensemble
        'gptoss': []    # Regular ensemble
    }
    
    batch_metrics = []
    
    print(f"\n{'='*60}")
    print("1. LOADING BATCH DATA")
    print(f"{'='*60}")
    
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            batch_results = batch_data['results']
            
            # Extract batch range from filename
            batch_name = batch_file.stem  # e.g., "batch_0_30_results"
            parts = batch_name.split('_')
            if len(parts) >= 4:
                start_idx = int(parts[1])
                end_idx = int(parts[2])
                print(f"\nBatch {start_idx}-{end_idx}: {len(batch_results)} articles")
            else:
                print(f"\nBatch {batch_file.name}: {len(batch_results)} articles")
            
            # Calculate metrics for this batch using same logic as ensemble
            batch_y_true = []
            batch_y_pred = []
            
            for result in batch_results:
                # Add to overall results
                all_results.append(result)
                
                # Extract individual model predictions from batch results
                if 'individual_scores' in result and 'filename' in result:
                    filename = result['filename']
                    individual_scores = result['individual_scores']
                    
                    # Create individual model result entries
                    for model_name, model_data in individual_scores.items():
                        if model_name in individual_model_results:
                            model_result = {
                                'filename': filename,
                                'direction': model_data.get('direction', ''),
                                'score': model_data.get('score', 0),
                                'article_id': filename_to_index.get(filename, -1)
                            }
                            individual_model_results[model_name].append(model_result)
                
                # Get ground truth using the same function as ensemble
                filename = result.get('filename', '')
                if filename in filename_to_article:
                    article_data = filename_to_article[filename]
                    ground_truth_text, has_ground_truth = get_ground_truth_text(article_data, dataset_type)
                    
                    if has_ground_truth:
                        # Convert to numeric classes (same as ensemble)
                        true_class = {"left": 0, "center": 1, "right": 2}.get(ground_truth_text, -1)
                        
                        # Handle different result formats (same as ensemble)
                        if 'final_direction' in result:
                            pred_direction = result['final_direction']
                        elif 'direction' in result:
                            pred_direction = result['direction']
                        else:
                            continue
                        
                        pred_class = {"left": 0, "center": 1, "right": 2}.get(pred_direction.lower(), -1)
                        
                        if true_class != -1 and pred_class != -1:
                            batch_y_true.append(true_class)
                            batch_y_pred.append(pred_class)
            
            # Calculate batch metrics
            if batch_y_true:
                batch_acc = accuracy_score(batch_y_true, batch_y_pred)
                batch_f1 = f1_score(batch_y_true, batch_y_pred, average='macro')
                print(f"  Accuracy: {batch_acc:.4f} ({batch_acc*100:.2f}%)")
                print(f"  Macro F1: {batch_f1:.4f}")
                
                batch_metrics.append({
                    'batch': batch_file.name,
                    'accuracy': batch_acc,
                    'macro_f1': batch_f1,
                    'n_samples': len(batch_y_true)
                })
            
            # Aggregate statistics
            stats = batch_data.get('statistics', {})
            for key in all_stats:
                if key in stats:
                    if key == 'model_errors' and isinstance(stats[key], dict):
                        for model, count in stats[key].items():
                            if model in all_stats['model_errors']:
                                all_stats['model_errors'][model] += count
                    elif key != 'model_errors':
                        all_stats[key] += stats[key]
            
            # Note: We extract individual model results from batch results above
            # because individual model files get overwritten each batch
    
    print(f"\n{'='*60}")
    print("2. PER-BATCH PERFORMANCE")
    print(f"{'='*60}")
    
    # Display per-batch metrics that were calculated
    for metric in batch_metrics:
        print(f"\nBatch {metric['batch'].replace('batch_', '').replace('_results.json', '')}:")
        print(f"  Articles: {metric['n_samples']}")
        print(f"  Accuracy: {metric['accuracy']:.4f} ({metric['accuracy']*100:.2f}%)")
        print(f"  Macro F1: {metric['macro_f1']:.4f}")
    
    print(f"\n{'='*60}")
    print("3. INDIVIDUAL MODEL PERFORMANCE (ACROSS ALL BATCHES)")
    print(f"{'='*60}")
    
    # Store individual model metrics for JSON output
    individual_model_metrics = {}
    
    # Determine which models were used based on available results
    model_names = []
    for model_name in ['llama32', 'qwen3', 'mistral']:
        if model_name in individual_model_results and individual_model_results[model_name]:
            model_names.append(model_name)
            break  # Found small models
    if not model_names:
        for model_name in ['qwen', 'gptoss', 'mistral']:
            if model_name in individual_model_results and individual_model_results[model_name]:
                model_names.append(model_name)
    
    # Calculate individual model metrics
    if model_names:
        # Detect ensemble type
        if 'llama32' in model_names:
            model_names = ['llama32', 'qwen3', 'mistral']
            print("\nSmall Model Ensemble:")
        else:
            model_names = ['qwen', 'gptoss', 'mistral']
            print("\nRegular Model Ensemble:")
        
        for model_name in model_names:
            all_model_results = individual_model_results.get(model_name, [])  # Safe get with default
            if all_model_results:
                print(f"\n{model_name.upper()} Model:")
                print(f"Total articles: {len(all_model_results)}")
                
                # Calculate metrics from ALL accumulated results
                y_true = []
                y_pred = []
                
                for result in all_model_results:  # Use all_model_results, not model_results
                    if 'article_id' in result and result['article_id'] < len(articles):
                        article_data = articles[result['article_id']][0]
                        ground_truth_text, has_ground_truth = get_ground_truth_text(article_data, dataset_type)
                        
                        if has_ground_truth:
                            true_class = {"left": 0, "center": 1, "right": 2}.get(ground_truth_text, -1)
                            pred_class = {"left": 0, "center": 1, "right": 2}.get(result.get('direction', '').lower(), -1)
                            
                            if true_class != -1 and pred_class != -1:
                                y_true.append(true_class)
                                y_pred.append(pred_class)
                
                if y_true:
                    accuracy = accuracy_score(y_true, y_pred)
                    macro_f1 = f1_score(y_true, y_pred, average='macro')
                    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
                    
                    print(f"{model_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"Macro F1 Score: {macro_f1:.4f}")
                    print(f"Weighted F1 Score: {weighted_f1:.4f}")
                    
                    # Per-class metrics
                    class_report = classification_report(
                        y_true, y_pred,
                        labels=[0, 1, 2],
                        target_names=['Left', 'Center', 'Right'],
                        output_dict=True,
                        zero_division=0
                    )
                    
                    print("\nPer-Class Metrics:")
                    print("-" * 60)
                    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
                    print("-" * 60)
                    for class_name in ['Left', 'Center', 'Right']:
                        if class_name in class_report:
                            cr = class_report[class_name]
                            print(
                                f"{class_name:<10} "
                                f"{cr['precision']:<10.4f} "
                                f"{cr['recall']:<10.4f} "
                                f"{cr['f1-score']:<10.4f} "
                                f"{int(cr['support']):<10}"
                            )
                    
                    # Store metrics for this model
                    individual_model_metrics[model_name] = {
                        'total_articles': len(all_model_results),
                        'accuracy': accuracy,
                        'macro_f1': macro_f1,
                        'weighted_f1': weighted_f1,
                        'per_class': class_report,
                        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
                    }
    
    print(f"\n{'='*60}")
    print("4. OVERALL ENSEMBLE PERFORMANCE")
    print(f"{'='*60}")
    
    # Check for duplicate articles (shouldn't happen with proper batching)
    unique_filenames = set()
    duplicate_count = 0
    for result in all_results:
        filename = result.get('filename', '')
        if filename in unique_filenames:
            duplicate_count += 1
        else:
            unique_filenames.add(filename)
    
    print(f"Total articles processed: {len(all_results)}")
    if duplicate_count > 0:
        print(f"WARNING: Found {duplicate_count} duplicate articles!")
        print(f"Unique articles: {len(unique_filenames)}")
    
    # Calculate overall metrics using same logic as ensemble
    y_true = []
    y_pred = []
    
    for result in all_results:
        filename = result.get('filename', '')
        if filename in filename_to_article:
            article_data = filename_to_article[filename]
            ground_truth_text, has_ground_truth = get_ground_truth_text(article_data, dataset_type)
            
            if has_ground_truth:
                # Convert to numeric classes (same as ensemble)
                true_class = {"left": 0, "center": 1, "right": 2}.get(ground_truth_text, -1)
                
                # Handle different result formats (same as ensemble)
                if 'final_direction' in result:
                    pred_direction = result['final_direction']
                elif 'direction' in result:
                    pred_direction = result['direction']
                else:
                    continue
                
                pred_class = {"left": 0, "center": 1, "right": 2}.get(pred_direction.lower(), -1)
                
                if true_class != -1 and pred_class != -1:
                    y_true.append(true_class)
                    y_pred.append(pred_class)
    
    metrics = {}
    if y_true:
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        
        # Per-class metrics
        class_report = classification_report(
            y_true, y_pred,
            labels=[0, 1, 2],
            target_names=['Left', 'Center', 'Right'],
            output_dict=True
        )
        
        print("\nPer-Class Metrics:")
        print("-" * 60)
        print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        for class_name in ['Left', 'Center', 'Right']:
            if class_name in class_report:
                cr = class_report[class_name]
                print(
                    f"{class_name:<10} "
                    f"{cr['precision']:<10.4f} "
                    f"{cr['recall']:<10.4f} "
                    f"{cr['f1-score']:<10.4f} "
                    f"{int(cr['support']):<10}"
                )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print("       Pred:")
        print("       L   C   R")
        for i, label in enumerate(["L", "C", "R"]):
            if i < len(cm):
                print(f"True {label}: {cm[i]}")
        
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class': class_report,
            'confusion_matrix': cm.tolist()
        }
    
    # Calculate consensus-only metrics
    print(f"\n{'='*60}")
    print("5. CONSENSUS-ONLY PERFORMANCE (No Discussion)")
    print(f"{'='*60}")
    
    y_true_consensus = []
    y_pred_consensus = []
    
    for result in all_results:
        filename = result.get('filename', '')
        if filename not in filename_to_article:
            continue
        
        article_data = filename_to_article[filename]
        ground_truth_text, has_ground_truth = get_ground_truth_text(article_data, dataset_type)
        
        if not has_ground_truth:
            continue
        
        true_class = {"left": 0, "center": 1, "right": 2}.get(ground_truth_text, -1)
        if true_class == -1:
            continue
        
        # Determine consensus-only prediction
        if 'discussion_method' in result:
            # Article went through discussion - use pre-discussion or calculate average
            if 'pre_discussion_direction' in result:
                # Use saved pre-discussion direction
                pred_direction = result['pre_discussion_direction']
            elif 'individual_scores' in result:
                # Calculate average from individual scores (works for both ensembles)
                scores = [
                    model_data['score']
                    for model_data in result['individual_scores'].values()
                    if 'score' in model_data
                ]
                if len(scores) == 3:
                    avg_score = sum(scores) / 3
                    pred_direction = 'Left' if avg_score <= -1 else 'Right' if avg_score >= 1 else 'Center'
                else:
                    print(f"WARNING: Expected 3 individual scores for {filename}, got {len(scores)} - skipping")
                    continue
            else:
                print(f"WARNING: No pre-discussion data for {filename}")
                continue
        else:
            # Unanimous/majority consensus - use final direction
            pred_direction = result.get('final_direction', '')
        
        pred_class = {"left": 0, "center": 1, "right": 2}.get(pred_direction.lower(), -1)
        if pred_class != -1:
            y_true_consensus.append(true_class)
            y_pred_consensus.append(pred_class)
    
    consensus_only_metrics = {}
    if y_true_consensus:
        consensus_only_metrics = {
            'accuracy': accuracy_score(y_true_consensus, y_pred_consensus),
            'macro_f1': f1_score(y_true_consensus, y_pred_consensus, average='macro'),
            'weighted_f1': f1_score(y_true_consensus, y_pred_consensus, average='weighted'),
            'total_samples': len(y_true_consensus),
            'confusion_matrix': confusion_matrix(y_true_consensus, y_pred_consensus).tolist()
        }
        
        print(f"\nConsensus-Only Performance (Averaging/Majority Voting):")
        print(f"  Articles evaluated: {len(y_true_consensus)}")
        print(f"  Accuracy: {consensus_only_metrics['accuracy']:.4f} ({consensus_only_metrics['accuracy']*100:.2f}%)")
        print(f"  Macro F1: {consensus_only_metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {consensus_only_metrics['weighted_f1']:.4f}")
    else:
        print("No ground truth available for consensus-only evaluation")
    
    print(f"\n{'='*60}")
    print("6. CONSENSUS vs DISCUSSION BREAKDOWN")
    print(f"{'='*60}")
    
    # Separate consensus and discussion articles
    consensus_results = []
    discussion_results = []
    for result in all_results:
        if 'discussion_method' in result:
            discussion_results.append(result)
        else:
            consensus_results.append(result)
    
    consensus_metrics = {}
    if consensus_results:
        print(f"\nConsensus-only articles: {len(consensus_results)}")
        unanimous_count = sum(1 for r in consensus_results if r.get('consensus_type') == 'unanimous')
        majority_count = sum(1 for r in consensus_results if r.get('consensus_type') == 'majority')
        print(f"  - Unanimous agreement: {unanimous_count}")
        print(f"  - Majority agreement: {majority_count}")
        
        consensus_metrics = {
            'total': len(consensus_results),
            'unanimous': unanimous_count,
            'majority': majority_count
        }
    
    discussion_metrics = {}
    if discussion_results:
        print(f"\nDiscussion articles: {len(discussion_results)}")
        converged = sum(1 for r in discussion_results if r.get('convergence_achieved'))
        direction_changes = sum(1 for r in discussion_results if r.get('direction_changed'))
        print(f"  - Discussions converged: {converged} ({converged/len(discussion_results)*100:.1f}%)")
        print(f"  - Direction changed: {direction_changes} ({direction_changes/len(discussion_results)*100:.1f}%)")
        
        discussion_metrics = {
            'total': len(discussion_results),
            'converged': converged,
            'direction_changed': direction_changes
        }
    
    # Ablation study comparison
    print(f"\n{'='*60}")
    print("7. ABLATION STUDY: Impact of Collaborative Discussion")
    print(f"{'='*60}")
    
    if consensus_only_metrics and metrics:
        acc_diff = metrics['accuracy'] - consensus_only_metrics['accuracy']
        f1_diff = metrics['macro_f1'] - consensus_only_metrics['macro_f1']
        acc_pct = (acc_diff / consensus_only_metrics['accuracy'] * 100) if consensus_only_metrics['accuracy'] > 0 else 0
        
        print(f"\n{'Method':<35} {'Accuracy':<15} {'Macro F1':<10}")
        print("-" * 60)
        print(f"{'Consensus-Only (no discussion)':<35} {consensus_only_metrics['accuracy']:.4f} ({consensus_only_metrics['accuracy']*100:.1f}%) {consensus_only_metrics['macro_f1']:.4f}")
        print(f"{'Full Ensemble (with discussion)':<35} {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%) {metrics['macro_f1']:.4f}")
        print("-" * 60)
        print(f"{'Improvement':<35} {acc_diff:+.4f} ({acc_pct:+.1f}%) {f1_diff:+.4f}")
        
        # Show breakdown
        unanimous = sum(1 for r in all_results if r.get('consensus_type') == 'unanimous')
        majority = sum(1 for r in all_results if r.get('consensus_type') == 'majority')
        discussion = len(discussion_results)
        
        print(f"\nArticle Distribution:")
        print(f"  Unanimous: {unanimous} ({unanimous/len(all_results)*100:.1f}%)")
        print(f"  Majority: {majority} ({majority/len(all_results)*100:.1f}%)")
        print(f"  Discussion: {discussion} ({discussion/len(all_results)*100:.1f}%)")
    else:
        print("Insufficient data for ablation study comparison")
    
    # Save comprehensive aggregated results
    aggregated_file = output_dir / "aggregated_results.json"
    with open(aggregated_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset_type': dataset_type,
            'total_batches': len(batch_files),
            'total_articles': len(all_results),
            'statistics': all_stats,
            'batch_metrics': batch_metrics,
            'individual_model_metrics': individual_model_metrics if individual_model_metrics else {},
            'consensus_only_metrics': consensus_only_metrics if consensus_only_metrics else {},
            'overall_ensemble_metrics': metrics if metrics else {},
            'ablation_study': {
                'consensus_accuracy': consensus_only_metrics.get('accuracy') if consensus_only_metrics else None,
                'consensus_f1': consensus_only_metrics.get('macro_f1') if consensus_only_metrics else None,
                'ensemble_accuracy': metrics.get('accuracy') if metrics else None,
                'ensemble_f1': metrics.get('macro_f1') if metrics else None,
                'accuracy_improvement': acc_diff if 'acc_diff' in locals() else None,
                'f1_improvement': f1_diff if 'f1_diff' in locals() else None,
                'accuracy_improvement_pct': acc_pct if 'acc_pct' in locals() else None
            } if consensus_only_metrics and metrics else {},
            'consensus_breakdown': consensus_metrics if consensus_metrics else {},
            'discussion_breakdown': discussion_metrics if discussion_metrics else {},
            'consensus_articles': len(consensus_results) if 'consensus_results' in locals() else 0,
            'discussion_articles': len(discussion_results) if 'discussion_results' in locals() else 0,
            'results': all_results
        }, f, indent=2)
    
    print(f"\nAggregated results saved to: {aggregated_file}")
    
    return metrics


def detect_session_config(session_path: str) -> tuple:
    """
    Auto-detect configuration from an existing session directory.
    
    Args:
        session_path: Path to existing session directory
        
    Returns:
        Tuple of (dataset, model_type, batch_size, start_index, output_dir)
    """
    session_dir = Path(session_path)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_path}")
    
    # Read dataset type from article_list.json
    article_list_file = session_dir / 'article_list.json'
    if article_list_file.exists():
        with open(article_list_file, 'r') as f:
            article_list = json.load(f)
            dataset = article_list.get('dataset_type', 'unknown')
    else:
        # Fallback: try to infer from batch files
        dataset = 'unknown'
    
    # Find all batch files and analyze configuration
    batch_files = sorted(session_dir.glob('batch_*_*_results.json'))
    if not batch_files:
        raise ValueError(f"No batch files found in session: {session_path}")
    
    # Read first batch file to detect model type and batch size
    with open(batch_files[0], 'r') as f:
        first_batch = json.load(f)
    
    # Detect model type from models_used
    models_used = first_batch['session_info']['models_used']
    if any('Llama-3.2-3B' in m for m in models_used):
        model_type = 'small'
    elif any('Qwen3-14B' in m or 'gpt-oss-20b' in m for m in models_used):
        model_type = 'regular'
    else:
        # Fallback: check output directory path
        if 'ensemble_outputs_small' in str(session_dir):
            model_type = 'small'
        else:
            model_type = 'regular'
    
    # Detect batch size
    batch_size = first_batch['statistics']['total_articles']
    
    # Find last processed article index
    batch_ranges = []
    for bf in batch_files:
        parts = bf.stem.split('_')  # e.g., 'batch_0_2_results' -> ['batch', '0', '2', 'results']
        if len(parts) >= 4:
            start = int(parts[1])
            end = int(parts[2])
            batch_ranges.append((start, end))
    
    # Check for gaps
    batch_ranges.sort()
    for i in range(1, len(batch_ranges)):
        if batch_ranges[i][0] != batch_ranges[i-1][1]:
            print(f"WARNING: Gap detected in batch sequence between {batch_ranges[i-1]} and {batch_ranges[i]}")
    
    # Get the last processed index
    if batch_ranges:
        start_index = batch_ranges[-1][1]  # Continue from where we left off
    else:
        start_index = 0
    
    print(f"\nDetected session configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Model type: {model_type}")
    print(f"  Batch size: {batch_size}")
    print(f"  Last processed: article {start_index}")
    print(f"  Batches found: {len(batch_files)}")
    
    return dataset, model_type, batch_size, start_index, str(session_dir)


def run_ensemble_batches(model_type='small', dataset='baly', total_articles=None, articles_per_run=None, resume_session=None):
    """
    Run ensemble model in batches.
    
    Args:
        model_type: 'small' for small models, 'regular' for large models
        dataset: Dataset to use ('baly', 'budak', 'ad_fontes', 'custom')
        total_articles: Total number of articles to process (None = auto-detect)
        articles_per_run: Number of articles per batch (None = use model-specific defaults)
        resume_session: Path to existing session directory to resume (overrides other args)
    """
    # If resuming, detect configuration from existing session
    start_index = 0
    if resume_session:
        print(f"Resuming session from: {resume_session}")
        dataset, model_type, detected_batch_size, start_index, output_dir = detect_session_config(resume_session)
        
        # Use detected batch size unless explicitly overridden
        if articles_per_run is None:
            articles_per_run = detected_batch_size
        
        # Convert output_dir to Path
        output_dir = Path(output_dir)
        print(f"\nResuming from article {start_index} with batch size {articles_per_run}")
    else:
        output_dir = None  # Will be created later
    
    # Auto-detect dataset size if not specified
    if total_articles is None:
        try:
            articles = load_dataset_with_ground_truth(dataset)
            total_articles = len(articles)
            print(f"Auto-detected dataset size: {total_articles} articles")
        except FileNotFoundError as e:
            print(f"Error: Could not auto-detect dataset size: {e}")
            return
    
    # Set appropriate batch size based on model type if not specified
    if articles_per_run is None:
        if model_type == 'small':
            articles_per_run = 8  # Small models can handle 8 articles
        else:
            articles_per_run = 3  # Regular models limited to 3 articles
        print(f"Using batch size: {articles_per_run} articles per batch")
    
    # Select the appropriate ensemble script
    if model_type == 'small':
        if dataset == 'custom':
            script_path = "src/ensemble/ensemble_small_custom.py"
        else:
            script_path = "src/ensemble/ensemble_small.py"
    else:
        if dataset == 'custom':
            script_path = "src/ensemble/ensemble_regular_custom.py"
        else:
            script_path = "src/ensemble/ensemble_regular.py"
    
    # Create output directory only if not resuming
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_type == 'small':
            output_dir = Path("ensemble_outputs_small") / f"session_{timestamp}"
        else:
            output_dir = Path("ensemble_outputs") / f"session_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running {model_type} ensemble model on {dataset} dataset")
    if resume_session:
        print(f"Resuming from article: {start_index}")
        print(f"Articles remaining: {total_articles - start_index}")
    print(f"Total articles to process: {total_articles}")
    print(f"Articles per run: {articles_per_run}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    successful_batches = 0
    failed_batches = []
    
    for start in range(start_index, total_articles, articles_per_run):
        end = min(start + articles_per_run, total_articles)
        
        print(f"\nBatch {start//articles_per_run + 1}: Processing articles {start}-{end}")
        print("-" * 40)
        
        # Build command
        cmd = [
            sys.executable,
            script_path,
            "--dataset", dataset,
            "--batch-start", str(start),
            "--batch-end", str(end),
            "--output-dir", str(output_dir),
            "--skip-evaluation"  # Skip evaluation during batches
        ]
        
        try:
            # Run the batch with timeout to prevent hanging
            result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)  # 2 hour timeout
            
            if result.returncode == 0:
                print(f"[OK] Batch {start}-{end} completed successfully")
                successful_batches += 1
            else:
                print(f"[FAILED] Batch {start}-{end} failed with return code {result.returncode}")
                failed_batches.append((start, end))
                print(f"[INFO] Automatically continuing to next batch...")
                continue
        
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Batch {start}-{end} exceeded 2 hour limit - skipping")
            failed_batches.append((start, end))
            print(f"[INFO] Automatically continuing to next batch...")
            continue
            
        except KeyboardInterrupt:
            print("\n\nProcess interrupted by user")
            break
            
        except Exception as e:
            print(f"[ERROR] Error processing batch {start}-{end}: {e}")
            failed_batches.append((start, end))
            print(f"[INFO] Automatically continuing to next batch...")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Successful batches: {successful_batches}")
    print(f"Failed batches: {len(failed_batches)}")
    
    if failed_batches:
        print("\nFailed batch ranges:")
        for start, end in failed_batches:
            print(f"  - Articles {start}-{end}")
    
    # Aggregate results and calculate overall metrics
    if successful_batches > 0:
        print("\n" + "=" * 60)
        print("AGGREGATING BATCH RESULTS")
        print("=" * 60)
        
        # Get actual article count from aggregation results
        metrics = aggregate_batch_results(output_dir, dataset)
        
        # Count actual articles processed from batch files
        batch_files = sorted(output_dir.glob("batch_*_*_results.json"))
        total_processed = 0
        for batch_file in batch_files:
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
                total_processed += len(batch_data['results'])
        
        print(f"\nTotal articles actually processed: {total_processed}")
    else:
        print(f"\nTotal articles processed: 0")
    
    print("\nBatch processing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run ensemble models in batches to avoid memory issues'
    )
    
    # Add resume argument
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to existing session directory to resume (e.g., ensemble_outputs/session_20250911_120132)')
    
    # Original arguments (ignored when using --resume)
    parser.add_argument('--model', type=str, default='small',
                       choices=['small', 'regular'],
                       help='Model type to run (small or regular) - ignored when using --resume')
    parser.add_argument('--dataset', type=str, default='baly',
                       choices=['baly', 'budak', 'ad_fontes', 'custom'],
                       help='Dataset to process - ignored when using --resume')
    parser.add_argument('--total', type=int, default=None,
                       help='Total number of articles to process (default: auto-detect from dataset)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Number of articles per batch (default: 8 for small, 3 for regular)')
    
    args = parser.parse_args()
    
    if args.resume:
        # When resuming, most arguments are auto-detected
        run_ensemble_batches(
            resume_session=args.resume,
            total_articles=args.total,  # Can override total if needed
            articles_per_run=args.batch_size  # Can override batch size if needed
        )
    else:
        # Normal run with specified parameters
        run_ensemble_batches(
            model_type=args.model,
            dataset=args.dataset,
            total_articles=args.total,
            articles_per_run=args.batch_size
        )
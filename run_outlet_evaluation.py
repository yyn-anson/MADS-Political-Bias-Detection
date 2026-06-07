#!/usr/bin/env python
"""
Complete workflow for outlet-level evaluation of media bias detection.
Automatically detects dataset size and runs appropriate batches.

Usage:
    python run_outlet_evaluation.py              # Use small models
    python run_outlet_evaluation.py regular      # Use regular models
    python run_outlet_evaluation.py --skip-ensemble  # Skip ensemble, only run evaluation

Author: Media Bias Detection Team
Date: 2025
"""

import subprocess
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_dataset_size(dataset_path):
    """
    Get total article count from dataset manifest with validation.
    
    Args:
        dataset_path: Path to the custom dataset directory
        
    Returns:
        Number of articles in the dataset
        
    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If manifest is invalid
    """
    manifest_path = dataset_path / "dataset_manifest.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
        articles = manifest.get('articles')
        if articles is None:
            raise ValueError("No 'articles' field in manifest")
            
        article_count = len(articles)
        if article_count == 0:
            raise ValueError("Dataset contains no articles")
            
        # Also extract outlet information
        outlets = set()
        for article in articles:
            outlet = article.get('outlet')
            if outlet:
                outlets.add(outlet)
        
        print(f"\nDataset Information:")
        print(f"  Total articles: {article_count}")
        print(f"  Number of outlets: {len(outlets)}")
        print(f"  Outlets: {', '.join(sorted(outlets))}")
        
        return article_count
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in manifest: {e}")


def run_ensemble(model_type, dataset_path):
    """
    Run the ensemble model on the custom dataset.
    
    Args:
        model_type: 'small' or 'regular'
        dataset_path: Path to custom dataset
        
    Returns:
        Path to the ensemble output directory
    """
    print(f"\n{'='*60}")
    print(f"STEP 1: Running {model_type} ensemble on custom dataset")
    print(f"{'='*60}")
    
    # Get dataset size
    try:
        total_articles = get_dataset_size(dataset_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to get dataset size: {e}")
        sys.exit(1)
    
    # Calculate expected batch count
    batch_size = 8 if model_type == 'small' else 2
    num_batches = (total_articles + batch_size - 1) // batch_size
    
    print(f"\nExpected processing:")
    print(f"  Model type: {model_type}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Estimated time: {num_batches * 10}-{num_batches * 15} minutes")
    
    # Build command
    cmd = [
        sys.executable,
        "run_batches.py",
        "--model", model_type,
        "--dataset", "custom"
        # total_articles will be auto-detected
        # batch_size will use defaults (8 for small, 2 for regular)
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    
    # Run the ensemble
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output for debugging
        if result.stdout:
            print("\nEnsemble output:")
            print(result.stdout)
        
        if result.stderr:
            print("\nEnsemble errors/warnings:")
            print(result.stderr)
        
        if result.returncode != 0:
            logger.error(f"Ensemble failed with return code {result.returncode}")
            sys.exit(1)
            
    except subprocess.TimeoutExpired:
        logger.error("Ensemble timed out")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to run ensemble: {e}")
        sys.exit(1)
    
    # Find the output directory
    if model_type == 'small':
        output_base = Path("ensemble_outputs_small")
    else:
        output_base = Path("ensemble_outputs")
    
    if not output_base.exists():
        logger.error(f"Output directory not found: {output_base}")
        sys.exit(1)
    
    # Get the latest session directory
    session_dirs = sorted(output_base.glob("session_*"))
    if not session_dirs:
        logger.error(f"No session directories found in {output_base}")
        sys.exit(1)
    
    ensemble_dir = session_dirs[-1]
    
    # Verify results exist
    required_files = [
        "aggregated_results.json",
        "session_summary.json",
        # Accept either one
    ]
    
    has_results = False
    for filename in required_files:
        if (ensemble_dir / filename).exists():
            has_results = True
            print(f"Found results file: {filename}")
            break
    
    # Also check for batch files
    if not has_results:
        batch_files = list(ensemble_dir.glob("batch_*_*_results.json"))
        if batch_files:
            has_results = True
            print(f"Found {len(batch_files)} batch result files")
    
    if not has_results:
        logger.error(f"No result files found in {ensemble_dir}")
        sys.exit(1)
    
    print(f"\nEnsemble completed successfully!")
    print(f"Results directory: {ensemble_dir}")
    
    return ensemble_dir


def run_outlet_evaluation(ensemble_dir, dataset_path):
    """
    Run the outlet-level evaluation.
    
    Args:
        ensemble_dir: Path to ensemble results
        dataset_path: Path to custom dataset
    """
    print(f"\n{'='*60}")
    print("STEP 2: Running outlet-level evaluation")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        sys.executable,
        "LLM/multi_agent/outlet_evaluation.py",
        "--ensemble-dir", str(ensemble_dir),
        "--custom-dataset", str(dataset_path)
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("\nEvaluation output:")
            print(result.stdout)
        
        if result.stderr:
            print("\nEvaluation errors/warnings:")
            print(result.stderr)
        
        if result.returncode != 0:
            logger.error(f"Evaluation failed with return code {result.returncode}")
            # Don't exit here - evaluation might have partial results
            
    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}")
        sys.exit(1)
    
    # Find evaluation output
    eval_dirs = sorted(ensemble_dir.glob("outlet_evaluation_*"))
    if eval_dirs:
        eval_dir = eval_dirs[-1]
        print(f"\nEvaluation completed!")
        print(f"Results directory: {eval_dir}")
        
        # Check for key output files
        expected_files = [
            "outlet_evaluation_report.json",
            "visualizations/violin_plot_raw_scores.pdf",
            "visualizations/confusion_matrix.pdf",
            "visualizations/per_class_performance.pdf",
            "visualizations/outlet_comparison.pdf"
        ]
        
        print("\nGenerated files:")
        for filename in expected_files:
            filepath = eval_dir / filename
            if filepath.exists():
                print(f"  [OK] {filename}")
            else:
                print(f"  [MISSING] {filename}")
    else:
        print("\nWarning: Could not find evaluation output directory")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run complete outlet-level evaluation workflow'
    )
    
    parser.add_argument(
        'model_type',
        nargs='?',
        default='small',
        choices=['small', 'regular'],
        help='Model type to use (default: small)'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='data/balanced_datasets/custom_100_per_outlet',
        help='Path to custom dataset (default: data/balanced_datasets/custom_100_per_outlet)'
    )
    
    parser.add_argument(
        '--skip-ensemble',
        action='store_true',
        help='Skip ensemble run and use existing results'
    )
    
    parser.add_argument(
        '--ensemble-dir',
        type=str,
        help='Path to existing ensemble results (required with --skip-ensemble)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("MEDIA OUTLET-LEVEL EVALUATION WORKFLOW")
    print(f"{'='*60}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Dataset: {dataset_path}")
    print(f"Model type: {args.model_type}")
    
    try:
        if args.skip_ensemble:
            # Use existing ensemble results
            if not args.ensemble_dir:
                # Try to find the latest ensemble directory
                if args.model_type == 'small':
                    output_base = Path("ensemble_outputs_small")
                else:
                    output_base = Path("ensemble_outputs")
                
                if output_base.exists():
                    session_dirs = sorted(output_base.glob("session_*"))
                    if session_dirs:
                        ensemble_dir = session_dirs[-1]
                        print(f"\nUsing latest ensemble results: {ensemble_dir}")
                    else:
                        logger.error("No ensemble results found. Please run ensemble first or specify --ensemble-dir")
                        sys.exit(1)
                else:
                    logger.error(f"No ensemble output directory found: {output_base}")
                    sys.exit(1)
            else:
                ensemble_dir = Path(args.ensemble_dir)
                if not ensemble_dir.exists():
                    logger.error(f"Ensemble directory not found: {ensemble_dir}")
                    sys.exit(1)
                print(f"\nUsing specified ensemble results: {ensemble_dir}")
        else:
            # Run the ensemble
            ensemble_dir = run_ensemble(args.model_type, dataset_path)
        
        # Run outlet evaluation
        run_outlet_evaluation(ensemble_dir, dataset_path)
        
        print(f"\n{'='*60}")
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
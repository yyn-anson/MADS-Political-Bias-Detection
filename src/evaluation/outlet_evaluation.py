"""
Media Outlet Level Evaluation System
=====================================
Evaluates LLM ensemble performance at the media outlet level by:
1. Aggregating article-level predictions per outlet
2. Comparing against AllSides ground truth (3-class system)
3. Calculating metrics and generating visualizations

Scoring note
------------
The ensemble models output a raw bias score on a -3 to +3 integer scale
(same prompts and parsing as the standard batch evaluation). This module
uses those raw scores in two ways:

* Violin / beeswarm plots  - raw -3..+3 scores are plotted directly,
  showing the full per-article distribution for each outlet.
* Accuracy / F1 metrics    - scores are collapsed to 3 classes before
  comparison with AllSides ground truth:
      score <= -1  ->  Left
      -1 < score < 1  ->  Center
      score >= 1  ->  Right

No separate prompt or output-parser is needed for outlet evaluation;
the only difference from batch evaluation is that results are aggregated
and visualised at the outlet level rather than the article level.

Author: Media Bias Detection Team
Date: 2025
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_config

# Define the 13 media outlets we're evaluating (matching custom dataset)
MEDIA_OUTLETS = {
    "CNN": "cnn.com",
    "BBC": "bbc.com",
    "Fox News": "foxnews.com",
    "Breitbart": "breitbart.com",
    "The Guardian": "theguardian.com",
    "New York Times": "nytimes.com",
    "TIME": "time.com",
    "New York Post": "nypost.com",
    "MSNBC": "msnbc.com",
    "The Nation": "thenation.com",
    "Washington Examiner": "washingtonexaminer.com",
    "Newsweek": "newsweek.com",
    "Forbes": "forbes.com"
}


class MediaOutletEvaluator:
    """
    Evaluates media bias detection at the outlet level using 3-class system.
    """
    
    def __init__(self, config: Dict = None, ensemble_results_dir: str = None, 
                 custom_dataset_dir: str = None):
        """
        Initialize the outlet evaluator.
        
        Args:
            config: Configuration dictionary
            ensemble_results_dir: Path to ensemble output directory
            custom_dataset_dir: Path to custom balanced dataset
        """
        if config is None:
            config = get_config()
        self.config = config
        self.ensemble_results_dir = ensemble_results_dir
        self.custom_dataset_dir = custom_dataset_dir
        
        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("ensemble_outputs") / f"outlet_evaluation_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        self.details_dir = self.output_dir / "detailed_predictions"
        self.details_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.article_predictions = {}  # outlet -> list of predictions
        self.outlet_results = {}  # outlet -> aggregated results
        self.ground_truth = {}  # outlet -> ground truth class
        
        print(f"Initialized MediaOutletEvaluator")
        print(f"Output directory: {self.output_dir}")
    
    def convert_allsides_to_3class(self, category: str) -> str:
        """
        Convert AllSides 5-category rating to 3-class system.
        
        Args:
            category: AllSides category (Left, Lean Left, Center, Lean Right, Right)
            
        Returns:
            3-class label (Left, Center, Right)
        """
        if category in ['Left', 'Lean Left']:
            return 'Left'
        elif category == 'Center':
            return 'Center'
        elif category in ['Lean Right', 'Right']:
            return 'Right'
        else:
            return 'Unknown'
    
    def score_to_3class(self, score: float) -> str:
        """
        Convert numerical bias score to 3-class system.
        
        Args:
            score: Numerical bias score
            
        Returns:
            3-class label (Left, Center, Right)
        """
        if score < -1.0:
            return 'Left'
        elif -1.0 <= score <= 1.0:
            return 'Center'
        else:  # score > 1.0
            return 'Right'
    
    def load_allsides_ground_truth(self) -> Dict[str, str]:
        """
        Load and process AllSides ground truth ratings.
        
        Returns:
            Dictionary mapping outlet names to 3-class labels
        """
        allsides_path = Path(self.config['files']['allsides_ratings'])
        
        if not allsides_path.exists():
            raise FileNotFoundError(f"AllSides ratings not found: {allsides_path}")
        
        print(f"\nLoading AllSides ground truth from: {allsides_path}")
        
        # Load CSV
        df = pd.read_csv(allsides_path)
        
        # Column names
        source_col = 'allsides_media_bias_ratings/publication/source_name'
        rating_col = 'allsides_media_bias_ratings/publication/media_bias_rating'
        url_col = 'allsides_media_bias_ratings/publication/source_url'
        
        ground_truth = {}
        
        # Process each outlet we're interested in
        for outlet_name, domain in MEDIA_OUTLETS.items():
            found = False
            
            # Search for matching entries
            for _, row in df.iterrows():
                source_url = str(row[url_col]).lower() if pd.notna(row[url_col]) else ''
                source_name = str(row[source_col]).lower() if pd.notna(row[source_col]) else ''
                
                # Check if domain matches
                if domain in source_url or outlet_name.lower() in source_name:
                    rating = row[rating_col].strip()
                    
                    # Skip mixed ratings
                    if rating == 'Mixed':
                        continue
                    
                    # Convert to 3-class
                    three_class = self.convert_allsides_to_3class(rating)
                    ground_truth[outlet_name] = three_class
                    found = True
                    print(f"  {outlet_name}: {rating} -> {three_class}")
                    break
            
            if not found:
                print(f"  WARNING: No AllSides rating found for {outlet_name}")
        
        self.ground_truth = ground_truth
        print(f"\nLoaded ground truth for {len(ground_truth)} outlets")

        return ground_truth

    def _save_figure(self, fig, base_path: Path, description: str = None):
        """
        Save figure in both PDF and SVG formats.

        Args:
            fig: Matplotlib figure object
            base_path: Path without extension (e.g., 'violin_plot_professional')
            description: Description for logging (optional)
        """
        # Save PDF
        pdf_path = base_path.parent / f"{base_path.stem}.pdf"
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')

        # Save SVG
        svg_path = base_path.parent / f"{base_path.stem}.svg"
        fig.savefig(svg_path, dpi=300, bbox_inches='tight')

        if description:
            print(f"  Saved {description} to:")
        else:
            print(f"  Saved to:")
        print(f"    PDF: {pdf_path}")
        print(f"    SVG: {svg_path}")

    def load_ensemble_results(self) -> List[Dict]:
        """
        Load and validate article-level predictions from ensemble output.
        
        Returns:
            List of validated ensemble results
        
        Raises:
            ValueError: If results are missing or invalid
            FileNotFoundError: If result files don't exist
        """
        if not self.ensemble_results_dir:
            raise ValueError("No ensemble results directory specified")
        
        ensemble_dir = Path(self.ensemble_results_dir)
        
        if not ensemble_dir.exists():
            raise FileNotFoundError(f"Ensemble results not found: {ensemble_dir}")
        
        print(f"\nLoading ensemble results from: {ensemble_dir}")
        
        # Try multiple possible result file locations
        results = None
        
        # First try aggregated_results.json (from batch processing)
        aggregated_path = ensemble_dir / "aggregated_results.json"
        if aggregated_path.exists():
            with open(aggregated_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results = data.get('results')
                if results is None:
                    raise ValueError(f"No 'results' field in {aggregated_path}")
                print(f"Loaded from aggregated_results.json")
        
        # Then try session_summary.json
        elif (ensemble_dir / "session_summary.json").exists():
            summary_path = ensemble_dir / "session_summary.json"
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                results = summary.get('results')
                if results is None:
                    raise ValueError(f"No 'results' field in {summary_path}")
                print(f"Loaded from session_summary.json")
        
        # Finally try batch result files
        else:
            batch_files = sorted(ensemble_dir.glob("batch_*_*_results.json"))
            if batch_files:
                results = []
                for batch_file in batch_files:
                    with open(batch_file, 'r', encoding='utf-8') as f:
                        batch_data = json.load(f)
                        batch_results = batch_data.get('results')
                        if batch_results:
                            results.extend(batch_results)
                print(f"Loaded from {len(batch_files)} batch files")
            else:
                raise FileNotFoundError(f"No result files found in {ensemble_dir}")
        
        if not results:
            raise ValueError("No results found in ensemble output")
        
        print(f"Loaded {len(results)} article predictions")
        
        # Validate results have required fields
        validated_results = []
        skipped_count = 0
        error_details = []
        
        for i, result in enumerate(results):
            try:
                # Validate required fields exist (no defaults!)
                filename = result.get('filename')
                if not filename:
                    raise KeyError(f"Result {i}: Missing 'filename'")
                
                final_score = result.get('final_score')
                if final_score is None:
                    raise KeyError(f"Result {i} ({filename}): Missing 'final_score'")
                
                final_direction = result.get('final_direction')
                if not final_direction:
                    raise KeyError(f"Result {i} ({filename}): Missing 'final_direction'")
                
                # Validate score is numeric and in valid range
                if not isinstance(final_score, (int, float)):
                    raise ValueError(f"Result {i} ({filename}): Invalid score type: {type(final_score)}")
                
                if not -3 <= final_score <= 3:
                    raise ValueError(f"Result {i} ({filename}): Score {final_score} out of range [-3, 3]")
                
                # Validate direction is valid
                valid_directions = ['Left', 'Center', 'Right', 'left', 'center', 'right']
                if final_direction not in valid_directions:
                    raise ValueError(f"Result {i} ({filename}): Invalid direction '{final_direction}'")
                
                validated_results.append(result)
                
            except (KeyError, ValueError) as e:
                error_details.append(str(e))
                skipped_count += 1
                logger.warning(f"Skipping invalid result: {e}")
                continue
        
        if skipped_count > 0:
            print(f"\nValidation Summary:")
            print(f"  Valid results: {len(validated_results)}")
            print(f"  Skipped (errors): {skipped_count}")
            
            if error_details:
                print(f"\nFirst 5 errors:")
                for error in error_details[:5]:
                    print(f"  - {error}")
        
        if not validated_results:
            raise ValueError("No valid results after validation")
        
        return validated_results
    
    def load_custom_dataset_results(self) -> Dict[str, List[Dict]]:
        """
        Map ensemble results to outlets using custom dataset manifest.
        Uses index-based mapping with strict validation.
        
        Returns:
            Dictionary mapping outlets to list of article predictions
        
        Raises:
            ValueError: If required fields are missing
            FileNotFoundError: If files don't exist
        """
        if not self.custom_dataset_dir:
            raise ValueError("No custom dataset directory specified")
        
        dataset_dir = Path(self.custom_dataset_dir)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Custom dataset not found: {dataset_dir}")
        
        print(f"\nLoading custom dataset from: {dataset_dir}")
        
        # Load manifest to get outlet mappings
        manifest_path = dataset_dir / "dataset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Create index to outlet mapping from manifest with validation
        index_to_outlet = {}
        manifest_errors = []
        
        for article_info in manifest.get('articles', []):
            try:
                index = article_info.get('index')
                outlet = article_info.get('outlet')
                filename = article_info.get('filename')
                
                if index is None:
                    raise KeyError(f"Missing 'index' in article: {article_info}")
                if not outlet:
                    raise KeyError(f"Missing 'outlet' for article {index}")
                if not filename:
                    raise KeyError(f"Missing 'filename' for article {index}")
                
                index_to_outlet[index] = outlet
                
            except KeyError as e:
                manifest_errors.append(str(e))
                continue
        
        if manifest_errors:
            print(f"\nManifest validation errors: {len(manifest_errors)}")
            for error in manifest_errors[:5]:
                print(f"  - {error}")
        
        print(f"Loaded manifest with {len(index_to_outlet)} valid article mappings")
        
        # Load ensemble results
        if not self.ensemble_results_dir:
            raise ValueError("Ensemble results directory required for custom dataset evaluation")
        
        ensemble_results = self.load_ensemble_results()
        
        # Map results to outlets using index-based mapping
        article_predictions = defaultdict(list)
        skipped_count = 0
        mapping_errors = []
        
        for result in ensemble_results:
            try:
                filename = result.get('filename')
                if not filename:
                    raise KeyError("Missing filename in result")
                
                # Extract index from filename (e.g., "0000_xxx.json" -> 0)
                if not filename[:4].isdigit():
                    raise ValueError(f"Invalid filename format '{filename}' (expected NNNN_*.json)")
                
                article_index = int(filename[:4])
                
                # Map to outlet using index
                if article_index not in index_to_outlet:
                    raise KeyError(f"Article index {article_index} not found in manifest")
                
                outlet = index_to_outlet[article_index]
                
                # Extract and validate prediction data (no defaults!)
                final_score = result.get('final_score')
                if final_score is None:
                    raise KeyError(f"Missing 'final_score' for {filename}")
                
                final_direction = result.get('final_direction')
                if not final_direction:
                    raise KeyError(f"Missing 'final_direction' for {filename}")
                
                consensus_type = result.get('consensus_type')
                if not consensus_type:
                    # consensus_type is optional for some older results
                    consensus_type = 'unknown'
                
                # Create prediction entry
                prediction = {
                    'filename': filename,
                    'score': final_score,
                    'direction': final_direction.capitalize(),  # Normalize to Title case
                    'consensus_type': consensus_type,
                    'article_index': article_index
                }
                
                article_predictions[outlet].append(prediction)
                
            except (KeyError, ValueError) as e:
                mapping_errors.append(f"{filename if 'filename' in locals() else 'unknown'}: {e}")
                skipped_count += 1
                continue
        
        # Convert to regular dict
        self.article_predictions = dict(article_predictions)
        
        # Log mapping summary
        total_mapped = sum(len(preds) for preds in self.article_predictions.values())
        print(f"\nMapping Summary:")
        print(f"  Total ensemble results: {len(ensemble_results)}")
        print(f"  Successfully mapped: {total_mapped}")
        print(f"  Skipped (errors): {skipped_count}")
        
        if mapping_errors:
            print(f"\nFirst 10 mapping errors:")
            for error in mapping_errors[:10]:
                print(f"  - {error}")
            
            # Save full error log
            error_log_path = self.output_dir / "mapping_errors.log"
            with open(error_log_path, 'w') as f:
                f.write("Mapping Errors\n")
                f.write("=" * 50 + "\n")
                for error in mapping_errors:
                    f.write(f"{error}\n")
            print(f"\nFull error log saved to: {error_log_path}")
        
        # Display articles per outlet
        print(f"\nArticles per outlet:")
        for outlet in sorted(self.article_predictions.keys()):
            preds = self.article_predictions[outlet]
            print(f"  {outlet}: {len(preds)} articles")
        
        if not self.article_predictions:
            raise ValueError("No articles successfully mapped to outlets")
        
        return self.article_predictions
    
    def aggregate_outlet_scores(self) -> Dict[str, Dict]:
        """
        Aggregate article-level scores to outlet level with strict validation.
        
        Returns:
            Dictionary with outlet-level aggregated results
        
        Raises:
            ValueError: If scores are invalid or missing
        """
        print("\nAggregating scores by outlet...")
        
        outlet_results = {}
        aggregation_errors = []
        
        for outlet, predictions in self.article_predictions.items():
            if not predictions:
                aggregation_errors.append(f"{outlet}: No predictions available")
                continue
            
            try:
                # Extract and validate scores
                scores = []
                for pred in predictions:
                    score = pred.get('score')
                    
                    # Strict validation - no defaults
                    if score is None:
                        raise ValueError(f"Missing score in prediction: {pred.get('filename', 'unknown')}")
                    
                    if not isinstance(score, (int, float)):
                        raise ValueError(f"Invalid score type in {pred.get('filename', 'unknown')}: {type(score)}")
                    
                    if not -3 <= score <= 3:
                        raise ValueError(f"Score out of range in {pred.get('filename', 'unknown')}: {score}")
                    
                    scores.append(float(score))  # Ensure float for numpy
                
                if not scores:
                    raise ValueError(f"No valid scores for outlet {outlet}")
                
                # Calculate statistics
                median_score = np.median(scores)
                mean_score = np.mean(scores)
                std_dev = np.std(scores)

                # Validate calculated values
                if np.isnan(median_score) or np.isnan(mean_score) or np.isnan(std_dev):
                    raise ValueError(f"NaN values in statistics for {outlet}")

                # Convert to 3-class using MEAN instead of median
                predicted_class = self.score_to_3class(mean_score)
                
                outlet_results[outlet] = {
                    'raw_scores': scores,
                    'median_score': float(median_score),
                    'mean_score': float(mean_score),
                    'std_dev': float(std_dev),
                    'predicted_class': predicted_class,
                    'article_count': len(scores)
                }
                
                print(f"  {outlet}: mean={mean_score:.3f}, class={predicted_class}, n={len(scores)}")
                
            except (ValueError, KeyError, TypeError) as e:
                aggregation_errors.append(f"{outlet}: {e}")
                logger.error(f"Failed to aggregate outlet {outlet}: {e}")
                continue
        
        if aggregation_errors:
            print(f"\nAggregation errors ({len(aggregation_errors)}):")
            for error in aggregation_errors[:5]:
                print(f"  - {error}")
            
            # Save error log
            if hasattr(self, 'output_dir'):
                error_log_path = self.output_dir / "aggregation_errors.log"
                with open(error_log_path, 'w') as f:
                    f.write("Aggregation Errors\n")
                    f.write("=" * 50 + "\n")
                    for error in aggregation_errors:
                        f.write(f"{error}\n")
                print(f"\nFull error log saved to: {error_log_path}")
        
        if not outlet_results:
            raise ValueError("No outlets successfully aggregated")
        
        self.outlet_results = outlet_results
        return outlet_results
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for outlet-level predictions.
        
        Returns:
            Dictionary containing all metrics
        """
        print("\nCalculating evaluation metrics...")
        
        # Prepare data for sklearn metrics
        outlets_with_truth = []
        y_true = []
        y_pred = []
        
        for outlet in self.outlet_results:
            if outlet in self.ground_truth:
                outlets_with_truth.append(outlet)
                y_true.append(self.ground_truth[outlet])
                y_pred.append(self.outlet_results[outlet]['predicted_class'])
        
        if not outlets_with_truth:
            print("ERROR: No outlets with ground truth found!")
            return {}
        
        # Convert to arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=['Left', 'Center', 'Right'], 
                           average='macro', zero_division=0)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=['Left', 'Center', 'Right'], 
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['Left', 'Center', 'Right'])
        
        # Build metrics dictionary
        metrics = {
            'overall_metrics': {
                'total_outlets': len(outlets_with_truth),
                'accuracy': float(accuracy),
                'macro_f1': float(macro_f1)
            },
            'per_class_metrics': {
                'Left': {
                    'precision': float(precision[0]),
                    'recall': float(recall[0]),
                    'f1': float(f1[0]),
                    'support': int(support[0])
                },
                'Center': {
                    'precision': float(precision[1]),
                    'recall': float(recall[1]),
                    'f1': float(f1[1]),
                    'support': int(support[1])
                },
                'Right': {
                    'precision': float(precision[2]),
                    'recall': float(recall[2]),
                    'f1': float(f1[2]),
                    'support': int(support[2])
                }
            },
            'confusion_matrix': cm.tolist(),
            'outlet_details': {}
        }
        
        # Add per-outlet details
        for outlet in outlets_with_truth:
            metrics['outlet_details'][outlet] = {
                'ground_truth': self.ground_truth[outlet],
                'predicted': self.outlet_results[outlet]['predicted_class'],
                'median_score': float(self.outlet_results[outlet]['median_score']),
                'mean_score': float(self.outlet_results[outlet]['mean_score']),
                'std_dev': float(self.outlet_results[outlet]['std_dev']),
                'article_count': self.outlet_results[outlet]['article_count'],
                'correct': self.ground_truth[outlet] == self.outlet_results[outlet]['predicted_class']
            }
        
        # Print summary
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {accuracy:.3f} ({int(accuracy * len(outlets_with_truth))}/{len(outlets_with_truth)} correct)")
        print(f"  Macro F1: {macro_f1:.3f}")
        
        print(f"\nPer-Class Performance:")
        for class_name in ['Left', 'Center', 'Right']:
            cm = metrics['per_class_metrics'][class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {cm['precision']:.3f}")
            print(f"    Recall: {cm['recall']:.3f}")
            print(f"    F1: {cm['f1']:.3f}")
            print(f"    Support: {cm['support']}")
        
        return metrics
    
    def create_violin_plot(self, save_path: str = None):
        """
        Create professional violin plot showing score distributions per outlet.
        Uses AllSides-style color scheme and formatting.
        """
        print("\nCreating violin plot...")
        
        # Sort outlets by mean score for better visualization
        sorted_outlets = sorted(self.outlet_results.items(),
                              key=lambda x: x[1]['mean_score'])
        
        # Prepare data for violin plot
        pos = []
        data = []
        names = []
        counts = []
        means = []
        ground_truths = []
        
        for i, (outlet_name, results) in enumerate(sorted_outlets):
            if outlet_name in self.ground_truth:
                pos.append(i)
                data.append(results['raw_scores'])
                names.append(outlet_name)
                counts.append(results['article_count'])
                means.append(results['mean_score'])
                ground_truths.append(self.ground_truth[outlet_name])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, len(names) * 0.5)))
        
        # Create violin plot
        violin_parts = ax.violinplot(
            data, 
            pos, 
            vert=False,
            showmeans=False, 
            showmedians=False,
            showextrema=False,
            widths=0.7
        )
        
        # Color violins based on ground truth using AllSides colors
        def get_violin_color(ground_truth):
            """Map an AllSides ground-truth class to its plot color."""
            if ground_truth == 'Left':
                return '#2166ac'  # Blue for Left
            elif ground_truth == 'Center':
                return '#808080'  # Gray for Center
            else:
                return '#b2182b'  # Red for Right
        
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(get_violin_color(ground_truths[i]))
            pc.set_edgecolor('black')
            pc.set_alpha(0.8)
        
        # Add median markers (white circles)
        for i, d in enumerate(data):
            ax.scatter(
                np.median(d),
                pos[i],
                marker='o',
                color='white',
                s=50,
                zorder=3,
                edgecolor='black',
                linewidth=1.5
            )
        
        # Add mean markers (black vertical lines)
        for i, mean in enumerate(means):
            ax.scatter(
                mean,
                pos[i],
                marker='|',
                color='black',
                s=200,
                linewidth=2,
                zorder=3
            )
        
        # Add outlet names (without article counts for cleaner appearance)
        ax.set_yticks(pos)
        ax.set_yticklabels(names, fontsize=16)
        
        # Add colored background regions for score ranges
        ax.axvspan(-3, -1, alpha=0.05, color='blue', label='Left Range')
        ax.axvspan(-1, 1, alpha=0.05, color='gray', label='Center Range')
        ax.axvspan(1, 3, alpha=0.05, color='red', label='Right Range')
        
        # Add boundary lines
        ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add category labels at the top
        y_top = len(pos) - 0.5
        ax.text(-2, y_top, 'LEFT', ha='center', va='bottom', fontweight='bold', fontsize=12, color='#2166ac')
        ax.text(0, y_top, 'CENTER', ha='center', va='bottom', fontweight='bold', fontsize=12, color='#808080')
        ax.text(2, y_top, 'RIGHT', ha='center', va='bottom', fontweight='bold', fontsize=12, color='#b2182b')
        
        # Set plot limits and labels
        ax.set_xlim(-3.5, 3.5)
        ax.set_xlabel('Political Bias Score', fontsize=18, fontweight='bold')
        ax.set_title('Media Outlet Bias Score Distributions', fontsize=20, fontweight='bold', pad=20)

        # Set tick label sizes
        ax.tick_params(axis='x', labelsize=16)

        # Add grid
        ax.grid(True, alpha=0.3, axis='x', linestyle=':')
        
        # Add legend - position in lower right to avoid data overlap
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#2166ac', label='Left (Ground Truth)', alpha=0.8),
            Patch(facecolor='#808080', label='Center (Ground Truth)', alpha=0.8),
            Patch(facecolor='#b2182b', label='Right (Ground Truth)', alpha=0.8),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=8, label='Median'),
            Line2D([0], [0], marker='|', color='black', markersize=10,
                   markeredgewidth=2, label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=13)

        plt.tight_layout()

        # Save both PDF and SVG
        if not save_path:
            save_path = self.viz_dir / "violin_plot_professional"
        else:
            save_path = Path(save_path).with_suffix('')  # Remove extension if provided

        self._save_figure(fig, save_path, "violin plot")
        plt.close()
    
    def create_confusion_matrix_plot(self, cm: np.ndarray, save_path: str = None):
        """
        Create confusion matrix heatmap for 3-class system.
        """
        print("\nCreating confusion matrix plot...")
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        labels = ['Left', 'Center', 'Right']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Outlet-Level Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('Ground Truth Class', fontsize=12)
        
        plt.tight_layout()

        # Save both PDF and SVG
        if not save_path:
            save_path = self.viz_dir / "confusion_matrix"
        else:
            save_path = Path(save_path).with_suffix('')  # Remove extension if provided

        self._save_figure(plt.gcf(), save_path, "confusion matrix")
        plt.close()
    
    def create_per_class_performance_plot(self, metrics: Dict, save_path: str = None):
        """
        Create bar chart showing per-class performance metrics.
        """
        print("\nCreating per-class performance plot...")
        
        # Prepare data
        classes = ['Left', 'Center', 'Right']
        metrics_types = ['Precision', 'Recall', 'F1']
        
        data = []
        for class_name in classes:
            class_metrics = metrics['per_class_metrics'][class_name]
            data.append([
                class_metrics['precision'],
                class_metrics['recall'],
                class_metrics['f1']
            ])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(classes))
        width = 0.25
        
        # Create bars
        for i, metric in enumerate(metrics_types):
            values = [data[j][i] for j in range(len(classes))]
            ax.bar(x + i*width, values, width, label=metric)
        
        # Customize plot
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, metric in enumerate(metrics_types):
            values = [data[j][i] for j in range(len(classes))]
            for j, v in enumerate(values):
                ax.text(j + i*width, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()

        # Save both PDF and SVG
        if not save_path:
            save_path = self.viz_dir / "per_class_performance"
        else:
            save_path = Path(save_path).with_suffix('')  # Remove extension if provided

        self._save_figure(fig, save_path, "per-class performance")
        plt.close()
    
    def create_beeswarm_plot(self, save_path: str = None):
        """
        Create beeswarm plot showing individual article scores per outlet.
        Points are colored by outlet ground truth, jittered for visibility.
        """
        print("\nCreating beeswarm plot...")
        
        # Prepare data
        plot_data = []
        outlet_order = []
        
        # Sort outlets by mean score
        sorted_outlets = sorted(self.outlet_results.items(),
                              key=lambda x: x[1]['mean_score'])
        
        for outlet_name, results in sorted_outlets:
            if outlet_name in self.ground_truth:
                outlet_order.append(outlet_name)
                for score in results['raw_scores']:
                    plot_data.append({
                        'outlet': outlet_name,
                        'score': score,
                        'ground_truth': self.ground_truth[outlet_name]
                    })
        
        if not plot_data:
            print("  No data for beeswarm plot")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, len(outlet_order) * 0.5)))
        
        # Plot points with jitter
        np.random.seed(42)  # For reproducibility
        y_positions = {outlet: i for i, outlet in enumerate(outlet_order)}
        
        for item in plot_data:
            y = y_positions[item['outlet']]
            # Add small random jitter for visibility
            y_jitter = y + np.random.normal(0, 0.15)
            
            # Color based on ground truth
            color = {'Left': '#2166ac', 'Center': '#808080', 'Right': '#b2182b'}[item['ground_truth']]
            
            ax.scatter(item['score'], y_jitter, alpha=0.6, s=40, 
                      color=color, edgecolor='black', linewidth=0.5)
        
        # Add median lines
        for outlet in outlet_order:
            y = y_positions[outlet]
            median = self.outlet_results[outlet]['median_score']
            ax.plot([median, median], [y - 0.3, y + 0.3], 
                   color='black', linewidth=2, zorder=5)
        
        # Add colored background regions
        ax.axvspan(-3, -1, alpha=0.05, color='blue')
        ax.axvspan(-1, 1, alpha=0.05, color='gray')
        ax.axvspan(1, 3, alpha=0.05, color='red')
        
        # Add boundary lines
        ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Set labels (without article counts for cleaner appearance)
        ax.set_yticks(list(range(len(outlet_order))))
        ax.set_yticklabels(outlet_order, fontsize=16)
        ax.set_xlim(-3.5, 3.5)
        ax.set_xlabel('Political Bias Score', fontsize=18, fontweight='bold')
        ax.set_title('Article-Level Bias Scores by Media Outlet (Beeswarm Plot)',
                    fontsize=20, fontweight='bold', pad=20)

        # Set tick label sizes
        ax.tick_params(axis='x', labelsize=16)

        # Add grid
        ax.grid(True, alpha=0.3, axis='x', linestyle=':')
        
        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#2166ac', label='Left Outlet', alpha=0.6),
            Patch(facecolor='#808080', label='Center Outlet', alpha=0.6),
            Patch(facecolor='#b2182b', label='Right Outlet', alpha=0.6),
            Line2D([0], [0], color='black', linewidth=2, label='Median Score')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=10)
        
        plt.tight_layout()

        # Save both PDF and SVG
        if not save_path:
            save_path = self.viz_dir / "beeswarm_plot"
        else:
            save_path = Path(save_path).with_suffix('')  # Remove extension if provided

        self._save_figure(fig, save_path, "beeswarm plot")
        plt.close()
    
    def create_outlet_comparison_plot(self, save_path: str = None):
        """
        Create professional bar chart comparing predicted vs ground truth for each outlet.
        Uses AllSides-style color scheme and formatting.
        """
        print("\nCreating outlet comparison plot...")
        
        # Prepare data sorted by mean score
        outlet_data = []
        for outlet, results in sorted(self.outlet_results.items(), key=lambda x: x[1]['mean_score']):
            if outlet in self.ground_truth:
                outlet_data.append({
                    'name': outlet,
                    'median_score': results['median_score'],
                    'mean_score': results['mean_score'],
                    'std_dev': results['std_dev'],
                    'predicted': results['predicted_class'],
                    'ground_truth': self.ground_truth[outlet],
                    'correct': results['predicted_class'] == self.ground_truth[outlet],
                    'article_count': results['article_count']
                })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(outlet_data))
        width = 0.35
        
        # Define color scheme
        def get_color(class_name):
            """Map a bias class name to its plot color."""
            return {'Left': '#2166ac', 'Center': '#808080', 'Right': '#b2182b'}[class_name]
        
        # Create bars for mean scores with error bars
        bars = ax.bar(x, [d['mean_score'] for d in outlet_data], width*2,
                      yerr=[d['std_dev'] for d in outlet_data],
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Color bars based on predicted class
        for bar, d in zip(bars, outlet_data):
            bar.set_facecolor(get_color(d['predicted']))
            if not d['correct']:
                # Add red border for incorrect predictions
                bar.set_edgecolor('red')
                bar.set_linewidth(3)
        
        # Note: Mean score is already shown as bar height, so no separate marker needed
        
        # Add outlet labels with ground truth indicators
        labels = []
        for d in outlet_data:
            checkmark = '[OK]' if d['correct'] else '[X]'
            labels.append(f"{d['name']}\n({d['ground_truth'][0]}) {checkmark}")
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=15)
        
        # Add horizontal lines for class boundaries
        ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add colored background regions
        ax.axhspan(-3, -1, alpha=0.05, color='blue')
        ax.axhspan(-1, 1, alpha=0.05, color='gray')
        ax.axhspan(1, 3, alpha=0.05, color='red')
        
        # Add category labels
        ax.text(len(outlet_data) + 0.5, -2, 'LEFT', fontsize=12, fontweight='bold', 
                va='center', color='#2166ac')
        ax.text(len(outlet_data) + 0.5, 0, 'CENTER', fontsize=12, fontweight='bold', 
                va='center', color='#808080')
        ax.text(len(outlet_data) + 0.5, 2, 'RIGHT', fontsize=12, fontweight='bold', 
                va='center', color='#b2182b')
        
        # Set plot properties
        ax.set_ylim(-3.5, 3.5)
        ax.set_ylabel('Political Bias Score (Mean +/- Std Dev)', fontsize=18, fontweight='bold')
        ax.set_title('Media Outlet Predictions vs Ground Truth', fontsize=20, fontweight='bold', pad=20)

        # Set tick label sizes
        ax.tick_params(axis='y', labelsize=16)

        ax.grid(True, alpha=0.3, axis='y', linestyle=':')
        
        # Add legend - position in lower right to avoid data overlap
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#2166ac', label='Predicted Left', alpha=0.8),
            Patch(facecolor='#808080', label='Predicted Center', alpha=0.8),
            Patch(facecolor='#b2182b', label='Predicted Right', alpha=0.8),
            Patch(facecolor='white', edgecolor='red', linewidth=3, label='Incorrect Prediction')
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=13)

        # Add accuracy text
        accuracy = sum(1 for d in outlet_data if d['correct']) / len(outlet_data)
        ax.text(0.02, 0.98, f'Accuracy: {accuracy:.1%} ({sum(1 for d in outlet_data if d["correct"])}/{len(outlet_data)})',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save both PDF and SVG
        if not save_path:
            save_path = self.viz_dir / "outlet_comparison_professional"
        else:
            save_path = Path(save_path).with_suffix('')  # Remove extension if provided

        self._save_figure(fig, save_path, "outlet comparison")
        plt.close()
    
    def save_detailed_predictions(self):
        """
        Save detailed predictions for each outlet.
        """
        print("\nSaving detailed predictions...")
        
        for outlet, predictions in self.article_predictions.items():
            if not predictions:
                continue
            
            # Prepare outlet report
            outlet_report = {
                'outlet': outlet,
                'ground_truth': self.ground_truth.get(outlet, 'Unknown'),
                'predicted_class': self.outlet_results[outlet]['predicted_class'],
                'median_score': self.outlet_results[outlet]['median_score'],
                'mean_score': self.outlet_results[outlet]['mean_score'],
                'std_dev': self.outlet_results[outlet]['std_dev'],
                'article_count': len(predictions),
                'articles': predictions
            }
            
            # Save to file
            output_path = self.details_dir / f"{outlet.replace(' ', '_')}_predictions.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(outlet_report, f, indent=2)
        
        print(f"  Saved detailed predictions to: {self.details_dir}")
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Complete evaluation report dictionary
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE OUTLET EVALUATION REPORT")
        print("="*60)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Create visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        self.create_violin_plot()
        self.create_beeswarm_plot()
        self.create_outlet_comparison_plot()
        
        if metrics:
            cm = np.array(metrics['confusion_matrix'])
            self.create_confusion_matrix_plot(cm)
            self.create_per_class_performance_plot(metrics)
        
        # Save detailed predictions
        self.save_detailed_predictions()
        
        # Build comprehensive report
        report = {
            'metadata': {
                'timestamp': self.timestamp,
                'output_directory': str(self.output_dir),
                'ensemble_results_dir': str(self.ensemble_results_dir) if self.ensemble_results_dir else None,
                'custom_dataset_dir': str(self.custom_dataset_dir) if self.custom_dataset_dir else None,
                'total_outlets_evaluated': len(self.outlet_results),
                'total_articles_processed': sum(r['article_count'] for r in self.outlet_results.values())
            },
            'metrics': metrics,
            'outlet_results': {
                outlet: {
                    'median_score': float(results['median_score']),
                    'mean_score': float(results['mean_score']),
                    'std_dev': float(results['std_dev']),
                    'predicted_class': results['predicted_class'],
                    'article_count': results['article_count']
                }
                for outlet, results in self.outlet_results.items()
            },
            'ground_truth': self.ground_truth,
            'visualizations': {
                'violin_plot': {
                    'pdf': str(self.viz_dir / "violin_plot_professional.pdf"),
                    'svg': str(self.viz_dir / "violin_plot_professional.svg")
                },
                'beeswarm_plot': {
                    'pdf': str(self.viz_dir / "beeswarm_plot.pdf"),
                    'svg': str(self.viz_dir / "beeswarm_plot.svg")
                },
                'outlet_comparison': {
                    'pdf': str(self.viz_dir / "outlet_comparison_professional.pdf"),
                    'svg': str(self.viz_dir / "outlet_comparison_professional.svg")
                },
                'confusion_matrix': {
                    'pdf': str(self.viz_dir / "confusion_matrix.pdf"),
                    'svg': str(self.viz_dir / "confusion_matrix.svg")
                },
                'per_class_performance': {
                    'pdf': str(self.viz_dir / "per_class_performance.pdf"),
                    'svg': str(self.viz_dir / "per_class_performance.svg")
                }
            }
        }
        
        # Save report
        report_path = self.output_dir / "outlet_evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        return report


def main():
    """
    Main entry point for outlet-level evaluation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate media bias detection at outlet level'
    )
    
    parser.add_argument(
        '--ensemble-dir',
        type=str,
        required=True,
        help='Path to ensemble results directory (e.g., ensemble_outputs_small/session_YYYYMMDD_HHMMSS)'
    )
    
    parser.add_argument(
        '--custom-dataset',
        type=str,
        default='data/balanced_datasets/custom_100_per_outlet',
        help='Path to custom dataset directory (default: data/balanced_datasets/custom_100_per_outlet)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    ensemble_dir = Path(args.ensemble_dir)
    if not ensemble_dir.exists():
        print(f"Error: Ensemble directory not found: {ensemble_dir}")
        sys.exit(1)
    
    dataset_dir = Path(args.custom_dataset)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    config = get_config()
    
    try:
        # Initialize evaluator
        print(f"\nInitializing outlet evaluator...")
        print(f"  Ensemble results: {ensemble_dir}")
        print(f"  Custom dataset: {dataset_dir}")
        
        evaluator = MediaOutletEvaluator(
            config=config,
            ensemble_results_dir=str(ensemble_dir),
            custom_dataset_dir=str(dataset_dir)
        )
        
        # Load data
        print("\nLoading ground truth and results...")
        evaluator.load_allsides_ground_truth()
        evaluator.load_custom_dataset_results()
        
        # Aggregate and evaluate
        print("\nPerforming outlet-level aggregation...")
        evaluator.aggregate_outlet_scores()
        
        # Generate report
        print("\nGenerating evaluation report...")
        report = evaluator.generate_report()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if 'metrics' in report and report['metrics']:
            metrics = report['metrics']
            overall = metrics.get('overall_metrics', {})
            
            print(f"\nOverall Performance:")
            print(f"  Total outlets: {overall.get('total_outlets', 'N/A')}")
            print(f"  Accuracy: {overall.get('accuracy', 0):.3f}")
            print(f"  Macro F1: {overall.get('macro_f1', 0):.3f}")
            
            # Show per-outlet results
            if 'outlet_details' in metrics:
                print(f"\nPer-Outlet Results:")
                for outlet, details in sorted(metrics['outlet_details'].items()):
                    correct = "[OK]" if details['correct'] else "[WRONG]"
                    print(f"  {outlet:20s}: Predicted={details['predicted']:6s}, Truth={details['ground_truth']:6s} {correct}")
        
        print(f"\nResults saved to: {evaluator.output_dir}")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
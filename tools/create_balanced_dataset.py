"""
Balanced Dataset Creator for Ensemble Model Evaluation

This utility creates a balanced, reproducible dataset by:
1. Loading articles from the original dataset (paths from config)
2. Balancing the number of left, center, and right articles
3. Using a fixed random seed for reproducibility  
4. Saving the balanced dataset to config-specified folder
5. Creating a manifest file with article filenames and metadata

Usage:
    python create_balanced_dataset.py --dataset baly --n-samples 1000
"""

import json
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_config

def process_json_file_batch(file_batch: List[Path], outlet_domains: Dict[str, str]) -> Dict[str, List[Tuple[dict, str]]]:
    """
    Process a batch of JSON files and categorize by outlet.
    This function runs in a separate process.
    
    Args:
        file_batch: List of file paths to process
        outlet_domains: Dictionary mapping outlet names to domains
    
    Returns:
        Dictionary with outlet names as keys and list of (article_data, filename) tuples as values
    """
    batch_results = {outlet: [] for outlet in outlet_domains.keys()}
    unmatched = []
    
    for json_file in file_batch:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
            
            # Get source name from article
            source_name = article_data.get('source_name', '').lower()
            
            # Try exact domain matching
            matched = False
            for outlet_name, domain in outlet_domains.items():
                # Check if source_name contains the domain
                if domain in source_name:
                    batch_results[outlet_name].append((article_data, json_file.name))
                    matched = True
                    break
                # Special case for BBC which might be bbc.co.uk
                elif outlet_name == "BBC" and "bbc.co.uk" in source_name:
                    batch_results[outlet_name].append((article_data, json_file.name))
                    matched = True
                    break
            
            if not matched:
                unmatched.append(source_name)
                
        except Exception:
            # Skip files with errors
            continue
    
    return batch_results, unmatched


# Define the 13 fixed media outlets for custom dataset
CUSTOM_MEDIA_OUTLETS = {
    "CNN": "cnn.com",
    "BBC": "bbc.com",  # Also matches bbc.co.uk
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


def get_ground_truth_label(article_data: dict, dataset_type: str) -> Optional[str]:
    """
    Extract ground truth label from article data using bias_text field.
    
    Args:
        article_data: Article data dictionary
        dataset_type: Type of dataset ('baly', 'budak', or 'ad_fontes')
    
    Returns:
        Ground truth label ('left', 'center', 'right') or None if not available
    """
    # All datasets use bias_text field
    bias_text = article_data.get('bias_text', None)
    
    if bias_text is None:
        return None
    
    bias_text = bias_text.lower().strip()
    
    # Handle different dataset formats
    if dataset_type == 'budak':
        # Budak has "lean left" and "lean right" - group them
        if bias_text in ['left', 'lean left']:
            return 'left'
        elif bias_text == 'center':
            return 'center'
        elif bias_text in ['right', 'lean right']:
            return 'right'
    else:
        # Baly and Ad Fontes use simple left/center/right
        if bias_text == 'left':
            return 'left'
        elif bias_text == 'center':
            return 'center'
        elif bias_text == 'right':
            return 'right'
    
    return None


def get_default_paths(config: Dict, dataset_type: str) -> Tuple[Path, Path]:
    """
    Get default input and output directories based on dataset type using config.
    
    Args:
        config: Configuration dictionary
        dataset_type: Type of dataset
    
    Returns:
        Tuple of (input_dir, base_output_dir)
    """
    # Get input directory from config
    if dataset_type == 'baly':
        input_dir = Path(config['dirs']['baly'])
    elif dataset_type == 'budak':
        # Try budak_articles first, then budak
        if 'budak_articles' in config['dirs'] and os.path.exists(config['dirs']['budak_articles']):
            input_dir = Path(config['dirs']['budak_articles'])
        else:
            input_dir = Path(config['dirs']['budak'])
    elif dataset_type == 'ad_fontes':
        input_dir = Path(config['dirs']['ad_fontes'])
    elif dataset_type == 'custom':
        input_dir = Path(config['dirs']['articles'])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Get output base directory from config
    output_base_dir = Path(config['dirs']['balanced_datasets'])
    
    return input_dir, output_base_dir


def load_articles_from_directory(data_dir: Path, dataset_type: str) -> List[Tuple[dict, str, str]]:
    """
    Load all articles from a directory and extract their labels.
    
    Args:
        data_dir: Directory containing article JSON files
        dataset_type: Type of dataset
    
    Returns:
        List of (article_data, filename, label) tuples
    """
    articles_with_labels = []
    
    # Get all JSON files
    json_files = list(data_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {data_dir}")
    
    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in {data_dir}")
    
    # Load and label each article
    skipped = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
            
            # Get ground truth label
            label = get_ground_truth_label(article_data, dataset_type)
            
            if label is not None:
                articles_with_labels.append((article_data, json_file.name, label))
            else:
                skipped += 1
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"Skipped {skipped} articles (no valid bias label)")
    
    # Print distribution
    label_counts = Counter([label for _, _, label in articles_with_labels])
    print(f"\nOriginal distribution:")
    for label in ['left', 'center', 'right']:
        count = label_counts.get(label, 0)
        print(f"  {label:6s}: {count:4d} articles")
    print(f"  Total : {len(articles_with_labels):4d} articles")
    
    return articles_with_labels


def load_custom_articles_by_outlet(data_dir: Path, n_jobs: int = -1) -> Dict[str, List[Tuple[dict, str]]]:
    """
    Load articles from custom dataset and group by media outlet using multiprocessing.
    
    Args:
        data_dir: Directory containing article JSON files
        n_jobs: Number of parallel jobs (-1 for all CPUs)
    
    Returns:
        Dictionary mapping outlet names to list of (article_data, filename) tuples
    """
    articles_by_outlet = {outlet: [] for outlet in CUSTOM_MEDIA_OUTLETS.keys()}
    
    # Get all JSON files
    json_files = list(data_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {data_dir}")
    
    # No sampling - process all files
    print(f"Processing all {len(json_files)} files...")
    
    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in {data_dir}")
    
    # Determine number of workers
    if n_jobs == -1:
        n_jobs = min(mp.cpu_count(), 60)  # Windows has a limit of 61 workers
    else:
        n_jobs = min(n_jobs, 60)
    
    print(f"Processing files using {n_jobs} parallel workers...")
    
    # Split files into batches for parallel processing
    # For large datasets, use bigger batches to reduce overhead
    if len(json_files) > 100000:
        batch_size = max(500, len(json_files) // (n_jobs * 4))  # Bigger batches for large datasets
    else:
        batch_size = max(100, len(json_files) // (n_jobs * 10))  # Standard batches
    
    file_batches = [json_files[i:i + batch_size] for i in range(0, len(json_files), batch_size)]
    print(f"Split into {len(file_batches)} batches of ~{batch_size} files each")
    
    # Track unmatched sources for reporting
    unmatched_sources = set()
    matched_count = 0
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all batch processing tasks
        future_to_batch = {
            executor.submit(process_json_file_batch, batch, CUSTOM_MEDIA_OUTLETS): batch
            for batch in file_batches
        }
        
        # Collect results with progress bar
        with tqdm(total=len(file_batches), desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                try:
                    batch_results, batch_unmatched = future.result()
                    
                    # Merge batch results into main dictionary
                    for outlet_name, articles in batch_results.items():
                        articles_by_outlet[outlet_name].extend(articles)
                        matched_count += len(articles)
                    
                    # Track unmatched sources
                    unmatched_sources.update(batch_unmatched)
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                
                pbar.update(1)
    
    # Print distribution report
    print(f"\nArticles matched by outlet (exact domain matching):")
    print(f"Domains used for matching:")
    for outlet_name, domain in CUSTOM_MEDIA_OUTLETS.items():
        domain_note = f" (also bbc.co.uk)" if outlet_name == "BBC" else ""
        print(f"  {outlet_name:20s}: {domain}{domain_note}")
    
    print(f"\nDistribution by outlet:")
    total_available = 0
    for outlet_name in CUSTOM_MEDIA_OUTLETS.keys():
        count = len(articles_by_outlet[outlet_name])
        total_available += count
        print(f"  {outlet_name:20s}: {count:4d} articles")
    print(f"  {'Total matched':20s}: {matched_count:4d} articles")
    
    if unmatched_sources:
        print(f"\nUnmatched sources ({len(unmatched_sources)} unique):")
        for source in sorted(unmatched_sources)[:10]:  # Show first 10
            print(f"  - {source}")
        if len(unmatched_sources) > 10:
            print(f"  ... and {len(unmatched_sources) - 10} more")
    
    return articles_by_outlet


def create_custom_dataset(
    articles_by_outlet: Dict[str, List[Tuple[dict, str]]],
    n_per_outlet: int,
    random_seed: int = 42
) -> Tuple[List[Tuple[dict, str, str]], Dict]:
    """
    Create a dataset by sampling n articles from each media outlet.
    
    Args:
        articles_by_outlet: Dictionary mapping outlet names to articles
        n_per_outlet: Number of articles to sample per outlet
        random_seed: Random seed for reproducibility
    
    Returns:
        Dataset and sampling statistics
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Filter out outlets with no articles
    available_outlets = {outlet: articles for outlet, articles in articles_by_outlet.items() 
                        if len(articles) > 0}
    
    if not available_outlets:
        print("\nError: No outlets have any articles available.")
        return [], {'error': 'No articles available'}
    
    # Check availability for outlets that have articles
    min_available = min(len(articles) for articles in available_outlets.values())
    
    if min_available < n_per_outlet:
        print(f"\nWarning: Some outlets have fewer than {n_per_outlet} articles.")
        print(f"Minimum available across outlets with data: {min_available} articles.")
        print(f"Adjusting to sample {min_available} articles per outlet.")
        n_per_outlet = min_available
    
    print(f"\nUsing {len(available_outlets)} outlets with available articles:")
    for outlet in sorted(available_outlets.keys()):
        print(f"  - {outlet}")
    
    if len(available_outlets) < len(CUSTOM_MEDIA_OUTLETS):
        missing = set(CUSTOM_MEDIA_OUTLETS.keys()) - set(available_outlets.keys())
        print(f"\nSkipping {len(missing)} outlets with no articles:")
        for outlet in sorted(missing):
            print(f"  - {outlet}")
    
    # Sample from each outlet
    sampled_dataset = []
    sampling_stats = {
        'target_per_outlet': n_per_outlet,
        'total_outlets': len(available_outlets),
        'outlets_used': list(available_outlets.keys()),
        'original_counts': {outlet: len(articles) for outlet, articles in articles_by_outlet.items()},
        'sampled_counts': {},
        'random_seed': random_seed
    }
    
    for outlet_name, articles in available_outlets.items():
        if len(articles) >= n_per_outlet:
            sampled = random.sample(articles, n_per_outlet)
        else:
            sampled = articles  # Take all available if less than n_per_outlet
        
        # Add outlet name as pseudo-label for tracking
        for article_data, filename in sampled:
            sampled_dataset.append((article_data, filename, outlet_name))
        
        sampling_stats['sampled_counts'][outlet_name] = len(sampled)
    
    # Shuffle the final dataset
    random.shuffle(sampled_dataset)
    
    # Update final counts
    sampling_stats['actual_total'] = len(sampled_dataset)
    
    print(f"\nCustom dataset created:")
    for outlet_name in sorted(available_outlets.keys()):
        count = sampling_stats['sampled_counts'].get(outlet_name, 0)
        print(f"  {outlet_name:20s}: {count:4d} articles")
    print(f"  {'Total':20s}: {len(sampled_dataset):4d} articles")
    
    return sampled_dataset, sampling_stats


def create_balanced_dataset(
    articles_with_labels: List[Tuple[dict, str, str]],
    n_samples: int,
    random_seed: int = 42
) -> Tuple[List[Tuple[dict, str, str]], Dict]:
    """
    Create a balanced dataset by sampling equal numbers from each class.
    
    Args:
        articles_with_labels: List of (article_data, filename, label) tuples
        n_samples: Total number of articles to sample
        random_seed: Random seed for reproducibility
    
    Returns:
        Balanced dataset and sampling statistics
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Group articles by label
    articles_by_label = {
        'left': [],
        'center': [],
        'right': []
    }
    
    for article_data, filename, label in articles_with_labels:
        articles_by_label[label].append((article_data, filename, label))
    
    # Calculate articles per class
    articles_per_class = n_samples // 3
    remainder = n_samples % 3
    
    # Check if we have enough articles for each class
    min_available = min(len(articles_by_label[label]) for label in articles_by_label)
    
    if min_available < articles_per_class:
        print(f"\nWarning: Only {min_available} articles available for smallest class.")
        print(f"Adjusting to sample {min_available * 3} articles total.")
        articles_per_class = min_available
        remainder = 0
        n_samples = articles_per_class * 3
    
    # Sample from each class
    balanced_dataset = []
    sampling_stats = {
        'target_total': n_samples,
        'articles_per_class': articles_per_class,
        'original_counts': {label: len(articles) for label, articles in articles_by_label.items()},
        'sampled_counts': {},
        'random_seed': random_seed
    }
    
    # Sample base amount from each class
    for label in ['left', 'center', 'right']:
        available_articles = articles_by_label[label]
        sampled = random.sample(available_articles, articles_per_class)
        balanced_dataset.extend(sampled)
        sampling_stats['sampled_counts'][label] = articles_per_class
    
    # Distribute remainder (if any) to classes with extra articles available
    if remainder > 0:
        for i, label in enumerate(['left', 'center', 'right']):
            if i < remainder and len(articles_by_label[label]) > articles_per_class:
                # Get articles not yet sampled
                already_sampled = set(filename for _, filename, _ in balanced_dataset if _ == label)
                available = [a for a in articles_by_label[label] 
                           if a[1] not in already_sampled]
                if available:
                    extra = random.sample(available, 1)
                    balanced_dataset.extend(extra)
                    sampling_stats['sampled_counts'][label] += 1
    
    # Shuffle the final dataset
    random.shuffle(balanced_dataset)
    
    # Update final counts
    actual_counts = Counter([label for _, _, label in balanced_dataset])
    sampling_stats['sampled_counts'] = dict(actual_counts)
    sampling_stats['actual_total'] = len(balanced_dataset)
    
    print(f"\nBalanced distribution:")
    for label in ['left', 'center', 'right']:
        count = actual_counts.get(label, 0)
        print(f"  {label:6s}: {count:4d} articles")
    print(f"  Total : {len(balanced_dataset):4d} articles")
    
    return balanced_dataset, sampling_stats


def save_balanced_dataset(
    balanced_dataset: List[Tuple[dict, str, str]],
    output_dir: Path,
    sampling_stats: Dict,
    dataset_type: str
) -> Path:
    """
    Save the balanced dataset to a new directory with manifest.
    
    Args:
        balanced_dataset: Balanced dataset
        output_dir: Output directory
        sampling_stats: Sampling statistics
        dataset_type: Type of dataset
    
    Returns:
        Path to the manifest file
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare dataset info based on type
    dataset_info = {
        'dataset_type': dataset_type,
        'random_seed': sampling_stats['random_seed'],
        'creation_time': datetime.now().isoformat(),
        'total_articles': len(balanced_dataset),
    }
    
    # Add type-specific info
    if dataset_type == 'custom':
        dataset_info['articles_per_outlet'] = sampling_stats.get('target_per_outlet', 0)
        dataset_info['total_outlets'] = sampling_stats.get('total_outlets', 0)
    else:
        dataset_info['articles_per_class'] = sampling_stats.get('articles_per_class', 0)
    
    # Save each article and build manifest
    manifest = {
        'dataset_info': dataset_info,
        'sampling_stats': sampling_stats,
        'articles': []
    }
    
    for idx, (article_data, original_filename, label) in enumerate(balanced_dataset):
        # Create new filename with index for easy reference
        new_filename = f"{idx:04d}_{original_filename}"
        output_path = output_dir / new_filename
        
        # Save article
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, indent=2, ensure_ascii=False)
        
        # Add to manifest
        article_entry = {
            'index': idx,
            'filename': new_filename,
            'original_filename': original_filename,
            'content_length': len(article_data.get('content', ''))
        }
        
        # Add label info based on dataset type
        if dataset_type == 'custom':
            article_entry['outlet'] = label  # For custom, label is outlet name
            article_entry['source_name'] = article_data.get('source_name', '')
        else:
            article_entry['ground_truth'] = label
            article_entry['bias_text'] = article_data.get('bias_text', '')
        
        manifest['articles'].append(article_entry)
    
    # Save manifest
    manifest_path = output_dir / 'dataset_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset saved to: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")
    
    # Also save a simple list of filenames for easy loading
    filenames_path = output_dir / 'article_filenames.txt'
    with open(filenames_path, 'w', encoding='utf-8') as f:
        for article in manifest['articles']:
            f.write(f"{article['filename']}\n")
    print(f"Filename list saved to: {filenames_path}")
    
    # Save a summary file
    summary_path = output_dir / 'dataset_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset Summary\n")
        f.write(f"===============\n\n")
        f.write(f"Dataset Type: {dataset_type}\n")
        f.write(f"Created: {manifest['dataset_info']['creation_time']}\n")
        f.write(f"Random Seed: {sampling_stats['random_seed']}\n")
        f.write(f"Total Articles: {len(balanced_dataset)}\n\n")
        f.write(f"Distribution:\n")
        
        if dataset_type == 'custom':
            for outlet, count in sampling_stats['sampled_counts'].items():
                f.write(f"  {outlet:20s}: {count:4d} articles\n")
        else:
            for label, count in sampling_stats['sampled_counts'].items():
                f.write(f"  {label:6s}: {count:4d} articles\n")
    
    return manifest_path


def load_balanced_dataset(balanced_dir: Path) -> Tuple[List[Tuple[dict, str]], Dict]:
    """
    Load a previously created balanced dataset.
    
    Args:
        balanced_dir: Directory containing balanced dataset
    
    Returns:
        List of (article_data, filename) tuples and manifest
    """
    manifest_path = balanced_dir / 'dataset_manifest.json'
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    
    # Load manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    # Load articles in order
    articles = []
    for article_info in manifest['articles']:
        article_path = balanced_dir / article_info['filename']
        with open(article_path, 'r', encoding='utf-8') as f:
            article_data = json.load(f)
        articles.append((article_data, article_info['filename']))
    
    print(f"Loaded {len(articles)} articles from balanced dataset")
    print(f"Dataset type: {manifest['dataset_info']['dataset_type']}")
    print(f"Random seed used: {manifest['dataset_info']['random_seed']}")
    print(f"Created at: {manifest['dataset_info']['creation_time']}")
    
    return articles, manifest


def find_balanced_dataset(config: Dict, dataset_type: str) -> Optional[Path]:
    """
    Find the balanced dataset for a specific dataset type.
    
    Args:
        config: Configuration dictionary
        dataset_type: Dataset type ('baly', 'budak', or 'ad_fontes')
    
    Returns:
        Path to the balanced dataset or None if not found
    """
    balanced_base = Path(config['dirs']['balanced_datasets'])
    if not balanced_base.exists():
        return None
    
    # Fixed path for each dataset type
    balanced_dir = balanced_base / f"balanced_{dataset_type}"
    
    if balanced_dir.exists() and (balanced_dir / 'dataset_manifest.json').exists():
        return balanced_dir
    
    return None


def main():
    """Command-line entry point: parse arguments and create the requested dataset."""
    parser = argparse.ArgumentParser(
        description='Create balanced dataset for ensemble evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create balanced dataset with 1000 articles from Baly dataset
  python create_balanced_dataset.py --dataset baly --n-samples 1000
  
  # Create custom dataset with 100 articles per outlet (1300 total)
  python create_balanced_dataset.py --dataset custom --n-per-outlet 100
  
  # Create custom dataset with custom output directory
  python create_balanced_dataset.py --dataset custom --n-per-outlet 50 --output-dir custom_50_per_outlet
  
  # Load and verify existing balanced dataset
  python create_balanced_dataset.py --load data/balanced_datasets/balanced_baly_1000
        """
    )
    
    # Main arguments
    parser.add_argument('--dataset', choices=['baly', 'budak', 'ad_fontes', 'custom'],
                       help='Type of dataset to balance')
    parser.add_argument('--n_samples', type=int,
                       help='Total number of articles in balanced dataset (for baly/budak/ad_fontes)')
    parser.add_argument('--n-per-outlet', type=int, default=100,
                       help='Number of articles per outlet for custom dataset (default: 100)')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str,
                       help='Output subdirectory name (default: balanced_{dataset}_{n_samples})')
    parser.add_argument('--input-dir', type=str,
                       help='Input directory (default: from config based on dataset type)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel workers for processing (-1 for all CPUs, default: -1)')
    
    # Load existing dataset
    parser.add_argument('--load', type=str,
                       help='Load and verify existing balanced dataset')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Load existing dataset mode
    if args.load:
        balanced_dir = Path(args.load)
        if not balanced_dir.exists():
            # Try relative to config balanced_datasets directory
            balanced_dir = Path(config['dirs']['balanced_datasets']) / args.load
            if not balanced_dir.exists():
                print(f"Error: Directory {args.load} does not exist")
                return 1
        
        try:
            articles, manifest = load_balanced_dataset(balanced_dir)
            
            # Display statistics
            print(f"\nDataset Statistics:")
            print(f"  Total articles: {len(articles)}")
            
            if 'sampling_stats' in manifest:
                stats = manifest['sampling_stats']
                print(f"\nClass Distribution:")
                for label, count in stats.get('sampled_counts', {}).items():
                    print(f"  {label:6s}: {count:4d} articles")
            
            print(f"\nDataset successfully verified!")
            return 0
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return 1
    
    # Create new balanced dataset mode
    if not args.dataset:
        parser.error("--dataset is required when creating a new dataset")
    
    # Check required parameters based on dataset type
    if args.dataset == 'custom':
        if not args.n_per_outlet:
            args.n_per_outlet = 100  # Default value
    else:
        if not args.n_samples:
            parser.error("--n_samples is required for baly/budak/ad_fontes datasets")
    
    # Get default paths from config
    default_input, default_output_base = get_default_paths(config, args.dataset)
    
    # Set up directories
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = default_input
    
    # Set up output directory - Fixed path per dataset type
    if args.output_dir:
        # User specified a subdirectory name (optional, for testing)
        output_dir = default_output_base / args.output_dir
    else:
        # Fixed naming convention - one balanced dataset per type
        if args.dataset == 'custom':
            output_dir = default_output_base / f'custom_{args.n_per_outlet}_per_outlet'
        else:
            output_dir = default_output_base / f'balanced_{args.dataset}'
    
    # Check input directory
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        print(f"Please ensure the {args.dataset} dataset is available")
        return 1
    
    # Check if output directory already exists - always overwrite for fixed paths
    if output_dir.exists():
        print(f"Output directory {output_dir} already exists. Will overwrite...")
        # Clean up existing directory
        import shutil
        shutil.rmtree(output_dir)
        print(f"Removed existing directory: {output_dir}")
    
    print(f"Creating dataset:")
    print(f"  Dataset type: {args.dataset}")
    if args.dataset == 'custom':
        print(f"  Articles per outlet: {args.n_per_outlet}")
        print(f"  Total outlets: {len(CUSTOM_MEDIA_OUTLETS)}")
        print(f"  Expected total: {args.n_per_outlet * len(CUSTOM_MEDIA_OUTLETS)}")
    else:
        print(f"  Target samples: {args.n_samples}")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Random seed: {args.seed}")
    print()
    
    try:
        if args.dataset == 'custom':
            # Custom dataset workflow
            print(f"Loading articles from {input_dir}...")
            articles_by_outlet = load_custom_articles_by_outlet(input_dir, n_jobs=args.n_jobs)
            
            # Check if we have any articles
            total_articles = sum(len(articles) for articles in articles_by_outlet.values())
            if total_articles == 0:
                print("Error: No articles found matching the specified outlets")
                return 1
            
            # Create custom dataset
            print(f"\nCreating custom dataset...")
            custom_dataset, sampling_stats = create_custom_dataset(
                articles_by_outlet,
                args.n_per_outlet,
                args.seed
            )
            
            # Save custom dataset
            print(f"\nSaving custom dataset...")
            manifest_path = save_balanced_dataset(
                custom_dataset,
                output_dir,
                sampling_stats,
                args.dataset
            )
        else:
            # Original balanced dataset workflow
            print(f"Loading articles from {input_dir}...")
            articles_with_labels = load_articles_from_directory(input_dir, args.dataset)
            
            if not articles_with_labels:
                print("Error: No valid articles found")
                return 1
            
            # Create balanced dataset
            print(f"\nCreating balanced dataset...")
            balanced_dataset, sampling_stats = create_balanced_dataset(
                articles_with_labels,
                args.n_samples,
                args.seed
            )
            
            # Save balanced dataset
            print(f"\nSaving balanced dataset...")
            manifest_path = save_balanced_dataset(
                balanced_dataset,
                output_dir,
                sampling_stats,
                args.dataset
            )
        
        print(f"\nDataset creation complete!")
        if args.dataset == 'custom':
            print(f"\nCreated custom dataset with {len(CUSTOM_MEDIA_OUTLETS)} media outlets")
            print(f"Total articles: {sampling_stats.get('actual_total', 0)}")
        print(f"\nTo use this dataset in ensemble models:")
        print(f"  python ensemble_multi_model.py --data-dir {output_dir}")
        print(f"  python ensemble_multi_model_small.py --data-dir {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
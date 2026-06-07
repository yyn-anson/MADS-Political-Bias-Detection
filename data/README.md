# Data Directory

This directory contains balanced datasets for political bias detection evaluation.

---

## 📁 Directory Structure

```
data/
└── balanced_datasets/          # Balanced datasets by political bias
    ├── balanced_baly/          # Baly et al. dataset
    ├── balanced_budak/         # Budak et al. dataset
    ├── balanced_ad_fontes/     # Ad Fontes Media dataset
    └── custom_100_per_outlet/  # Custom outlet-based dataset
```

---

## 🎯 What You Need

Each dataset folder should contain:
1. **Article JSON files** - Individual articles with metadata
2. **dataset_manifest.json** - Metadata about the dataset

---

## 📥 Option 1: Download Pre-Balanced Datasets (Recommended)

### Baly Dataset
- **Size**: ~1000 articles
- **Source**: [Download link to be added]
- **Extract to**: `data/balanced_datasets/balanced_baly/`

### Budak Dataset
- **Size**: ~500 articles
- **Source**: [Download link to be added]
- **Extract to**: `data/balanced_datasets/balanced_budak/`

### Ad Fontes Dataset
- **Size**: ~800 articles
- **Source**: [Download link to be added]
- **Extract to**: `data/balanced_datasets/balanced_ad_fontes/`

### Custom Outlet Dataset
- **Size**: 100 articles × 13 outlets = 1300 articles
- **Source**: [Download link to be added]
- **Extract to**: `data/balanced_datasets/custom_100_per_outlet/`

---

## 🛠️ Option 2: Create Balanced Datasets from Raw Data

If you have raw datasets, use the balancing tool:

```bash
# Create balanced Baly dataset
python tools/create_balanced_dataset.py --dataset baly --n-samples 1000

# Create balanced Budak dataset
python tools/create_balanced_dataset.py --dataset budak --n-samples 500

# Create balanced Ad Fontes dataset
python tools/create_balanced_dataset.py --dataset ad_fontes --n-samples 800

# Create custom outlet dataset (100 articles per outlet)
python tools/create_balanced_dataset.py --dataset custom --samples-per-outlet 100
```

**Note**: You need raw datasets in the original format first. See [docs/DATASETS.md](../docs/DATASETS.md) for details.

---

## 📄 Dataset Format

### Article JSON File Format

Each article should be a JSON file with the following structure:

```json
{
  "source_name": "cnn.com",
  "content": "Article text content here...",
  "bias": 0,
  "title": "Article Title",
  "url": "https://example.com/article",
  "date": "2024-01-15"
}
```

**Required fields**:
- `source_name`: Media outlet domain (e.g., "cnn.com")
- `content`: Full article text

**Optional fields** (dataset-specific):
- `bias`: Numeric bias score (Baly: -3 to +3, Ad Fontes: -42 to +42)
- `bias_text`: Text bias label ("left", "center", "right")
- `title`, `url`, `date`: Metadata

### Dataset Manifest Format

The `dataset_manifest.json` file contains dataset metadata:

```json
{
  "dataset_type": "baly",
  "total_articles": 1000,
  "creation_date": "2025-01-20",
  "balance": {
    "left": 333,
    "center": 334,
    "right": 333
  },
  "articles": [
    {
      "filename": "article_001.json",
      "bias": -2.0,
      "source": "cnn.com"
    },
    ...
  ]
}
```

---

## ✅ Verify Dataset

After placing or creating datasets, verify they're correctly formatted:

```bash
# Check directory structure
ls data/balanced_datasets/balanced_baly/

# Should see:
# - Multiple .json files (article_*.json)
# - dataset_manifest.json

# Verify manifest
cat data/balanced_datasets/balanced_baly/dataset_manifest.json

# Test loading
python -c "
import json
from pathlib import Path

manifest_path = Path('data/balanced_datasets/balanced_baly/dataset_manifest.json')
manifest = json.load(open(manifest_path))
print(f'Dataset: {manifest[\"dataset_type\"]}')
print(f'Total articles: {manifest[\"total_articles\"]}')
print(f'Balance: {manifest[\"balance\"]}')
"
```

---

## 📊 Dataset Characteristics

### Baly Dataset
- **Papers**: Baly et al. (2018, 2020)
- **Bias Scale**: -3 (left) to +3 (right)
- **Articles**: News articles from various outlets
- **Ground Truth**: Human-annotated media bias scores

### Budak Dataset
- **Paper**: Budak et al. (2016)
- **Bias Scale**: Text labels ("left", "center", "right")
- **Articles**: U.S. political news articles
- **Ground Truth**: Expert annotations

### Ad Fontes Dataset
- **Source**: Ad Fontes Media
- **Bias Scale**: -42 (left) to +42 (right)
- **Articles**: Multi-topic news articles
- **Ground Truth**: Professional media analysts

### Custom Outlet Dataset
- **Purpose**: Outlet-level bias evaluation
- **Structure**: 100 articles per outlet × 13 outlets
- **Outlets**: CNN, BBC, Fox News, NYT, MSNBC, etc.
- **No Ground Truth**: Designed for outlet comparison, not accuracy evaluation

---

## 🔍 Data Sources

### Original Papers
- Baly, R., et al. (2018). "Predicting factuality of reporting and bias of news media sources." EMNLP.
- Baly, R., et al. (2020). "What was written vs. who read it: News media profiling using text analysis and social media context." ACL.
- Budak, C., et al. (2016). "Fair and balanced? Quantifying media bias through crowdsourced content analysis." Public Opinion Quarterly.

### Dataset Access
- **Baly**: [Original repository link]
- **Budak**: [Original repository link]
- **Ad Fontes**: [Ad Fontes Media website]

---

## 💡 Tips

1. **Start Small**: Test with 10-100 articles first to verify everything works
2. **Balance Matters**: Ensure equal representation of left/center/right for fair evaluation
3. **Quality Check**: Validate article content is complete (no truncated text)
4. **Storage**: Large datasets (1000+ articles) may require 1-2GB of storage

---

## 🆘 Troubleshooting

### "Dataset not found" error
```bash
# Verify path
ls data/balanced_datasets/balanced_baly/dataset_manifest.json

# If missing, create it
python tools/create_balanced_dataset.py --dataset baly --n-samples 100
```

### "No articles found" error
```bash
# Check for JSON files
ls data/balanced_datasets/balanced_baly/*.json | wc -l

# Should show number of article files
```

### Invalid JSON format
```bash
# Validate a sample file
python -c "import json; print(json.load(open('data/balanced_datasets/balanced_baly/article_001.json')))"
```

---

## 📧 Need Help?

- See [docs/DATASETS.md](../docs/DATASETS.md) for detailed dataset information
- Open an issue on GitHub for dataset-related questions
- Check [QUICKSTART.md](../QUICKSTART.md) for quick setup guide

---

**Important**: Do not commit large dataset files to Git! They are excluded via `.gitignore`.

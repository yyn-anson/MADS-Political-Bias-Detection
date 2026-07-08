# Dataset Information

Comprehensive guide to datasets used in the Multi-Agent Bias Detection System.

---

## Supported Datasets

### 1. Baly Dataset (Baly et al., 2018, 2020)

**Source**: Academic research on media bias detection

**Format**:
```json
{
  "source_name": "cnn.com",
  "content": "Article full text...",
  "bias": -1.5,
  "title": "Article Title",
  "url": "https://...",
  "date": "2024-01-15"
}
```

**Bias Scale**: -3 (strongly left) to +3 (strongly right)
- -3 to -1: Left-leaning
- -1 to +1: Center
- +1 to +3: Right-leaning

**Size**: ~3,000 articles
**Balanced Subset**: 1,000 articles (333 left, 334 center, 333 right)

**References**:
- Baly, R., et al. (2018). "Predicting factuality of reporting and bias of news media sources." EMNLP.
- Baly, R., et al. (2020). "What was written vs. who read it." ACL.

---

### 2. Budak Dataset (Budak et al., 2016)

**Source**: Crowdsourced content analysis research

**Format**:
```json
{
  "source_name": "foxnews.com",
  "content": "Article text...",
  "bias_text": "right",
  "title": "...",
  "url": "...",
  "topic": "politics"
}
```

**Bias Scale**: Text labels
- "left" or "lean left"
- "center" or "neutral"
- "right" or "lean right"

**Size**: ~1,500 articles
**Balanced Subset**: 600 articles (200 left, 200 center, 200 right)

**Reference**:
- Budak, C., et al. (2016). "Fair and balanced? Quantifying media bias through crowdsourced content analysis." POQ.

---

### 3. Ad Fontes Dataset

**Source**: Ad Fontes Media professional analysis

**Format**:
```json
{
  "source_name": "nytimes.com",
  "content": "Article text...",
  "Bias": -15.0,
  "Reliability": 42.0,
  "title": "...",
  "date": "2024-01-15"
}
```

**Bias Scale**: -42 (extremely left) to +42 (extremely right)
- < -15: Left
- -15 to +15: Center
- > +15: Right

**Size**: ~2,000 articles
**Balanced Subset**: 900 articles (300 left, 300 center, 300 right)

**Source**: https://www.adfontesmedia.com/

---

### 4. Custom Outlet Dataset

**Purpose**: Outlet-level bias analysis (no ground truth)

**Format**:
```json
{
  "source_name": "cnn.com",
  "content": "Article text...",
  "title": "...",
  "outlet": "CNN",
  "date": "2024-01-15"
}
```

**Structure**: 100 articles x 13 outlets = 1,300 articles

**Outlets**:
1. CNN (left-leaning)
2. BBC (center)
3. Fox News (right-leaning)
4. Breitbart (right-leaning)
5. The Guardian (left-leaning)
6. New York Times (left-center)
7. TIME (center)
8. New York Post (right-center)
9. MSNBC (left-leaning)
10. The Nation (left-leaning)
11. Washington Examiner (right-leaning)
12. Newsweek (center)
13. Forbes (center-right)

---

## Dataset Preparation

### Creating Balanced Datasets

```bash
# Baly dataset
python tools/create_balanced_dataset.py --dataset baly --n-samples 1000

# Budak dataset
python tools/create_balanced_dataset.py --dataset budak --n-samples 600

# Ad Fontes dataset
python tools/create_balanced_dataset.py --dataset ad_fontes --n-samples 900

# Custom outlet dataset
python tools/create_balanced_dataset.py --dataset custom --samples-per-outlet 100
```

### Required Directory Structure

```
data/balanced_datasets/
├── balanced_baly/
│   ├── article_001.json
│   ├── article_002.json
│   ├── ...
│   └── dataset_manifest.json
├── balanced_budak/
│   └── ...
├── balanced_ad_fontes/
│   └── ...
└── custom_100_per_outlet/
    └── ...
```

### Manifest File Format

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
      "source": "cnn.com",
      "direction": "left"
    },
    ...
  ]
}
```

---

## Dataset Statistics

| Dataset | Total | Left | Center | Right | Outlets | Avg Length |
|---------|-------|------|--------|-------|---------|------------|
| Baly | 1000 | 333 | 334 | 333 | ~50 | 800 words |
| Budak | 600 | 200 | 200 | 200 | ~30 | 650 words |
| Ad Fontes | 900 | 300 | 300 | 300 | ~40 | 750 words |
| Custom | 1300 | N/A | N/A | N/A | 13 | 700 words |

---

## Dataset Validation

### Check Dataset Integrity

```python
import json
from pathlib import Path
from collections import Counter

# Load manifest
manifest_path = Path('data/balanced_datasets/balanced_baly/dataset_manifest.json')
manifest = json.load(open(manifest_path))

# Verify counts
print(f"Total articles: {manifest['total_articles']}")
print(f"Balance: {manifest['balance']}")

# Check files exist
dataset_dir = manifest_path.parent
missing = []
for article in manifest['articles']:
    filepath = dataset_dir / article['filename']
    if not filepath.exists():
        missing.append(article['filename'])

print(f"Missing files: {len(missing)}")
```

### Verify Article Format

```bash
# Test load a sample article
python -c "
import json
article = json.load(open('data/balanced_datasets/balanced_baly/article_001.json'))
print('Required fields:', all(k in article for k in ['source_name', 'content']))
print('Content length:', len(article.get('content', '')))
"
```

---

## Ground Truth Mapping

### Converting to Common Format

All datasets normalized to:
- 0 = Left
- 1 = Center
- 2 = Right

**Baly**:
```python
if bias <= -1: label = 0  # Left
elif bias >= 1: label = 2  # Right
else: label = 1            # Center
```

**Budak**:
```python
label = {
    'left': 0, 'lean left': 0,
    'center': 1, 'neutral': 1,
    'right': 2, 'lean right': 2
}[bias_text.lower()]
```

**Ad Fontes**:
```python
if Bias < -15: label = 0   # Left
elif Bias > 15: label = 2  # Right
else: label = 1            # Center
```

---

## Data Sources & Access

### Baly Dataset
- **Access**: Distributed by the original authors; the balanced evaluation
  subset used here is included in this repository under
  `data/balanced_datasets/balanced_baly/`
- **License**: Research use
- **Citation Required**: Yes

### Budak Dataset
- **Access**: Distributed by the original authors; the balanced evaluation
  subset used here is included in this repository under
  `data/balanced_datasets/balanced_budak/`
- **License**: Research use
- **Citation Required**: Yes

### Ad Fontes Dataset
- **Website**: https://www.adfontesmedia.com/
- **License**: May require license for research
- **Citation**: Ad Fontes Media

### Raw Article Collection
For creating your own datasets:
```python
# Example: Collect from GDELT or NewsAPI
# See tools/create_balanced_dataset.py for implementation
```

---

## Best Practices

1. **Balance**: Always ensure equal class distribution
2. **Quality**: Remove truncated or incomplete articles
3. **Diversity**: Include multiple outlets per bias category
4. **Size**: Start small (100-500) for testing, scale up for production
5. **Validation**: Always verify dataset integrity before training/evaluation

---

## Troubleshooting

**Issue**: "Dataset not found"
```bash
# Check path
ls data/balanced_datasets/balanced_baly/dataset_manifest.json

# Verify config.py points to correct location
python -c "from config import get_config; print(get_config()['datasets']['baly'])"
```

**Issue**: "Unbalanced dataset"
```bash
# Re-create with balancing
python tools/create_balanced_dataset.py --dataset baly --n-samples 1000
```

**Issue**: "Invalid JSON format"
```bash
# Validate JSON
python -m json.tool data/balanced_datasets/balanced_baly/article_001.json
```

---

For dataset preparation scripts, see `tools/create_balanced_dataset.py`.

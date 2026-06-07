"""
Shared fixtures for all tests inside multi_agent_bias_detection/.

All paths are resolved relative to the package root (parent of this tests/ dir),
so this file works whether the package is run as a standalone repo or as a subfolder.

  PACKAGE_ROOT  — multi_agent_bias_detection/
  DATA_DIR      — multi_agent_bias_detection/data/
  ALLSIDES_CSV  — data/allsides/AllSides_Rating.csv
  LABELS_CSV    — data/article_outlet_labels.csv
  EXCEL_PATH    — data/outlet_bias_reference.xlsx
  LABELED_DIR   — data/labeled_articles/   (gitignored; created by running the script)
"""

import json
import pytest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PACKAGE_ROOT / "data"
ALLSIDES_CSV = DATA_DIR / "allsides" / "AllSides_Rating.csv"
LABELS_CSV   = DATA_DIR / "article_outlet_labels.csv"
EXCEL_PATH   = DATA_DIR / "outlet_bias_reference.xlsx"
LABELED_DIR  = DATA_DIR / "labeled_articles"
ARTICLES_DIR = DATA_DIR / "articles"

EXPECTED_LABELS  = {"Left", "Lean Left", "Center", "Lean Right", "Right", "Mixed", "no_label"}
EXPECTED_FOLDERS = {"Left", "Lean_Left", "Center", "Lean_Right", "Right", "Mixed", "no_label"}


@pytest.fixture(scope="session")
def allsides_csv_path():
    if not ALLSIDES_CSV.exists():
        pytest.skip(f"AllSides CSV not present: {ALLSIDES_CSV}")
    return ALLSIDES_CSV


@pytest.fixture(scope="session")
def labels_csv_path():
    if not LABELS_CSV.exists():
        pytest.skip("Labels CSV not present — run: python tools/label_articles_by_outlet.py")
    return LABELS_CSV


@pytest.fixture(scope="session")
def excel_path():
    if not EXCEL_PATH.exists():
        pytest.skip("Excel file not present — run: python tools/label_articles_by_outlet.py")
    return EXCEL_PATH


@pytest.fixture(scope="session")
def labeled_dir():
    if not LABELED_DIR.exists():
        pytest.skip(
            "labeled_articles/ not present — run: python tools/label_articles_by_outlet.py\n"
            "(This directory is gitignored because it is 3.9 GB; regenerate locally.)"
        )
    return LABELED_DIR


@pytest.fixture(scope="session")
def sample_articles_dir(tmp_path_factory):
    """Small temp directory with hand-crafted article JSONs for labeling pipeline tests."""
    d = tmp_path_factory.mktemp("articles")
    articles = [
        {"record_id": "test-001", "source_name": "cnn.com",        "title": "Test 1", "content_original": "..."},
        {"record_id": "test-002", "source_name": "foxnews.com",     "title": "Test 2", "content_original": "..."},
        {"record_id": "test-003", "source_name": "nytimes.com",     "title": "Test 3", "content_original": "..."},
        {"record_id": "test-004", "source_name": "unknown-xyz.io",  "title": "Test 4", "content_original": "..."},
        {"record_id": "test-005", "source_name": "",                 "title": "Test 5", "content_original": "..."},
        {"record_id": "test-006", "source_name": "www.bbc.co.uk",   "title": "Test 6", "content_original": "..."},
    ]
    for art in articles:
        (d / f"{art['record_id']}.json").write_text(json.dumps(art), encoding="utf-8")
    return d

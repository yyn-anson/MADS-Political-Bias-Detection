"""Strict loaders for JSON, JSONL, CSV, and plain-text article inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .types import Article, BiasLabel

SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv", ".txt"}


def _stable_id(title: str, text: str) -> str:
    digest = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()[:12]
    return f"article-{digest}"


def article_from_mapping(data: Mapping[str, Any], origin: str = "input") -> Article:
    text = str(data.get("text") or data.get("content") or "").strip()
    title = str(data.get("title") or "Untitled article").strip()
    if len(text) < 80:
        raise ValueError(f"{origin}: article text must contain at least 80 characters")
    article_id = str(data.get("id") or _stable_id(title, text)).strip()
    label_raw = data.get("label") or data.get("expected_label")
    label = BiasLabel.parse(label_raw) if label_raw not in (None, "") else None
    known = {
        "id",
        "title",
        "text",
        "content",
        "source",
        "url",
        "published_at",
        "label",
        "expected_label",
    }
    metadata = {key: value for key, value in data.items() if key not in known}
    return Article(
        id=article_id,
        title=title,
        text=text,
        source=str(data["source"]).strip() if data.get("source") else None,
        url=str(data["url"]).strip() if data.get("url") else None,
        published_at=str(data["published_at"]).strip() if data.get("published_at") else None,
        label=label,
        metadata=metadata,
    )


def _load_json(path: Path, limit: int | None = None) -> list[Article]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    records = data if isinstance(data, list) else [data]
    if limit is not None:
        records = records[:limit]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: JSON must be an article object or an array of objects")
    return [
        article_from_mapping(record, f"{path}[{index}]") for index, record in enumerate(records)
    ]


def _load_jsonl(path: Path, limit: int | None = None) -> list[Article]:
    articles = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: each line must be a JSON object")
            articles.append(article_from_mapping(record, f"{path}:{line_number}"))
            if limit is not None and len(articles) >= limit:
                break
    return articles


def _load_csv(path: Path, limit: int | None = None) -> list[Article]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        articles = []
        for index, row in enumerate(csv.DictReader(handle), 2):
            articles.append(article_from_mapping(row, f"{path}:{index}"))
            if limit is not None and len(articles) >= limit:
                break
        return articles


def _load_text(path: Path, limit: int | None = None) -> list[Article]:
    text = path.read_text(encoding="utf-8").strip()
    return [Article(id=path.stem, title=path.stem.replace("_", " ").title(), text=text)]


def load_articles(path: str | Path, *, limit: int | None = None) -> list[Article]:
    """Load every supported article file in a deterministic order."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    files = (
        [input_path]
        if input_path.is_file()
        else sorted(
            item
            for item in input_path.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
        )
    )
    loaders = {
        ".json": _load_json,
        ".jsonl": _load_jsonl,
        ".csv": _load_csv,
        ".txt": _load_text,
    }
    articles: list[Article] = []
    for file_path in files:
        remaining = None if limit is None else limit - len(articles)
        if remaining == 0:
            break
        articles.extend(loaders[file_path.suffix.lower()](file_path, remaining))
    if not articles:
        raise ValueError(
            f"no articles found in {input_path}; supported formats: "
            + ", ".join(sorted(SUPPORTED_SUFFIXES))
        )
    seen: set[str] = set()
    duplicates: set[str] = set()
    for article in articles:
        if article.id in seen:
            duplicates.add(article.id)
        seen.add(article.id)
    if duplicates:
        raise ValueError(f"article ids must be unique; duplicates: {sorted(duplicates)}")
    return articles


def validate_article_lengths(articles: Iterable[Article], max_characters: int) -> None:
    too_long = [article.id for article in articles if len(article.text) > max_characters]
    if too_long:
        raise ValueError(
            "articles exceed the configured full-context safety limit "
            f"({max_characters} characters): {too_long}. Increase "
            "method.max_article_characters only if the model context supports them."
        )

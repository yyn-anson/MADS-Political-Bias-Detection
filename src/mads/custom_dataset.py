"""Convert the project's historical article corpus to the public MADS schema."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ALLSIDES_TO_MADS = {
    "Left": "Left",
    "Lean Left": "Left",
    "Center": "Center",
    "Lean Right": "Right",
    "Right": "Right",
}


@dataclass(frozen=True)
class OutletLabel:
    source_name: str
    allsides_source_name: str
    allsides_label: str
    matched: bool


@dataclass(frozen=True)
class ConversionStats:
    discovered: int
    written: int
    labeled: int
    unlabeled: int
    invalid: int
    labels: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_columns(reader: csv.DictReader, path: Path) -> None:
    required = {
        "record_id",
        "source_name",
        "allsides_label",
        "allsides_source_name",
        "matched",
    }
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path}: missing columns: {sorted(missing)}")


def load_label_index(path: str | Path) -> dict[str, OutletLabel]:
    """Load the committed article-to-AllSides lookup and reject ambiguous IDs."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"label table does not exist: {csv_path}")

    labels: dict[str, OutletLabel] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _required_columns(reader, csv_path)
        for line_number, row in enumerate(reader, 2):
            record_id = (row.get("record_id") or "").strip()
            if not record_id:
                raise ValueError(f"{csv_path}:{line_number}: blank record_id")
            if record_id in labels:
                raise ValueError(f"{csv_path}:{line_number}: duplicate record_id {record_id!r}")
            matched = (row.get("matched") or "").strip().lower()
            if matched not in {"true", "false"}:
                raise ValueError(f"{csv_path}:{line_number}: matched must be True or False")
            labels[record_id] = OutletLabel(
                source_name=(row.get("source_name") or "").strip(),
                allsides_source_name=(row.get("allsides_source_name") or "").strip(),
                allsides_label=(row.get("allsides_label") or "").strip(),
                matched=matched == "true",
            )
    return labels


def _first_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_custom_article(
    record: dict[str, Any], origin: Path, labels: dict[str, OutletLabel]
) -> dict[str, Any]:
    """Normalize one historical article without changing its article body."""
    record_id = _first_text(record, "record_id", "id", "ID") or origin.stem
    text = _first_text(record, "text", "content_original", "content")
    if len(text) < 80:
        raise ValueError("article text must contain at least 80 characters")

    reference = labels.get(record_id)
    source = _first_text(record, "source", "source_name", "outlet")
    if not source and reference:
        source = reference.source_name

    normalized: dict[str, Any] = {
        "id": record_id,
        "title": _first_text(record, "title") or "Untitled article",
        "text": text,
    }
    optional = {
        "source": source,
        "url": _first_text(record, "url", "source_url"),
        "published_at": _first_text(record, "published_at", "date", "published"),
    }
    normalized.update({key: value for key, value in optional.items() if value})

    if reference:
        normalized["outlet_label_allsides_2025"] = reference.allsides_label
        mapped = ALLSIDES_TO_MADS.get(reference.allsides_label) if reference.matched else None
        if mapped:
            normalized["label"] = mapped
            normalized["label_source"] = "AllSides 2025 outlet rating"
    return normalized


def convert_custom_dataset(
    articles_dir: str | Path,
    labels_csv: str | Path,
    output_path: str | Path,
    *,
    errors_path: str | Path | None = None,
) -> ConversionStats:
    """Stream historical JSON files into one deterministic, atomic JSONL file."""
    source_dir = Path(articles_dir)
    output = Path(output_path)
    errors = Path(errors_path) if errors_path else output.with_suffix(".errors.csv")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"article directory does not exist: {source_dir}")
    article_files = sorted(source_dir.rglob("*.json"))
    if not article_files:
        raise ValueError(f"no JSON articles found in {source_dir}")

    labels = load_label_index(labels_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors.parent.mkdir(parents=True, exist_ok=True)
    label_counts: Counter[str] = Counter()
    written = labeled = invalid = 0
    seen: set[str] = set()

    with (
        tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent, delete=False
        ) as output_handle,
        tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=errors.parent, delete=False
        ) as error_handle,
    ):
        temporary_output = Path(output_handle.name)
        temporary_errors = Path(error_handle.name)
        error_writer = csv.DictWriter(error_handle, fieldnames=["file", "error"])
        error_writer.writeheader()
        for article_path in article_files:
            try:
                record = json.loads(article_path.read_text(encoding="utf-8"))
                if not isinstance(record, dict):
                    raise TypeError("article JSON must be an object")
                normalized = normalize_custom_article(record, article_path, labels)
                if normalized["id"] in seen:
                    raise ValueError(f"duplicate article id {normalized['id']!r}")
                seen.add(normalized["id"])
                output_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                written += 1
                if "label" in normalized:
                    labeled += 1
                    label_counts[normalized["label"]] += 1
            except (json.JSONDecodeError, OSError, TypeError, UnicodeError, ValueError) as exc:
                invalid += 1
                error_writer.writerow({"file": str(article_path), "error": str(exc)})

    if written == 0:
        temporary_output.unlink(missing_ok=True)
        temporary_errors.replace(errors)
        raise ValueError(f"none of the {len(article_files)} discovered articles were valid")
    temporary_output.replace(output)
    if invalid:
        temporary_errors.replace(errors)
    else:
        temporary_errors.unlink(missing_ok=True)
        errors.unlink(missing_ok=True)

    return ConversionStats(
        discovered=len(article_files),
        written=written,
        labeled=labeled,
        unlabeled=written - labeled,
        invalid=invalid,
        labels=dict(sorted(label_counts.items())),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the historical custom corpus to the recommended MADS JSONL format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--articles", default="data/custom_dataset/raw_articles")
    parser.add_argument("--labels", default="data/article_outlet_labels.csv")
    parser.add_argument("--output", default="data/custom_dataset/articles.jsonl")
    parser.add_argument("--errors", default="data/custom_dataset/conversion_errors.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        stats = convert_custom_dataset(
            args.articles, args.labels, args.output, errors_path=args.errors
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Converted {stats.written:,} of {stats.discovered:,} article(s) -> {args.output}")
    print(f"Labeled: {stats.labeled:,}; unlabeled: {stats.unlabeled:,}")
    if stats.labels:
        print("Labels: " + ", ".join(f"{key}={value:,}" for key, value in stats.labels.items()))
    if stats.invalid:
        print(f"Invalid: {stats.invalid:,}; details: {args.errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

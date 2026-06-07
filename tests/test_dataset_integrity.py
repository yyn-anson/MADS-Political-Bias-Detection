"""
Integration tests that verify the integrity of the labeled dataset produced by
label_articles_by_outlet.py.

These tests read the real output files (data/article_outlet_labels.csv,
data/outlet_bias_reference.xlsx, data/labeled_articles/) and assert that counts,
structure, and cross-file consistency are correct.

Skip automatically if the output files haven't been generated yet.
Mark: pytest -m integration  (all tests in this file are integration-level)
"""

import csv
import pytest
from collections import Counter
from pathlib import Path

from tests.conftest import (
    LABELS_CSV, EXCEL_PATH, LABELED_DIR, ARTICLES_DIR,
    EXPECTED_LABELS, EXPECTED_FOLDERS,
)

pytestmark = pytest.mark.integration

EXPECTED_TOTAL = 473_989
EXPECTED_MATCHED = 203_555
EXPECTED_UNMATCHED = 270_434


# ---------------------------------------------------------------------------
# CSV integrity
# ---------------------------------------------------------------------------

class TestLabelsCsvIntegrity:

    def test_csv_exists(self, labels_csv_path):
        assert labels_csv_path.is_file()

    def test_csv_row_count(self, labels_csv_path):
        with open(labels_csv_path, newline="", encoding="utf-8") as f:
            count = sum(1 for _ in csv.DictReader(f))
        assert count == EXPECTED_TOTAL, (
            f"Expected {EXPECTED_TOTAL} rows, got {count}"
        )

    def test_csv_headers(self, labels_csv_path):
        with open(labels_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            next(reader)  # read one row to populate fieldnames
            assert set(reader.fieldnames) == {
                "record_id", "source_name", "allsides_label", "allsides_source_name", "matched"
            }

    def test_csv_no_blank_record_ids(self, labels_csv_path):
        blank = 0
        with open(labels_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row["record_id"].strip():
                    blank += 1
        assert blank == 0, f"{blank} rows have a blank record_id"

    def test_csv_all_labels_valid(self, labels_csv_path):
        invalid = set()
        with open(labels_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row["allsides_label"]
                if lbl not in EXPECTED_LABELS:
                    invalid.add(lbl)
        assert not invalid, f"Unexpected label values found: {invalid}"

    def test_csv_matched_flag_is_boolean_string(self, labels_csv_path):
        bad = set()
        with open(labels_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["matched"] not in ("True", "False"):
                    bad.add(row["matched"])
        assert not bad, f"'matched' column has unexpected values: {bad}"

    def test_csv_matched_count(self, labels_csv_path):
        matched = 0
        with open(labels_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["matched"] == "True":
                    matched += 1
        assert matched == EXPECTED_MATCHED, (
            f"Expected {EXPECTED_MATCHED} matched rows, got {matched}"
        )

    def test_csv_unmatched_no_allsides_source_name(self, labels_csv_path):
        wrong = 0
        with open(labels_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["matched"] == "False" and row["allsides_source_name"].strip():
                    wrong += 1
        assert wrong == 0, (
            f"{wrong} unmatched rows unexpectedly have an allsides_source_name"
        )

    def test_csv_no_label_implies_unmatched(self, labels_csv_path):
        inconsistent = 0
        with open(labels_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["allsides_label"] == "no_label" and row["matched"] == "True":
                    inconsistent += 1
        assert inconsistent == 0, (
            f"{inconsistent} rows have 'no_label' but matched=True"
        )

    def test_csv_label_distribution(self, labels_csv_path):
        counter = Counter()
        with open(labels_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counter[row["allsides_label"]] += 1

        assert counter["no_label"]   == EXPECTED_UNMATCHED
        assert counter["Center"]     == 101_203
        assert counter["Lean Left"]  == 53_533
        assert counter["Lean Right"] == 21_363
        assert counter["Right"]      == 12_892
        assert counter["Left"]       == 12_065
        assert counter["Mixed"]      == 2_499


# ---------------------------------------------------------------------------
# Labeled folders integrity
# ---------------------------------------------------------------------------

class TestLabeledFoldersIntegrity:

    def test_all_expected_folders_exist(self, labeled_dir):
        for folder in EXPECTED_FOLDERS:
            assert (labeled_dir / folder).exists(), f"Missing folder: {folder}"

    def test_no_unexpected_folders(self, labeled_dir):
        actual = {d.name for d in labeled_dir.iterdir() if d.is_dir()}
        unexpected = actual - EXPECTED_FOLDERS
        assert not unexpected, f"Unexpected folders found: {unexpected}"

    def test_total_file_count_matches_expected(self, labeled_dir):
        total = sum(len(list(d.iterdir())) for d in labeled_dir.iterdir() if d.is_dir())
        assert total == EXPECTED_TOTAL, (
            f"Expected {EXPECTED_TOTAL} total files in labeled folders, got {total}"
        )

    def test_folder_counts_match_csv_label_counts(self, labeled_dir, labels_csv_path):
        # Count files per folder
        folder_counts = {}
        for d in labeled_dir.iterdir():
            if d.is_dir():
                folder_counts[d.name] = len(list(d.iterdir()))

        # Count labels from CSV
        csv_counts = Counter()
        with open(labels_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                csv_counts[row["allsides_label"]] += 1

        label_to_folder = {v: k for k, v in {
            "Left":       "Left",
            "Lean Left":  "Lean_Left",
            "Center":     "Center",
            "Lean Right": "Lean_Right",
            "Right":      "Right",
            "Mixed":      "Mixed",
            "no_label":   "no_label",
        }.items()}

        for label, folder in {
            "Left":       "Left",
            "Lean Left":  "Lean_Left",
            "Center":     "Center",
            "Lean Right": "Lean_Right",
            "Right":      "Right",
            "Mixed":      "Mixed",
            "no_label":   "no_label",
        }.items():
            assert folder_counts.get(folder, 0) == csv_counts[label], (
                f"Folder {folder} has {folder_counts.get(folder, 0)} files "
                f"but CSV has {csv_counts[label]} rows with label '{label}'"
            )

    def test_all_files_are_json(self, labeled_dir):
        non_json = []
        for d in labeled_dir.iterdir():
            if d.is_dir():
                for f in d.iterdir():
                    if f.suffix != ".json":
                        non_json.append(str(f))
        assert not non_json, f"Non-JSON files found in labeled_articles/: {non_json[:5]}"

    def test_no_label_folder_is_largest(self, labeled_dir):
        counts = {d.name: len(list(d.iterdir())) for d in labeled_dir.iterdir() if d.is_dir()}
        assert counts["no_label"] == max(counts.values()), (
            "Expected 'no_label' to be the largest folder"
        )


# ---------------------------------------------------------------------------
# Excel integrity
# ---------------------------------------------------------------------------

class TestExcelIntegrity:

    def test_excel_exists(self, excel_path):
        assert excel_path.is_file()

    def test_excel_two_sheets(self, excel_path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.load_workbook(excel_path)
        assert wb.sheetnames == ["Outlet Labels", "Match Summary"]

    def test_outlet_labels_row_count(self, excel_path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.load_workbook(excel_path)
        ws = wb["Outlet Labels"]
        # max_row includes header
        assert ws.max_row > 1000, "Expected at least 1000 outlet rows"

    def test_outlet_labels_headers(self, excel_path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.load_workbook(excel_path)
        ws = wb["Outlet Labels"]
        headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
        assert headers == ["source_name", "allsides_source_name", "allsides_label", "matched", "article_count"]

    def test_match_summary_total(self, excel_path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.load_workbook(excel_path)
        ws = wb["Match Summary"]
        summary = {row[0]: row[1] for row in ws.iter_rows(values_only=True) if row[0]}
        assert summary["Total articles"] == EXPECTED_TOTAL
        assert summary["Matched"]        == EXPECTED_MATCHED
        assert summary["Unmatched"]      == EXPECTED_UNMATCHED

    def test_match_rate_approximately_correct(self, excel_path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.load_workbook(excel_path)
        ws = wb["Match Summary"]
        summary = {row[0]: row[1] for row in ws.iter_rows(values_only=True) if row[0]}
        rate = summary["Match rate (%)"]
        assert 42.0 < rate < 44.0, f"Unexpected match rate: {rate}%"

    def test_outlet_article_counts_sum_to_total(self, excel_path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.load_workbook(excel_path)
        ws = wb["Outlet Labels"]
        total = sum(
            ws.cell(r, 5).value or 0
            for r in range(2, ws.max_row + 1)
        )
        assert total == EXPECTED_TOTAL, (
            f"Sum of article_count column ({total}) != expected total ({EXPECTED_TOTAL})"
        )

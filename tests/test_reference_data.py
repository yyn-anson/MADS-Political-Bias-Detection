import csv
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class ReferenceDataTests(unittest.TestCase):
    def test_article_outlet_lookup_counts(self):
        path = ROOT / "data" / "article_outlet_labels.csv"
        counts: Counter[str] = Counter()
        matched = total = 0
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(
                reader.fieldnames,
                [
                    "record_id",
                    "source_name",
                    "allsides_label",
                    "allsides_source_name",
                    "matched",
                ],
            )
            for row in reader:
                total += 1
                self.assertTrue(row["record_id"])
                counts[row["allsides_label"]] += 1
                matched += row["matched"] == "True"

        self.assertEqual(total, 473_989)
        self.assertEqual(matched, 203_555)
        self.assertEqual(
            counts,
            {
                "Left": 12_065,
                "Lean Left": 53_533,
                "Center": 101_203,
                "Lean Right": 21_363,
                "Right": 12_892,
                "Mixed": 2_499,
                "no_label": 270_434,
            },
        )

    def test_allsides_2025_reference_shape(self):
        path = ROOT / "data" / "allsides" / "AllSides_Rating.csv"
        counts: Counter[str] = Counter()
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            counts[row["allsides_media_bias_ratings/publication/media_bias_rating"]] += 1

        self.assertEqual(len(rows), 1_886)
        self.assertEqual(
            counts,
            {
                "Left": 134,
                "Lean Left": 293,
                "Center": 1_218,
                "Lean Right": 108,
                "Right": 101,
                "Mixed": 32,
            },
        )


if __name__ == "__main__":
    unittest.main()

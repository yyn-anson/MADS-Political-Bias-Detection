import csv
import json
import tempfile
import unittest
from pathlib import Path

from mads.custom_dataset import convert_custom_dataset, load_label_index


class CustomDatasetTests(unittest.TestCase):
    def _labels(self, directory: Path, rows: list[dict[str, str]]) -> Path:
        path = directory / "labels.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "record_id",
                    "source_name",
                    "allsides_label",
                    "allsides_source_name",
                    "matched",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_converts_all_valid_articles_and_maps_five_classes_to_three(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            articles = root / "raw"
            articles.mkdir()
            rows = [
                {
                    "record_id": "left-1",
                    "source_name": "left.example",
                    "allsides_label": "Lean Left",
                    "allsides_source_name": "Left Example",
                    "matched": "True",
                },
                {
                    "record_id": "center-1",
                    "source_name": "center.example",
                    "allsides_label": "Center",
                    "allsides_source_name": "Center Example",
                    "matched": "True",
                },
                {
                    "record_id": "right-1",
                    "source_name": "right.example",
                    "allsides_label": "Right",
                    "allsides_source_name": "Right Example",
                    "matched": "True",
                },
                {
                    "record_id": "mixed-1",
                    "source_name": "mixed.example",
                    "allsides_label": "Mixed",
                    "allsides_source_name": "Mixed Example",
                    "matched": "True",
                },
            ]
            labels = self._labels(root, rows)
            raw_records = [
                {"record_id": "left-1", "title": "L", "content_original": "L" * 100},
                {"record_id": "center-1", "title": "C", "content": "C" * 100},
                {"record_id": "right-1", "title": "R", "text": "R" * 100},
                {"record_id": "mixed-1", "title": "M", "content": "M" * 100},
                {
                    "record_id": "unknown-1",
                    "source_name": "unknown.example",
                    "content_original": "U" * 100,
                },
                {"record_id": "invalid-1", "content_original": "too short"},
            ]
            for index, record in enumerate(raw_records):
                (articles / f"{index}.json").write_text(json.dumps(record), encoding="utf-8")

            output = root / "normalized" / "articles.jsonl"
            errors = root / "logs" / "errors.csv"
            stats = convert_custom_dataset(articles, labels, output, errors_path=errors)
            converted = [json.loads(line) for line in output.read_text().splitlines()]
            by_id = {record["id"]: record for record in converted}

            self.assertEqual(stats.discovered, 6)
            self.assertEqual(stats.written, 5)
            self.assertEqual(stats.labeled, 3)
            self.assertEqual(stats.unlabeled, 2)
            self.assertEqual(stats.invalid, 1)
            self.assertEqual(stats.labels, {"Center": 1, "Left": 1, "Right": 1})
            self.assertEqual(by_id["left-1"]["label"], "Left")
            self.assertEqual(by_id["left-1"]["source"], "left.example")
            self.assertEqual(by_id["center-1"]["label"], "Center")
            self.assertEqual(by_id["right-1"]["label"], "Right")
            self.assertNotIn("label", by_id["mixed-1"])
            self.assertNotIn("label", by_id["unknown-1"])
            self.assertTrue(errors.is_file())

    def test_label_index_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {
                "record_id": "duplicate",
                "source_name": "example.com",
                "allsides_label": "Center",
                "allsides_source_name": "Example",
                "matched": "True",
            }
            labels = self._labels(root, [row, row])
            with self.assertRaisesRegex(ValueError, "duplicate record_id"):
                load_label_index(labels)


if __name__ == "__main__":
    unittest.main()

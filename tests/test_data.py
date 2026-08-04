import json
import tempfile
import unittest
from pathlib import Path

from mads.data import load_articles
from mads.types import BiasLabel

TEXT = (
    "This is a complete example article with enough text to satisfy strict input validation. "
    "It contains a second sentence so that accidental titles or snippets are not accepted."
)


class DataTests(unittest.TestCase):
    def test_jsonl_limit_stops_after_requested_articles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "articles.jsonl"
            records = [
                {"id": f"item-{index}", "text": f"Article {index} " + "x" * 100}
                for index in range(5)
            ]
            path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            articles = load_articles(path, limit=2)
            self.assertEqual([article.id for article in articles], ["item-0", "item-1"])

    def test_json_and_csv_are_loaded_in_filename_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "01.json").write_text(
                json.dumps({"id": "json-1", "title": "JSON", "text": TEXT, "label": "Left"}),
                encoding="utf-8",
            )
            (root / "02.csv").write_text(
                'id,title,text,label\ncsv-1,CSV,"' + TEXT + '",Center\n', encoding="utf-8"
            )
            articles = load_articles(root)
            self.assertEqual([item.id for item in articles], ["json-1", "csv-1"])
            self.assertIs(articles[1].label, BiasLabel.CENTER)

    def test_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicates.json"
            path.write_text(
                json.dumps([{"id": "same", "text": TEXT}, {"id": "same", "text": TEXT + " x"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicates"):
                load_articles(path)

    def test_short_snippet_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.json"
            path.write_text(json.dumps({"text": "Too short"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least 80"):
                load_articles(path)


if __name__ == "__main__":
    unittest.main()

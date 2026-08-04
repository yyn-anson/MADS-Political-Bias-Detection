import tempfile
import unittest

from mads.config import AppConfig
from mads.metrics import classification_metrics
from mads.reporting import write_report
from mads.types import Analysis, Article, ArticleResult, BiasLabel


def analysis(name):
    return Analysis(
        agent=name,
        score=0,
        label=BiasLabel.CENTER,
        understanding="summary",
        reasoning="balanced framing",
        evidence=["evidence"],
        probabilities={"Left": 0.1, "Center": 0.8, "Right": 0.1},
        entropy=0.639,
        confidence_source="token_logprobs",
    )


class ReportingTests(unittest.TestCase):
    def test_metrics_and_all_artifacts(self):
        article = Article(
            id="center-1",
            title="Balanced report",
            text="x" * 100,
            label=BiasLabel.CENTER,
        )
        analyses = {name: analysis(name) for name in ("analyst_a", "analyst_b", "analyst_c")}
        result = ArticleResult(
            article=article,
            route="unanimous",
            final_label=BiasLabel.CENTER,
            final_score=0,
            winning_agent=None,
            panel_unanimous=True,
            initial_analyses=analyses,
            final_analyses=analyses,
            debate=[],
            decision_trace=[{"event": "early_exit"}],
            embedding_backend="test",
            elapsed_seconds=1.0,
        )
        metrics = classification_metrics([result])
        self.assertEqual(metrics["accuracy"], 1.0)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = write_report(directory, [result], [], AppConfig(), input_path="test")
            for filename in ("report.html", "results.json", "results.jsonl", "summary.csv"):
                self.assertTrue((run_dir / filename).is_file(), filename)
            self.assertIn("MADS analysis report", (run_dir / "report.html").read_text())


if __name__ == "__main__":
    unittest.main()

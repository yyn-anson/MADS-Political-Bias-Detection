import unittest
import xml.etree.ElementTree as ET

from mads.outlets import (
    outlet_comparison_svg,
    outlet_metrics,
    score_distribution_svg,
    summarize_outlets,
)
from mads.types import Analysis, Article, ArticleResult, BiasLabel


def _analysis(name: str, score: float, label: BiasLabel) -> Analysis:
    return Analysis(
        agent=name,
        score=score,
        label=label,
        understanding="summary",
        reasoning="reason",
        evidence=["evidence"],
        probabilities={"Left": 0.1, "Center": 0.8, "Right": 0.1},
        entropy=0.64,
    )


def _result(identifier: str, source: str, expected: BiasLabel, score: float) -> ArticleResult:
    predicted = (
        BiasLabel.LEFT if score <= -1 else BiasLabel.RIGHT if score >= 1 else BiasLabel.CENTER
    )
    article = Article(
        id=identifier,
        title=identifier,
        text="x" * 100,
        source=source,
        label=expected,
    )
    analyses = {name: _analysis(name, score, predicted) for name in ("a", "b", "c")}
    return ArticleResult(
        article=article,
        route="unanimous",
        final_label=predicted,
        final_score=score,
        winning_agent=None,
        panel_unanimous=True,
        initial_analyses=analyses,
        final_analyses=analyses,
        debate=[],
        decision_trace=[],
        embedding_backend="test",
        elapsed_seconds=0.1,
    )


class OutletTests(unittest.TestCase):
    def test_summaries_metrics_and_valid_svg(self):
        results = [
            _result("l1", "Left & Local", BiasLabel.LEFT, -2.0),
            _result("l2", "Left & Local", BiasLabel.LEFT, -1.0),
            _result("c1", "Center News", BiasLabel.CENTER, 0.2),
            _result("r1", "Right News", BiasLabel.RIGHT, 1.5),
        ]
        summaries = summarize_outlets(results)
        metrics = outlet_metrics(summaries)

        self.assertEqual(
            [summary.source for summary in summaries],
            [
                "Left & Local",
                "Center News",
                "Right News",
            ],
        )
        self.assertEqual(summaries[0].mean_score, -1.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        for svg in (score_distribution_svg(summaries), outlet_comparison_svg(summaries)):
            ET.fromstring(svg)
            self.assertIn("Left &amp; Local", svg)


if __name__ == "__main__":
    unittest.main()

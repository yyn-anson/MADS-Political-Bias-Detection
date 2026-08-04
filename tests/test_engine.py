import unittest
from dataclasses import replace

from mads.config import AppConfig
from mads.engine import MADSAnalyzer
from mads.llm import Completion
from mads.types import Article, BiasLabel

ARTICLE = Article(
    id="test-article",
    title="Test article",
    text=(
        "The article describes a policy dispute in sufficient detail for the three analysts. "
        "It includes competing claims and enough context to pass the input validation threshold."
    ),
)


def payload(score, reasoning, probabilities, *, kind="analysis"):
    value = {
        "evidence": ["short article evidence"],
        "probabilities": probabilities,
        "final_prediction": "Left" if score <= -1 else "Right" if score >= 1 else "Center",
    }
    if kind == "analysis":
        value.update(
            {
                "article_understanding": "A short factual summary.",
                "reasoning": reasoning,
                "score": score,
            }
        )
    elif kind == "challenge":
        value.update(
            {
                "acknowledgement": "The other view has one valid point.",
                "argument": reasoning,
                "adjusted_score": score,
            }
        )
    else:
        value.update(
            {
                "acknowledgement": "The challenge identifies relevant evidence.",
                "counterargument": reasoning,
                "reasoning": reasoning,
                "adjusted_score": score,
            }
        )
    return value


class FakeBackend:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def available_models(self):
        return ["qwen2.5vl:3b"]

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if not self.payloads:
            raise AssertionError("unexpected model call")
        value = self.payloads.pop(0)
        return Completion(
            payload=value,
            raw_text="{}",
            class_probabilities=value["probabilities"],
            confidence_source="token_logprobs",
            latency_seconds=0.01,
            usage={"total_tokens": 10},
        )


class StubEmbedder:
    name = "stub-embedder"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "left reason" in lowered:
                vectors.append([1.0, 0.0])
            elif "right reason" in lowered:
                vectors.append([-1.0, 0.0])
            elif "center reason" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 1.0])
        return vectors


def serial_config():
    base = AppConfig()
    return replace(base, runtime=replace(base.runtime, parallel_initial_analysis=False))


class EngineTests(unittest.TestCase):
    def test_unanimous_early_exit_uses_mean(self):
        backend = FakeBackend(
            [
                payload(-1, "left reason a", {"Left": 0.8, "Center": 0.15, "Right": 0.05}),
                payload(-2, "left reason b", {"Left": 0.9, "Center": 0.08, "Right": 0.02}),
                payload(-3, "left reason c", {"Left": 0.95, "Center": 0.04, "Right": 0.01}),
            ]
        )
        result = MADSAnalyzer(serial_config(), backend, StubEmbedder()).analyze_article(ARTICLE)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(result.route, "unanimous")
        self.assertEqual(result.final_score, -2.0)
        self.assertIs(result.final_label, BiasLabel.LEFT)
        self.assertEqual(result.debate, [])

    def test_majority_also_triggers_debate(self):
        backend = FakeBackend(
            [
                payload(-2, "left reason a", {"Left": 0.95, "Center": 0.04, "Right": 0.01}),
                payload(-1, "left reason b", {"Left": 0.7, "Center": 0.2, "Right": 0.1}),
                payload(2, "right reason c", {"Left": 0.05, "Center": 0.1, "Right": 0.85}),
                payload(
                    -2,
                    "left reason challenge",
                    {"Left": 0.9, "Center": 0.08, "Right": 0.02},
                    kind="challenge",
                ),
                payload(
                    -1,
                    "left reason response",
                    {"Left": 0.75, "Center": 0.2, "Right": 0.05},
                    kind="response",
                ),
            ]
        )
        result = MADSAnalyzer(serial_config(), backend, StubEmbedder()).analyze_article(ARTICLE)
        self.assertEqual(result.route, "majority_debate")
        self.assertEqual(len(result.debate), 1)
        self.assertEqual(result.debate[0].termination, "pair_consensus")
        self.assertTrue(result.panel_unanimous)
        self.assertIs(result.final_label, BiasLabel.LEFT)

    def test_all_different_uses_most_divergent_pair(self):
        backend = FakeBackend(
            [
                payload(-2, "left reason", {"Left": 0.8, "Center": 0.1, "Right": 0.1}),
                payload(0, "center reason", {"Left": 0.1, "Center": 0.8, "Right": 0.1}),
                payload(2, "right reason", {"Left": 0.1, "Center": 0.1, "Right": 0.8}),
                payload(
                    0,
                    "center reason challenge",
                    {"Left": 0.1, "Center": 0.8, "Right": 0.1},
                    kind="challenge",
                ),
                payload(
                    0,
                    "center reason response",
                    {"Left": 0.1, "Center": 0.85, "Right": 0.05},
                    kind="response",
                ),
            ]
        )
        result = MADSAnalyzer(serial_config(), backend, StubEmbedder()).analyze_article(ARTICLE)
        route = next(item for item in result.decision_trace if item.get("event") == "route")
        self.assertEqual(route["stage_1_pair"], ["analyst_a", "analyst_c"])
        self.assertEqual(result.route, "all_different_debate")
        self.assertEqual(len(result.debate), 1)
        self.assertTrue(result.panel_unanimous)
        self.assertIs(result.final_label, BiasLabel.CENTER)


if __name__ == "__main__":
    unittest.main()

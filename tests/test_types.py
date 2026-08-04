import math
import unittest

from mads.types import (
    Analysis,
    BiasLabel,
    normalize_probabilities,
    prediction_entropy,
    score_to_label,
)


class TypeTests(unittest.TestCase):
    def test_score_boundaries_follow_paper(self):
        self.assertIs(score_to_label(-1.0), BiasLabel.LEFT)
        self.assertIs(score_to_label(-0.999), BiasLabel.CENTER)
        self.assertIs(score_to_label(0.999), BiasLabel.CENTER)
        self.assertIs(score_to_label(1.0), BiasLabel.RIGHT)

    def test_probabilities_are_normalized(self):
        result = normalize_probabilities({"Left": 2, "Center": 1, "Right": 1}, BiasLabel.LEFT)
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertEqual(result["Left"], 0.5)

    def test_lower_entropy_means_higher_confidence(self):
        certain = prediction_entropy({"Left": 0.9, "Center": 0.05, "Right": 0.05})
        uncertain = prediction_entropy({"Left": 1 / 3, "Center": 1 / 3, "Right": 1 / 3})
        self.assertLess(certain, uncertain)
        self.assertAlmostEqual(uncertain, math.log(3), places=6)

    def test_inconsistent_score_and_text_label_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            Analysis.from_payload(
                "a",
                {
                    "score": -2,
                    "reasoning": "The narration consistently favors Democratic arguments.",
                    "final_prediction": "Right",
                    "probabilities": {"Left": 0.8, "Center": 0.1, "Right": 0.1},
                },
                threshold=1,
            )


if __name__ == "__main__":
    unittest.main()

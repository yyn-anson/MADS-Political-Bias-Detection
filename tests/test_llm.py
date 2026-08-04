import unittest

from mads.llm import (
    _encode_semantic_score,
    _score_and_prediction_agree,
    probabilities_from_logprobs,
)


class LogprobTests(unittest.TestCase):
    def test_extracts_prediction_token_distribution(self):
        content = [
            {"token": "{", "logprob": -0.1, "top_logprobs": []},
            {
                "token": '"Left"',
                "logprob": -0.2,
                "top_logprobs": [
                    {"token": '"Left"', "logprob": -0.2},
                    {"token": '"Center"', "logprob": -1.5},
                    {"token": '"Right"', "logprob": -3.0},
                ],
            },
        ]
        probabilities = probabilities_from_logprobs(content, "Left")
        self.assertIsNotNone(probabilities)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities["Left"], probabilities["Center"])
        self.assertGreater(probabilities["Center"], probabilities["Right"])

    def test_missing_alternatives_receive_small_floor(self):
        content = [{"token": "Right", "logprob": -0.1, "top_logprobs": []}]
        probabilities = probabilities_from_logprobs(content, "Right")
        self.assertGreater(probabilities["Right"], 0.98)
        self.assertGreater(probabilities["Left"], 0)

    def test_score_prediction_consistency(self):
        self.assertTrue(_score_and_prediction_agree({"score": 0, "final_prediction": "Center"}))
        self.assertFalse(_score_and_prediction_agree({"score": -3, "final_prediction": "Center"}))

    def test_semantic_score_encoding_avoids_numeric_grammar_bug(self):
        center = {
            "article_understanding": "summary",
            "lean_strength": "Neutral",
            "final_prediction": "Center",
        }
        right = {"lean_strength": "Moderate", "final_prediction": "Right"}
        _encode_semantic_score(center)
        _encode_semantic_score(right)
        self.assertEqual(center["score"], 0)
        self.assertEqual(right["adjusted_score"], 2)


if __name__ == "__main__":
    unittest.main()

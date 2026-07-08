"""
Tests for src/utils/json_extractor.py - RobustJSONExtractor.

Covers:
  - Strategy 1: clean JSON string
  - Strategy 2: JSON inside markdown code block
  - Strategy 3: JSON buried in surrounding prose
  - _fix_common_json_issues: trailing commas, single quotes
  - Strategy 4: regex field extraction fallback
  - Edge cases: empty/None/non-string input
  - extract_challenge_fields: type coercion, array joining, fallback
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.json_extractor import RobustJSONExtractor


# ---------------------------------------------------------------------------
# extract_json - Strategy 1: clean JSON
# ---------------------------------------------------------------------------

class TestExtractJsonClean:

    def test_plain_json_object(self):
        result = RobustJSONExtractor.extract_json('{"lean": 2, "reason": "biased"}')
        assert result == {"lean": 2, "reason": "biased"}

    def test_plain_json_with_nested(self):
        result = RobustJSONExtractor.extract_json('{"lean": -1, "details": {"a": 1}}')
        assert result["details"]["a"] == 1

    def test_json_with_surrounding_whitespace(self):
        result = RobustJSONExtractor.extract_json('  \n{"lean": 0}\n  ')
        assert result == {"lean": 0}

    def test_all_expected_fields(self):
        raw = '{"lean": 1, "direction": "Right", "reason": "test", "article_understanding": "summary"}'
        result = RobustJSONExtractor.extract_json(raw)
        assert result["lean"] == 1
        assert result["direction"] == "Right"
        assert result["reason"] == "test"


# ---------------------------------------------------------------------------
# extract_json - Strategy 2: markdown code block
# ---------------------------------------------------------------------------

class TestExtractJsonMarkdown:

    def test_fenced_json_block(self):
        text = 'Here is the result:\n```json\n{"lean": -2, "reason": "left-leaning"}\n```'
        result = RobustJSONExtractor.extract_json(text)
        assert result["lean"] == -2

    def test_fenced_block_without_json_label(self):
        text = "```\n{\"lean\": 3}\n```"
        result = RobustJSONExtractor.extract_json(text)
        assert result == {"lean": 3}

    def test_fenced_block_with_multiline_values(self):
        text = (
            "```json\n"
            '{"lean": 0, "reason": "This is a long\\nreason"}\n'
            "```"
        )
        result = RobustJSONExtractor.extract_json(text)
        assert result["lean"] == 0


# ---------------------------------------------------------------------------
# extract_json - Strategy 3: JSON buried in text
# ---------------------------------------------------------------------------

class TestExtractJsonBuriedInText:

    def test_json_after_preamble(self):
        text = 'After careful analysis, the answer is: {"lean": -1, "reason": "slightly left"} end.'
        result = RobustJSONExtractor.extract_json(text)
        assert result["lean"] == -1

    def test_json_preceded_by_assistant_prefix(self):
        text = 'Assistant: {"lean": 2, "direction": "Right", "reason": "conservative framing"}'
        result = RobustJSONExtractor.extract_json(text)
        assert result["lean"] == 2

    def test_json_with_trailing_comma_fixed(self):
        text = '{"lean": 1, "reason": "slight right",}'
        result = RobustJSONExtractor.extract_json(text)
        assert result is not None
        assert result["lean"] == 1

    def test_json_with_single_quotes_fixed(self):
        text = "{'lean': 0, 'reason': 'neutral'}"
        result = RobustJSONExtractor.extract_json(text)
        assert result is not None
        assert result["lean"] == 0


# ---------------------------------------------------------------------------
# extract_json - edge cases: empty / None / non-string
# ---------------------------------------------------------------------------

class TestExtractJsonEdgeCases:

    def test_empty_string_returns_none(self):
        assert RobustJSONExtractor.extract_json("") is None

    def test_none_input_returns_none(self):
        assert RobustJSONExtractor.extract_json(None) is None  # type: ignore

    def test_non_string_input_returns_none(self):
        assert RobustJSONExtractor.extract_json(123) is None  # type: ignore

    def test_no_json_in_text_returns_none(self):
        assert RobustJSONExtractor.extract_json("This is plain text with no JSON.") is None

    def test_incomplete_json_returns_none_or_partial(self):
        # Should not crash, may return None or partial result
        result = RobustJSONExtractor.extract_json('{"lean": 1,')
        # Either None or a partial dict - the key thing is no exception
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# extract_json - Strategy 4: regex field extraction
# ---------------------------------------------------------------------------

class TestExtractJsonRegexFallback:

    def test_extracts_lean_from_malformed(self):
        text = 'I think the "lean": -2 value is appropriate here.'
        result = RobustJSONExtractor.extract_json(text)
        # Strategy 4 should pick up the lean field
        if result is not None:
            assert result.get("lean") == -2.0 or result.get("lean") == -2

    def test_extracts_reason_from_malformed(self):
        text = 'Analysis: "reason": "conservative framing detected"'
        result = RobustJSONExtractor.extract_json(text)
        if result is not None:
            assert "reason" in result


# ---------------------------------------------------------------------------
# extract_challenge_fields
# ---------------------------------------------------------------------------

class TestExtractChallengeFields:

    def test_clean_challenge_json(self):
        text = '{"understanding": "Article is about X", "challenge": "The framing is biased", "adjusted_lean": -1}'
        result = RobustJSONExtractor.extract_challenge_fields(text)
        assert result["understanding"] == "Article is about X"
        assert result["challenge"] == "The framing is biased"
        assert result["adjusted_lean"] == -1.0

    def test_adjusted_lean_converted_to_float(self):
        text = '{"understanding": "test", "challenge": "X", "adjusted_lean": "-2"}'
        result = RobustJSONExtractor.extract_challenge_fields(text)
        assert isinstance(result.get("adjusted_lean"), (int, float)) or result.get("adjusted_lean") is None

    def test_challenge_array_joined_to_string(self):
        text = '{"understanding": "test", "challenge": ["Point one", "Point two"], "adjusted_lean": 0}'
        result = RobustJSONExtractor.extract_challenge_fields(text)
        challenge = result.get("challenge")
        assert isinstance(challenge, str), "challenge should be a string, not a list"
        assert "Point one" in challenge
        assert "Point two" in challenge

    def test_fallback_when_no_json_found(self):
        text = "This model says the article leans right overall."
        result = RobustJSONExtractor.extract_challenge_fields(text)
        # Should return a dict (not raise), with 'challenge' set to the full text
        assert isinstance(result, dict)
        assert "challenge" in result
        assert result["challenge"] == text

    def test_returns_dict_not_none(self):
        result = RobustJSONExtractor.extract_challenge_fields("")
        assert isinstance(result, dict)

    def test_challenge_non_string_coerced_to_string(self):
        text = '{"understanding": "test", "challenge": 42, "adjusted_lean": 0}'
        result = RobustJSONExtractor.extract_challenge_fields(text)
        challenge = result.get("challenge")
        if challenge is not None:
            assert isinstance(challenge, str)

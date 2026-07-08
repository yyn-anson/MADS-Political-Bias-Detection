"""
Regression tests: a model that outputs a lean of exactly 0 (Neutral/Center)
must be handled like any other score.

An earlier version used `parsed.get("lean") or parsed.get("final_score")`,
which treats 0 as falsy - predict() then raised "missing 'lean' field" and the
article was dropped, silently discarding Center-scored articles. The same
pattern in generate_discussion_response() made agents unable to revise their
position to Center during discussion.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.llama32_labeler import LlamaLabeler
from src.models.qwen3_labeler import QwenLabeler
from src.models.mistral_labeler import MistralLabeler
from src.models.gptoss_labeler import GPTOSSLabeler


def _make_labeler(cls):
    """Construct a labeler of the given class with dummy connection settings."""
    kwargs = dict(base_url="http://localhost:9/v1", model_id="test/model", api_key="k")
    return cls(**kwargs)


def _mock_client(content: str) -> MagicMock:
    """Build a mock OpenAI client whose chat completion returns the given content."""
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return client


ALL_LABELERS = [LlamaLabeler, QwenLabeler, MistralLabeler, GPTOSSLabeler]


@pytest.mark.parametrize("cls", ALL_LABELERS)
class TestZeroLeanPredict:

    def test_predict_accepts_zero_lean(self, cls):
        labeler = _make_labeler(cls)
        labeler._client = _mock_client(json.dumps({
            "article_understanding": "An article about infrastructure funding.",
            "reason": "Balanced coverage of both parties with neutral language.",
            "lean": 0,
        }))
        result = labeler.predict("word " * 200)
        assert result["lean"] == 0
        assert result["direction"] == "Center"

    def test_predict_still_rejects_missing_lean(self, cls):
        labeler = _make_labeler(cls)
        labeler._client = _mock_client(json.dumps({
            "article_understanding": "Summary.",
            "reason": "Reasoning.",
        }))
        with pytest.raises(ValueError):
            labeler.predict("word " * 200)


@pytest.mark.parametrize("cls", ALL_LABELERS)
class TestZeroLeanDiscussionResponse:

    def test_response_final_lean_zero_is_applied(self, cls):
        labeler = _make_labeler(cls)
        labeler._client = _mock_client(json.dumps({
            "acknowledgment": "The challenger raises valid points.",
            "counter_argument": "On reflection the article is balanced.",
            "final_lean": 0,
            "reason": "Revised to neutral after considering the challenge.",
        }))
        own = {"score": 2, "direction": "Right", "reason": "initial"}
        challenger = {"score": -2, "direction": "Left", "reason": "challenger"}
        _, result = labeler.generate_discussion_response(
            article_content="word " * 200,
            conversation_history="",
            challenge="Your reading overweights one quote.",
            own_analysis=own,
            challenger_analysis=challenger,
        )
        assert result["lean"] == 0, "final_lean of 0 must override the previous score"

    def test_response_missing_lean_keeps_own_score(self, cls):
        labeler = _make_labeler(cls)
        labeler._client = _mock_client(json.dumps({
            "acknowledgment": "Noted.",
            "counter_argument": "I maintain my analysis.",
            "reason": "No score change.",
        }))
        own = {"score": 2, "direction": "Right", "reason": "initial"}
        challenger = {"score": -2, "direction": "Left", "reason": "challenger"}
        _, result = labeler.generate_discussion_response(
            article_content="word " * 200,
            conversation_history="",
            challenge="Challenge text.",
            own_analysis=own,
            challenger_analysis=challenger,
        )
        assert result["lean"] == 2

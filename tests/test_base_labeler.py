"""
Tests for src/models/base_labeler.py.

Covers:
  - Abstract class enforcement (cannot instantiate without implementing required methods)
  - score_to_direction boundary conditions
  - label_articles_batch default sequential behaviour + error propagation
  - unload_model default cleanup
  - generate_discussion_challenge/response raise NotImplementedError
  - __repr__ format
"""

import gc
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.base_labeler import BaseLabeler


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------

class _MinimalLabeler(BaseLabeler):
    def load_model(self) -> None:
        self.model = object()  # sentinel

    def predict(self, article_text: str) -> Dict[str, Any]:
        return {
            "lean": 1,
            "direction": self.score_to_direction(1),
            "reason": "test",
        }


class _RaisingLabeler(BaseLabeler):
    """Predict always raises — used to verify error propagation."""
    def load_model(self) -> None:
        pass

    def predict(self, article_text: str) -> Dict[str, Any]:
        raise RuntimeError("deliberate failure")


# ---------------------------------------------------------------------------
# Abstract class enforcement
# ---------------------------------------------------------------------------

class TestAbstractEnforcement:

    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseLabeler(model_name="x")  # type: ignore[abstract]

    def test_must_implement_predict(self):
        class _NoPredict(BaseLabeler):
            def load_model(self) -> None:
                pass
            # predict NOT implemented

        with pytest.raises(TypeError):
            _NoPredict(model_name="x")

    def test_must_implement_load_model(self):
        class _NoLoad(BaseLabeler):
            def predict(self, article_text: str) -> Dict[str, Any]:
                return {}
            # load_model NOT implemented

        with pytest.raises(TypeError):
            _NoLoad(model_name="x")

    def test_concrete_subclass_instantiates(self):
        labeler = _MinimalLabeler(model_name="test/model")
        assert labeler.model_name == "test/model"


# ---------------------------------------------------------------------------
# score_to_direction
# ---------------------------------------------------------------------------

class TestScoreToDirection:

    @pytest.mark.parametrize("score,expected", [
        (-3,   "Left"),
        (-2,   "Left"),
        (-1,   "Left"),     # boundary: exactly -1 → Left
        (-0.5, "Center"),   # between -1 and 1 → Center
        (0,    "Center"),
        (0.5,  "Center"),
        (1,    "Right"),    # boundary: exactly 1 → Right
        (2,    "Right"),
        (3,    "Right"),
    ])
    def test_score_boundary(self, score, expected):
        assert BaseLabeler.score_to_direction(score) == expected

    def test_slightly_below_minus_one_is_left(self):
        assert BaseLabeler.score_to_direction(-1.0001) == "Left"

    def test_slightly_above_one_is_right(self):
        assert BaseLabeler.score_to_direction(1.0001) == "Right"


# ---------------------------------------------------------------------------
# label_articles_batch — default sequential implementation
# ---------------------------------------------------------------------------

class TestLabelArticlesBatch:

    def test_calls_predict_for_each_prompt(self):
        labeler = _MinimalLabeler(model_name="test/model")
        results = labeler.label_articles_batch(["article A", "article B", "article C"])
        assert len(results) == 3

    def test_returns_predict_results_in_order(self):
        call_log = []

        class _OrderedLabeler(BaseLabeler):
            def load_model(self) -> None:
                pass

            def predict(self, article_text: str) -> Dict[str, Any]:
                call_log.append(article_text)
                return {"lean": len(call_log), "direction": "Center", "reason": ""}

        labeler = _OrderedLabeler(model_name="test/model")
        results = labeler.label_articles_batch(["first", "second", "third"])
        assert call_log == ["first", "second", "third"]
        assert results[0]["lean"] == 1
        assert results[2]["lean"] == 3

    def test_predict_exception_propagates(self):
        """Exceptions from predict() must NOT be swallowed — they must propagate."""
        labeler = _RaisingLabeler(model_name="test/model")
        with pytest.raises(RuntimeError, match="deliberate failure"):
            labeler.label_articles_batch(["some article"])

    def test_empty_batch_returns_empty_list(self):
        labeler = _MinimalLabeler(model_name="test/model")
        assert labeler.label_articles_batch([]) == []


# ---------------------------------------------------------------------------
# unload_model
# ---------------------------------------------------------------------------

class TestUnloadModel:

    def test_sets_model_to_none(self):
        labeler = _MinimalLabeler(model_name="test/model")
        labeler.load_model()
        assert labeler.model is not None
        labeler.unload_model()
        assert labeler.model is None

    def test_sets_tokenizer_to_none(self):
        labeler = _MinimalLabeler(model_name="test/model")
        labeler.tokenizer = object()  # simulate a loaded tokenizer
        labeler.unload_model()
        assert labeler.tokenizer is None

    def test_safe_when_model_already_none(self):
        labeler = _MinimalLabeler(model_name="test/model")
        # model is None by default — must not raise
        labeler.unload_model()
        assert labeler.model is None

    def test_calls_torch_empty_cache_if_available(self):
        labeler = _MinimalLabeler(model_name="test/model")
        labeler.model = object()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            labeler.unload_model()
        mock_torch.cuda.empty_cache.assert_called_once()


# ---------------------------------------------------------------------------
# Discussion stubs raise NotImplementedError
# ---------------------------------------------------------------------------

class TestDiscussionNotImplemented:

    def test_generate_discussion_challenge_raises(self):
        labeler = _MinimalLabeler(model_name="test/model")
        with pytest.raises(NotImplementedError, match="_MinimalLabeler does not support"):
            labeler.generate_discussion_challenge(
                article_content="...",
                conversation_history="",
                own_analysis={},
                target_analysis={},
            )

    def test_generate_discussion_response_raises(self):
        labeler = _MinimalLabeler(model_name="test/model")
        with pytest.raises(NotImplementedError, match="_MinimalLabeler does not support"):
            labeler.generate_discussion_response(
                article_content="...",
                conversation_history="",
                challenge="...",
                own_analysis={},
                challenger_analysis={},
            )


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_class_name(self):
        labeler = _MinimalLabeler(model_name="org/model-7b", batch_size=4)
        r = repr(labeler)
        assert "_MinimalLabeler" in r

    def test_repr_contains_model_name(self):
        labeler = _MinimalLabeler(model_name="org/model-7b", batch_size=4)
        r = repr(labeler)
        assert "org/model-7b" in r

    def test_repr_contains_batch_size(self):
        labeler = _MinimalLabeler(model_name="org/model-7b", batch_size=4)
        r = repr(labeler)
        assert "4" in r

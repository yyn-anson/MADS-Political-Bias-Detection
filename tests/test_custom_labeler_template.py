"""
Tests for src/models/custom_labeler_template.py.

Uses unittest.mock to avoid any real HTTP calls to a vLLM server.

Covers:
  - CustomLabeler is a proper BaseLabeler subclass
  - load_model() creates an OpenAI client and verifies server reachability
  - load_model() raises RuntimeError when the server is unreachable
  - predict() correct output contract (lean, direction, reason, raw_response)
  - predict() raises on API failure
  - predict() raises on unparseable JSON output
  - predict() raises on missing 'lean' field
  - predict() clamps out-of-range lean values
  - predict() calls load_model() lazily when client is None
  - unload_model() drops the client
  - label_articles_batch() processes multiple articles
  - label_articles_batch() propagates errors
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.base_labeler import BaseLabeler
from src.models.custom_labeler_template import CustomLabeler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_completion(content: str) -> MagicMock:
    """Build a fake openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_labeler(client_response: str | None = None) -> CustomLabeler:
    """
    Return a CustomLabeler with a pre-connected mock OpenAI client.

    client_response: the string the mock client will return for
                     chat.completions.create(); defaults to a valid JSON result.
    """
    if client_response is None:
        client_response = json.dumps({
            "lean": 1,
            "reason": "test reason",
            "article_understanding": "test understanding",
        })

    labeler = CustomLabeler.__new__(CustomLabeler)
    labeler.model_name = "test/model"
    labeler.batch_size = 1
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion(client_response)
    labeler._client = mock_client
    labeler._base_url = "http://localhost:8001/v1"
    labeler._api_key = "token-abc123"
    labeler._temperature = 0.7
    labeler._max_tokens = 4096
    labeler._top_p = 0.9
    return labeler


# ---------------------------------------------------------------------------
# Subclass relationship
# ---------------------------------------------------------------------------

class TestInheritance:

    def test_is_base_labeler_subclass(self):
        assert issubclass(CustomLabeler, BaseLabeler)

    def test_implements_load_model(self):
        assert callable(getattr(CustomLabeler, "load_model", None))

    def test_implements_predict(self):
        assert callable(getattr(CustomLabeler, "predict", None))

    def test_implements_unload_model(self):
        assert callable(getattr(CustomLabeler, "unload_model", None))


# ---------------------------------------------------------------------------
# load_model()
# ---------------------------------------------------------------------------

class TestLoadModel:

    def test_creates_client_on_success(self):
        labeler = CustomLabeler.__new__(CustomLabeler)
        labeler.model_name = "test/model"
        labeler.batch_size = 1
        labeler._base_url = "http://localhost:8001/v1"
        labeler._api_key = "token-abc123"
        labeler._client = None

        mock_client = MagicMock()
        with patch("src.models.custom_labeler_template.OpenAI", return_value=mock_client):
            labeler.load_model()

        assert labeler._client is mock_client
        mock_client.models.list.assert_called_once()

    def test_raises_runtime_error_when_server_unreachable(self):
        labeler = CustomLabeler.__new__(CustomLabeler)
        labeler.model_name = "test/model"
        labeler.batch_size = 1
        labeler._base_url = "http://localhost:8001/v1"
        labeler._api_key = "token-abc123"
        labeler._client = None

        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("Connection refused")

        with patch("src.models.custom_labeler_template.OpenAI", return_value=mock_client):
            with pytest.raises(RuntimeError, match="not reachable"):
                labeler.load_model()

        assert labeler._client is None


# ---------------------------------------------------------------------------
# predict() - correct output contract
# ---------------------------------------------------------------------------

class TestPredict:

    def test_returns_lean_int(self):
        labeler = _make_labeler()
        result = labeler.predict("Some article text")
        assert isinstance(result["lean"], int)

    def test_returns_direction_string(self):
        labeler = _make_labeler()
        result = labeler.predict("Some article text")
        assert result["direction"] in {"Left", "Center", "Right"}

    def test_returns_reason_string(self):
        labeler = _make_labeler()
        result = labeler.predict("Some article text")
        assert isinstance(result["reason"], str)

    def test_returns_raw_response(self):
        labeler = _make_labeler()
        result = labeler.predict("Some article text")
        assert "raw_response" in result
        assert isinstance(result["raw_response"], str)

    def test_no_error_key_in_success_response(self):
        labeler = _make_labeler()
        result = labeler.predict("Some article text")
        assert "error" not in result, (
            "predict() must not include an 'error' key on success - "
            "use exceptions for failure signalling"
        )

    def test_lean_positive_maps_to_right(self):
        payload = json.dumps({"lean": 2, "reason": "r", "article_understanding": "u"})
        labeler = _make_labeler(client_response=payload)
        result = labeler.predict("...")
        assert result["lean"] == 2
        assert result["direction"] == "Right"

    def test_lean_negative_maps_to_left(self):
        payload = json.dumps({"lean": -2, "reason": "r", "article_understanding": "u"})
        labeler = _make_labeler(client_response=payload)
        result = labeler.predict("...")
        assert result["lean"] == -2
        assert result["direction"] == "Left"

    def test_lean_zero_maps_to_center(self):
        payload = json.dumps({"lean": 0, "reason": "r", "article_understanding": "u"})
        labeler = _make_labeler(client_response=payload)
        result = labeler.predict("...")
        assert result["lean"] == 0
        assert result["direction"] == "Center"

    def test_lean_clamped_above_3(self):
        payload = json.dumps({"lean": 10, "reason": "extreme"})
        labeler = _make_labeler(client_response=payload)
        result = labeler.predict("...")
        assert result["lean"] == 3

    def test_lean_clamped_below_minus_3(self):
        payload = json.dumps({"lean": -99, "reason": "extreme"})
        labeler = _make_labeler(client_response=payload)
        result = labeler.predict("...")
        assert result["lean"] == -3

    def test_lean_float_is_cast_to_int(self):
        payload = json.dumps({"lean": 1.7, "reason": "slight"})
        labeler = _make_labeler(client_response=payload)
        result = labeler.predict("...")
        assert isinstance(result["lean"], int)

    def test_article_text_forwarded_to_api(self):
        labeler = _make_labeler()
        labeler.predict("Unique article content XYZ123")
        call_args = labeler._client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else []
        if not messages:
            messages = call_args[1].get("messages", [])
        full_text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        assert "Unique article content XYZ123" in full_text


# ---------------------------------------------------------------------------
# predict() - error propagation
# ---------------------------------------------------------------------------

class TestPredictErrorPropagation:

    def test_api_error_raises_runtime_error(self):
        labeler = _make_labeler()
        labeler._client.chat.completions.create.side_effect = Exception("GPU OOM")
        with pytest.raises(RuntimeError, match="GPU OOM"):
            labeler.predict("article text")

    def test_unparseable_output_raises_value_error(self):
        labeler = _make_labeler(client_response="I cannot determine the bias.")
        with pytest.raises(ValueError, match="could not be parsed as JSON"):
            labeler.predict("article text")

    def test_missing_lean_field_raises_value_error(self):
        payload = json.dumps({"reason": "no lean key"})
        labeler = _make_labeler(client_response=payload)
        with pytest.raises(ValueError, match="missing required 'lean' field"):
            labeler.predict("article text")

    def test_loads_client_lazily_when_none(self):
        """predict() must call load_model() when self._client is None."""
        labeler = _make_labeler()
        labeler._client = None  # simulate not yet connected

        payload = json.dumps({"lean": 0, "reason": "x"})
        fresh_client = MagicMock()
        fresh_client.chat.completions.create.return_value = _mock_completion(payload)

        load_called = []

        def fake_load():
            """Record the call and attach a fresh mock client."""
            load_called.append(1)
            labeler._client = fresh_client

        # Patch on the instance so Python does not inject 'self'
        with patch.object(labeler, "load_model", side_effect=fake_load):
            result = labeler.predict("test article")

        assert len(load_called) == 1, "load_model() was not called"
        assert result["lean"] == 0


# ---------------------------------------------------------------------------
# unload_model()
# ---------------------------------------------------------------------------

class TestUnloadModel:

    def test_drops_client(self):
        labeler = _make_labeler()
        assert labeler._client is not None
        labeler.unload_model()
        assert labeler._client is None

    def test_unload_then_predict_reconnects(self):
        labeler = _make_labeler()
        labeler.unload_model()
        assert labeler._client is None

        payload = json.dumps({"lean": 1, "reason": "r"})
        fresh_client = MagicMock()
        fresh_client.chat.completions.create.return_value = _mock_completion(payload)

        with patch.object(CustomLabeler, "load_model", side_effect=lambda: setattr(labeler, "_client", fresh_client)):
            result = labeler.predict("text")

        assert result["lean"] == 1


# ---------------------------------------------------------------------------
# label_articles_batch()
# ---------------------------------------------------------------------------

class TestLabelArticlesBatch:

    def test_processes_multiple_articles(self):
        payloads = [
            json.dumps({"lean": i - 1, "reason": "r", "article_understanding": "u"})
            for i in range(3)
        ]
        labeler = _make_labeler()
        labeler._client.chat.completions.create.side_effect = [
            _mock_completion(p) for p in payloads
        ]
        results = labeler.label_articles_batch(["a", "b", "c"])
        assert len(results) == 3

    def test_batch_api_error_propagates(self):
        labeler = _make_labeler()
        labeler._client.chat.completions.create.side_effect = RuntimeError("batch failure")
        with pytest.raises(RuntimeError, match="batch failure"):
            labeler.label_articles_batch(["a", "b"])

    def test_batch_unparseable_response_raises(self):
        labeler = _make_labeler(client_response="not json at all")
        with pytest.raises(ValueError, match="could not be parsed as JSON"):
            labeler.label_articles_batch(["a"])

    def test_batch_missing_lean_raises(self):
        payload = json.dumps({"reason": "no lean"})
        labeler = _make_labeler(client_response=payload)
        with pytest.raises(ValueError, match="missing required 'lean' field"):
            labeler.label_articles_batch(["only one"])

    def test_batch_no_error_key_on_success(self):
        payload = json.dumps({"lean": 1, "reason": "r"})
        labeler = _make_labeler(client_response=payload)
        results = labeler.label_articles_batch(["a"])
        assert "error" not in results[0]

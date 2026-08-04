"""OpenAI-compatible local LLM client with structured output and token logprobs."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib import error, request


@dataclass
class Completion:
    payload: dict[str, Any]
    raw_text: str
    class_probabilities: dict[str, float] | None
    confidence_source: str
    latency_seconds: float
    usage: dict[str, int] = field(default_factory=dict)


class LLMBackend(Protocol):
    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        temperature: float,
        seed: int,
    ) -> Completion: ...

    def available_models(self) -> list[str]: ...


def _token_label(token: str) -> str | None:
    cleaned = re.sub(r"[^a-z]", "", token.lower())
    return cleaned.capitalize() if cleaned in {"left", "center", "right"} else None


def probabilities_from_logprobs(
    content_logprobs: list[Mapping[str, Any]], expected_label: str
) -> dict[str, float] | None:
    """Extract the label-token distribution at the final prediction position."""
    expected = expected_label.capitalize()
    for item in reversed(content_logprobs):
        if _token_label(str(item.get("token", ""))) != expected:
            continue
        label_logprobs: dict[str, float] = {}
        raw_top = list(item.get("top_logprobs") or [])
        raw_top.append({"token": item.get("token"), "logprob": item.get("logprob")})
        for candidate in raw_top:
            label = _token_label(str(candidate.get("token", "")))
            try:
                value = float(candidate.get("logprob"))
            except (TypeError, ValueError):
                continue
            if label:
                label_logprobs[label] = max(value, label_logprobs.get(label, -math.inf))
        if not label_logprobs:
            continue
        floor = min(label_logprobs.values()) - 5.0
        weights = {
            label: math.exp(label_logprobs.get(label, floor))
            for label in ("Left", "Center", "Right")
        }
        total = sum(weights.values())
        return {label: value / total for label, value in weights.items()}
    return None


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("model response must be a JSON object")
    return value


def _score_and_prediction_agree(payload: Mapping[str, Any]) -> bool:
    score = payload.get("score", payload.get("adjusted_score", payload.get("final_score")))
    prediction = str(payload.get("final_prediction", "")).strip().lower()
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return False
    expected = "left" if numeric <= -1 else "right" if numeric >= 1 else "center"
    return prediction == expected


def _encode_semantic_score(payload: dict[str, Any]) -> None:
    """Translate grammar-safe label/strength fields to the paper's signed score."""
    prediction = str(payload.get("final_prediction", "")).strip().lower()
    strength = str(payload.get("lean_strength", "")).strip().lower()
    if prediction not in {"left", "center", "right"}:
        return
    if prediction == "center":
        score = 0
    else:
        magnitude = {"neutral": 1, "slight": 1, "moderate": 2, "strong": 3}.get(strength)
        if magnitude is None:
            return
        score = -magnitude if prediction == "left" else magnitude
    key = "score" if "article_understanding" in payload else "adjusted_score"
    payload[key] = score


class OllamaOpenAIClient:
    """Small, dependency-free client for Ollama's `/v1/chat/completions` API."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: int = 240,
        retries: int = 2,
        max_output_tokens: int = 700,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.max_output_tokens = max_output_tokens

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        http_request = request.Request(
            f"{self.endpoint}{path}",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(f"LLM HTTP {exc.code}: {detail}")
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(
            f"local LLM request failed after {self.retries + 1} attempts: {last_error}"
        )

    def available_models(self) -> list[str]:
        http_request = request.Request(
            f"{self.endpoint}/models",
            headers={"Authorization": "Bearer ollama"},
        )
        try:
            with request.urlopen(http_request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"cannot reach Ollama at {self.endpoint}. Start the Ollama app and try again: {exc}"
            ) from exc
        return [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        temperature: float,
        seed: int,
    ) -> Completion:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": self.max_output_tokens,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "mads_response", "strict": True, "schema": schema},
            },
        }
        started = time.monotonic()
        response: dict[str, Any] = {}
        choice: dict[str, Any] = {}
        raw_text = ""
        payload: dict[str, Any] = {}
        for semantic_attempt in range(2):
            if semantic_attempt:
                body["seed"] = seed + 10_000
                body["messages"] = [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": user
                        + "\n\nCORRECTION: Re-evaluate and return complete JSON. Center must use "
                        "Neutral strength; directional predictions use Slight, Moderate, or Strong.",
                    },
                ]
            response = self._post("/chat/completions", body)
            try:
                choice = response["choices"][0]
                raw_text = choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"unexpected LLM response shape: {response}") from exc
            payload = _parse_json_object(raw_text)
            _encode_semantic_score(payload)
            if _score_and_prediction_agree(payload):
                break
        else:
            raise RuntimeError(
                "model returned contradictory score and final_prediction twice; "
                f"last response: {raw_text[:2000]}"
            )
        latency = time.monotonic() - started
        expected = str(payload.get("final_prediction", ""))
        logprob_items = (choice.get("logprobs") or {}).get("content") or []
        token_probabilities = probabilities_from_logprobs(logprob_items, expected)
        fallback = payload.get("probabilities")
        probabilities = token_probabilities or (
            dict(fallback) if isinstance(fallback, dict) else None
        )
        source = "token_logprobs" if token_probabilities else "model_reported"
        usage = {
            key: int(value)
            for key, value in (response.get("usage") or {}).items()
            if isinstance(value, (int, float))
        }
        return Completion(
            payload=payload,
            raw_text=raw_text,
            class_probabilities=probabilities,
            confidence_source=source,
            latency_seconds=latency,
            usage=usage,
        )

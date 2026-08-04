"""Validated domain types shared by the MADS pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class BiasLabel(str, Enum):
    LEFT = "Left"
    CENTER = "Center"
    RIGHT = "Right"

    @classmethod
    def parse(cls, value: object) -> BiasLabel:
        text = str(value).strip().lower()
        aliases = {
            "left": cls.LEFT,
            "lean left": cls.LEFT,
            "democrat": cls.LEFT,
            "democratic": cls.LEFT,
            "center": cls.CENTER,
            "centre": cls.CENTER,
            "neutral": cls.CENTER,
            "balanced": cls.CENTER,
            "right": cls.RIGHT,
            "lean right": cls.RIGHT,
            "republican": cls.RIGHT,
        }
        if text not in aliases:
            raise ValueError(f"unknown bias label: {value!r}")
        return aliases[text]


def score_to_label(score: float, threshold: float = 1.0) -> BiasLabel:
    if score <= -threshold:
        return BiasLabel.LEFT
    if score >= threshold:
        return BiasLabel.RIGHT
    return BiasLabel.CENTER


def clamp_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"score must be numeric, got {value!r}") from exc
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    return max(-3.0, min(3.0, score))


def normalize_probabilities(
    values: Mapping[str, object] | None, predicted: BiasLabel
) -> dict[str, float]:
    """Return a normalized Left/Center/Right distribution with a safe fallback."""
    parsed: dict[str, float] = {}
    for label in BiasLabel:
        raw = None
        if values:
            raw = values.get(label.value, values.get(label.value.lower()))
        try:
            number = max(0.0, float(raw))
        except (TypeError, ValueError):
            number = 0.0
        parsed[label.value] = number if math.isfinite(number) else 0.0

    total = sum(parsed.values())
    if total <= 0:
        parsed = {label.value: 0.1 for label in BiasLabel}
        parsed[predicted.value] = 0.8
        total = 1.0
    return {key: value / total for key, value in parsed.items()}


def prediction_entropy(probabilities: Mapping[str, float]) -> float:
    return -sum(p * math.log(p) for p in probabilities.values() if p > 0)


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    text: str
    source: str | None = None
    url: str | None = None
    published_at: str | None = None
    label: BiasLabel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["label"] = self.label.value if self.label else None
        if not include_text:
            result.pop("text", None)
        return result


@dataclass
class Analysis:
    agent: str
    score: float
    label: BiasLabel
    understanding: str
    reasoning: str
    evidence: list[str]
    probabilities: dict[str, float]
    entropy: float
    confidence_source: str = "model_reported"
    latency_seconds: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        agent: str,
        payload: Mapping[str, Any],
        *,
        threshold: float,
        probabilities: Mapping[str, object] | None = None,
        confidence_source: str = "model_reported",
        latency_seconds: float = 0.0,
        usage: Mapping[str, int] | None = None,
    ) -> Analysis:
        score_value = payload.get(
            "score", payload.get("adjusted_score", payload.get("final_score"))
        )
        score = clamp_score(score_value)
        label = score_to_label(score, threshold)
        reported_label = payload.get("final_prediction")
        if reported_label is not None and BiasLabel.parse(reported_label) is not label:
            raise ValueError(
                f"{agent} returned inconsistent score/prediction: "
                f"score {score} maps to {label.value}, not {reported_label}"
            )
        reason = str(
            payload.get("reasoning")
            or payload.get("reason")
            or payload.get("argument")
            or payload.get("counterargument")
            or ""
        ).strip()
        if not reason:
            raise ValueError(f"{agent} returned no reasoning")
        understanding = str(payload.get("article_understanding") or "").strip()
        evidence_raw = payload.get("evidence", [])
        if isinstance(evidence_raw, str):
            evidence_raw = [evidence_raw]
        evidence = [str(item).strip() for item in evidence_raw if str(item).strip()]
        distribution = normalize_probabilities(probabilities or payload.get("probabilities"), label)
        return cls(
            agent=agent,
            score=score,
            label=label,
            understanding=understanding,
            reasoning=reason,
            evidence=evidence,
            probabilities=distribution,
            entropy=prediction_entropy(distribution),
            confidence_source=confidence_source,
            latency_seconds=latency_seconds,
            usage=dict(usage or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["label"] = self.label.value
        return result


@dataclass
class DebateExchange:
    stage: str
    round: int
    challenger: str
    target: str
    challenge: str
    response: str
    challenger_before: dict[str, Any]
    challenger_after: dict[str, Any]
    target_before: dict[str, Any]
    target_after: dict[str, Any]
    similarity_to_previous: float | None
    termination: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArticleResult:
    article: Article
    route: str
    final_label: BiasLabel
    final_score: float
    winning_agent: str | None
    panel_unanimous: bool
    initial_analyses: dict[str, Analysis]
    final_analyses: dict[str, Analysis]
    debate: list[DebateExchange]
    decision_trace: list[dict[str, Any]]
    embedding_backend: str
    elapsed_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "article": self.article.to_dict(include_text=False),
            "route": self.route,
            "final_label": self.final_label.value,
            "final_score": self.final_score,
            "winning_agent": self.winning_agent,
            "panel_unanimous": self.panel_unanimous,
            "initial_analyses": {
                name: analysis.to_dict() for name, analysis in self.initial_analyses.items()
            },
            "final_analyses": {
                name: analysis.to_dict() for name, analysis in self.final_analyses.items()
            },
            "debate": [exchange.to_dict() for exchange in self.debate],
            "decision_trace": self.decision_trace,
            "embedding_backend": self.embedding_backend,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }

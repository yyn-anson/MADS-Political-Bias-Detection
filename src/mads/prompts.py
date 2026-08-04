"""Paper-aligned content-only prompts and structured-output schemas."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

SYSTEM_PROMPT = """You are an expert media analyst specializing in U.S. political bias detection.
Judge only the supplied article text. Do not infer bias from its publisher, URL, author, date,
or facts not present in the article.

Use this continuous scale:
-3 strong support for Democratic positions or strong criticism of Republicans
-2 moderate Democratic lean
-1 slight Democratic lean
 0 balanced or politically neutral
+1 slight Republican lean
+2 moderate Republican lean
+3 strong support for Republican positions or strong criticism of Democrats

Distinguish an article's own narrative framing from partisan language inside attributed quotes.
Look for selection and omission of perspectives, loaded language, favorable or critical portrayal,
causal claims, headline framing, and source balance. Cite short concrete passages as evidence.
Your numeric score is authoritative: <= -1 is Left, between -1 and +1 is Center, and >= +1 is Right.
Select final_prediction and lean_strength after completing the reasoning. Use Neutral for balanced
reporting, Slight for a limited lean, Moderate for sustained one-sided framing, and Strong for
explicit advocacy or attack. Center must use Neutral.
Return only JSON matching the required schema. Put final_prediction last."""


ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "article_understanding": {"type": "string"},
        "reasoning": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "lean_strength": {
            "type": "string",
            "enum": ["Neutral", "Slight", "Moderate", "Strong"],
        },
        "probabilities": {
            "type": "object",
            "properties": {
                "Left": {"type": "number", "minimum": 0, "maximum": 1},
                "Center": {"type": "number", "minimum": 0, "maximum": 1},
                "Right": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["Left", "Center", "Right"],
            "additionalProperties": False,
        },
        "final_prediction": {"type": "string", "enum": ["Left", "Center", "Right"]},
    },
    "required": [
        "article_understanding",
        "reasoning",
        "evidence",
        "lean_strength",
        "probabilities",
        "final_prediction",
    ],
    "additionalProperties": False,
}


CHALLENGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "acknowledgement": {"type": "string"},
        "argument": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "lean_strength": {
            "type": "string",
            "enum": ["Neutral", "Slight", "Moderate", "Strong"],
        },
        "probabilities": ANALYSIS_SCHEMA["properties"]["probabilities"],
        "final_prediction": {"type": "string", "enum": ["Left", "Center", "Right"]},
    },
    "required": [
        "acknowledgement",
        "argument",
        "evidence",
        "lean_strength",
        "probabilities",
        "final_prediction",
    ],
    "additionalProperties": False,
}


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "acknowledgement": {"type": "string"},
        "counterargument": {"type": "string"},
        "reasoning": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "lean_strength": {
            "type": "string",
            "enum": ["Neutral", "Slight", "Moderate", "Strong"],
        },
        "probabilities": ANALYSIS_SCHEMA["properties"]["probabilities"],
        "final_prediction": {"type": "string", "enum": ["Left", "Center", "Right"]},
    },
    "required": [
        "acknowledgement",
        "counterargument",
        "reasoning",
        "evidence",
        "lean_strength",
        "probabilities",
        "final_prediction",
    ],
    "additionalProperties": False,
}


def analysis_prompt(article_text: str) -> str:
    return f"""Analyze the complete article below in two distinct steps.

1. Summarize what it says without judging it.
2. Assess the article's own political framing. Do not count a quoted partisan statement as the
article's bias unless the surrounding narration adopts, privileges, or fails to contextualize it.

ARTICLE START
{article_text}
ARTICLE END

Return an evidence-grounded lean_strength, a class probability distribution that sums to 1,
and final_prediction. Check that Center uses Neutral before returning JSON only."""


def _history_text(history: Sequence[Mapping[str, Any]]) -> str:
    if not history:
        return "No previous exchanges."
    return json.dumps(list(history), ensure_ascii=False, separators=(",", ":"))


def challenge_prompt(
    article_text: str,
    own: Mapping[str, Any],
    target: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> str:
    return f"""Reconsider the article in light of another analyst's disagreement.
First acknowledge any valid point. Then challenge weak or missing reasoning with concrete textual
evidence. You may keep or revise your score.

DIRECTION CHECK: Favoring Republicans or criticizing Democrats is Right. Favoring Democrats or
criticizing Republicans is Left. Neutral description without adopted partisan framing is Center.

ARTICLE START
{article_text}
ARTICLE END

YOUR CURRENT ANALYSIS
{json.dumps(own, ensure_ascii=False)}

TARGET ANALYSIS
{json.dumps(target, ensure_ascii=False)}

SHARED DEBATE HISTORY
{_history_text(history)}

Return JSON only. Put final_prediction last."""


def response_prompt(
    article_text: str,
    own: Mapping[str, Any],
    challenger: Mapping[str, Any],
    challenge: str,
    history: Sequence[Mapping[str, Any]],
) -> str:
    return f"""Respond to the latest challenge using the article text. Acknowledge valid evidence,
offer counterevidence where appropriate, and revise your score if warranted. Do not defend a prior
position merely for consistency.

DIRECTION CHECK: Favoring Republicans or criticizing Democrats is Right. Favoring Democrats or
criticizing Republicans is Left. Neutral description without adopted partisan framing is Center.

ARTICLE START
{article_text}
ARTICLE END

YOUR CURRENT ANALYSIS
{json.dumps(own, ensure_ascii=False)}

CHALLENGER CURRENT ANALYSIS
{json.dumps(challenger, ensure_ascii=False)}

LATEST CHALLENGE
{challenge}

SHARED DEBATE HISTORY
{_history_text(history)}

Return JSON only. Put final_prediction last."""

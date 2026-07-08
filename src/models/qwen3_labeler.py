"""
Qwen3 political bias labeler backed by a vLLM server.

Start the server before using this class:
    vllm serve Qwen/Qwen3-4B  --port 8002 --api-key token-abc123
    vllm serve Qwen/Qwen3-14B --port 8001 --api-key token-abc123
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from src.models.base_labeler import BaseLabeler
from src.utils.json_extractor import RobustJSONExtractor

logger = logging.getLogger(__name__)

# -- Prompts -------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert media analyst specializing in political bias detection.

Use this scoring scale for all bias analysis:
- -3: Strong support for Democrats or criticism of Republicans
- -2: Moderate Democratic support
- -1: Slight Democratic lean
-  0: Neutral or balanced
-  1: Slight Republican lean
-  2: Moderate Republican support
-  3: Strong support for Republicans or criticism of Democrats

Note: Articles that lean toward Democratic viewpoints should score at most -1.
Articles that lean toward Republican viewpoints should score at least 1.\
"""

_USER_TEMPLATE = """\
First, read the article and write a paragraph demonstrating your understanding of its content, main topic, and key points discussed.

####################
{article}
####################

Step 1: Article Understanding
Write a paragraph summarizing what this article is about, including the main topic, key events, and primary subjects discussed.

Step 2: Political Bias Analysis
Analyze its political leaning step by step under the following guidance:
- Political figures, parties, or policies mentioned and how they're portrayed
- Whether the article presents these entities favorably, neutrally, or critically
- Core Democratic or Republican values being promoted or criticized
- Whether multiple perspectives are presented or only one side
- Language tone, framing choices, and source selection
- Overall which political direction the article implicitly or explicitly supports

IMPORTANT: Write a SEPARATE and DISTINCT explanation paragraph for your bias reasoning. This paragraph must be completely different from your article understanding paragraph. The article understanding paragraph summarizes WHAT the article is about. The reasoning paragraph explains WHY you gave it a specific bias score based on political indicators.

Your output must be a JSON object with exactly these fields:

{{
  "article_understanding": "[Your paragraph showing you understood the article's content]",
  "reason": "[Political bias analysis: explain what bias indicators led to your score]",
  "lean": [Integer from -3 to 3]
}}

Return only the JSON object. Do not include any other text.\
"""

_DISCUSSION_SYSTEM = """\
You are an expert media analyst engaging in collaborative bias analysis discussion.
Always respond with valid JSON only, no additional text or commentary.

Use this scoring scale for all bias analysis:
- -3: Strong support for Democrats or criticism of Republicans
- -2: Moderate Democratic support
- -1: Slight Democratic lean
-  0: Neutral or balanced
-  1: Slight Republican lean
-  2: Moderate Republican support
-  3: Strong support for Republicans or criticism of Democrats

Note: Articles that lean toward Democratic viewpoints should score at most -1.
Articles that lean toward Republican viewpoints should score at least 1.\
"""

# -- Helpers -------------------------------------------------------------------

def _strip_thinking(text: str) -> Tuple[str, Optional[str]]:
    """Remove <think>...</think> blocks; return (clean_text, thinking_content or None)."""
    pattern = r'<think>(.*?)</think>'
    thinking = '\n'.join(re.findall(pattern, text, re.DOTALL)) or None
    clean = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return clean, thinking


def _build_challenge_user(article_content, conversation_history, own_analysis, target_analysis):
    """Build the user prompt asking this model to challenge another model's analysis."""
    return (
        f"ARTICLE CONTENT:\n{article_content}\n\n"
        f"YOUR ANALYSIS:\nScore: {own_analysis['score']} ({own_analysis['direction']})\n"
        f"Reasoning: {own_analysis['reason']}\n\n"
        f"TARGET ANALYSIS:\nScore: {target_analysis['score']} ({target_analysis['direction']})\n"
        f"Reasoning: {target_analysis['reason']}\n\n"
        f"{conversation_history}\n\n"
        "Consider the target's different interpretation. Generate a structured challenge.\n\n"
        'Output ONLY this JSON:\n'
        '{\n'
        '    "understanding": "What valid points the target analysis presents",\n'
        '    "challenge": "Your evidence-based challenge with article citations",\n'
        '    "adjusted_lean": <your bias score from -3 to 3>\n'
        '}'
    )


def _build_response_user(article_content, conversation_history, challenge, own_analysis, challenger_analysis):
    """Build the user prompt asking this model to respond to a challenge against its analysis."""
    return (
        f"ARTICLE CONTENT:\n{article_content}\n\n"
        f"YOUR INITIAL ANALYSIS:\nScore: {own_analysis['score']} ({own_analysis['direction']})\n"
        f"Reasoning: {own_analysis['reason']}\n\n"
        f"CHALLENGER'S ANALYSIS:\nScore: {challenger_analysis['score']} ({challenger_analysis['direction']})\n"
        f"Reasoning: {challenger_analysis['reason']}\n\n"
        f"{conversation_history}\n\n"
        f"LATEST CHALLENGE:\n{challenge}\n\n"
        "Respond with your assessment.\n\n"
        'Output ONLY this JSON:\n'
        '{\n'
        '    "acknowledgment": "Valid points from the challenger",\n'
        '    "counter_argument": "Your response with article evidence",\n'
        '    "final_lean": <your final bias score -3 to 3>,\n'
        '    "reason": "Brief justification"\n'
        '}'
    )


# -- Labeler -------------------------------------------------------------------

class QwenLabeler(BaseLabeler):
    """Political bias labeler using Qwen3 served by vLLM.

    The vLLM server must be running before instantiating this class.
    Example (small ensemble):
        vllm serve Qwen/Qwen3-4B --port 8002 --api-key token-abc123
    Example (regular ensemble):
        vllm serve Qwen/Qwen3-14B --port 8001 --api-key token-abc123
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str = "token-abc123",
        enable_thinking: bool = True,
        temperature: float = 0.6,
        max_tokens: int = 8192,
        top_p: float = 0.95,
        batch_size: int = 1,
    ):
        """Configure the vLLM endpoint, served model ID, and generation parameters."""
        super().__init__(model_name=model_id, batch_size=batch_size)
        self._base_url = base_url
        self._api_key = api_key
        self._enable_thinking = enable_thinking
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._client: Optional[OpenAI] = None

    # -- BaseLabeler interface -------------------------------------------------

    def load_model(self) -> None:
        """Connect to the vLLM server and verify it is reachable."""
        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        try:
            self._client.models.list()
        except Exception as exc:
            self._client = None
            raise RuntimeError(
                f"Qwen vLLM server at {self._base_url} is not reachable: {exc}"
            ) from exc
        logger.info(f"Connected to Qwen vLLM server at {self._base_url} (model: {self.model_name})")

    def predict(self, article_text: str) -> Dict[str, Any]:
        """Label a single article for political bias.

        Raises:
            RuntimeError: If the API call fails.
            ValueError: If the response cannot be parsed or 'lean' is missing.
        """
        if self._client is None:
            self.load_model()

        extra_body = {"chat_template_kwargs": {"enable_thinking": True}} if self._enable_thinking else None

        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _USER_TEMPLATE.format(article=article_text)},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
                extra_body=extra_body,
            )
        except Exception as exc:
            raise RuntimeError(f"Qwen vLLM API call failed: {exc}") from exc

        raw = resp.choices[0].message.content
        clean, thinking = _strip_thinking(raw)

        parsed = RobustJSONExtractor.extract_json(clean)
        if parsed is None:
            raise ValueError(
                f"Model output could not be parsed as JSON.\n"
                f"Raw response (first 500 chars): {raw[:500]}"
            )

        lean_raw = parsed["lean"] if parsed.get("lean") is not None else parsed.get("final_score")
        if lean_raw is None:
            raise ValueError(
                f"Model output missing required 'lean' field.\n"
                f"Parsed fields: {list(parsed.keys())}"
            )

        lean = max(-3, min(3, int(float(lean_raw))))
        result = {
            "lean": lean,
            "direction": self.score_to_direction(lean),
            "reason": parsed.get("reason", ""),
            "article_understanding": parsed.get("article_understanding", ""),
            "raw_response": raw,
        }
        if thinking:
            result["_thinking"] = thinking
        return result

    def unload_model(self) -> None:
        """Drop the client connection (vLLM server keeps running)."""
        self._client = None

    # -- Discussion support ----------------------------------------------------

    def generate_discussion_challenge(
        self,
        article_content: str,
        conversation_history: str,
        own_analysis: Dict,
        target_analysis: Dict,
    ) -> Tuple[str, str]:
        """Generate a structured challenge against target_analysis.

        Returns:
            Tuple of (user prompt sent, raw model response).
        """
        if self._client is None:
            self.load_model()

        user_msg = _build_challenge_user(article_content, conversation_history, own_analysis, target_analysis)
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _DISCUSSION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
            )
        except Exception as exc:
            raise RuntimeError(f"Qwen challenge API call failed: {exc}") from exc

        raw = resp.choices[0].message.content
        clean, _ = _strip_thinking(raw)
        return user_msg, clean

    def generate_discussion_response(
        self,
        article_content: str,
        conversation_history: str,
        challenge: str,
        own_analysis: Dict,
        challenger_analysis: Dict,
    ) -> Tuple[str, Dict]:
        """Respond to a challenge, possibly revising the bias score.

        Returns:
            Tuple of (user prompt sent, dict with lean/reason/raw_response).
        """
        if self._client is None:
            self.load_model()

        user_msg = _build_response_user(article_content, conversation_history, challenge, own_analysis, challenger_analysis)
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _DISCUSSION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
            )
        except Exception as exc:
            raise RuntimeError(f"Qwen response API call failed: {exc}") from exc

        raw = resp.choices[0].message.content
        clean, _ = _strip_thinking(raw)
        parsed = RobustJSONExtractor.extract_challenge_fields(clean)
        lean_raw = next(
            (parsed[k] for k in ("final_lean", "adjusted_lean", "lean") if parsed.get(k) is not None),
            None,
        )
        if lean_raw is None:
            logger.warning(
                "%s discussion response contained no final_lean/adjusted_lean/lean field; "
                "keeping previous score %s. Parsed fields: %s",
                self.__class__.__name__, own_analysis["score"], list(parsed.keys()),
            )
        return user_msg, {
            "lean": max(-3, min(3, int(float(lean_raw)))) if lean_raw is not None else own_analysis["score"],
            "reason": parsed.get("reason", parsed.get("acknowledgment", "")),
            "acknowledgment": parsed.get("acknowledgment", ""),
            "counter_argument": parsed.get("counter_argument", ""),
            "raw_response": raw,
        }

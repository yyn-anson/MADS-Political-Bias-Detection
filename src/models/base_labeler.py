"""
Base class for all political bias labelers.

All model wrappers must extend BaseLabeler and implement load_model() and predict().
The ensemble system accepts any list of BaseLabeler subclasses, so you can swap in
any model without touching the ensemble logic.

Output contract for predict():
    {
        "lean":      float,  # bias score in [-3, 3]
        "direction": str,    # "Left" | "Center" | "Right"
        "reason":    str,    # brief explanation
        "error":     bool    # True only if inference failed
    }

Optional fields (pass-through, not required by the ensemble):
    "article_understanding": str   # article summary (model-specific)
    "raw_response": str            # raw LLM output for debugging
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseLabeler(ABC):
    """
    Abstract base class for a political bias labeler.

    Subclass this and implement load_model(), predict(), and optionally unload_model()
    and generate_discussion_challenge() / generate_discussion_response() if you want
    the labeler to participate in collaborative discussions.
    """

    def __init__(self, model_name: str, cache_dir: str = "models/", batch_size: int = 1):
        """Store the model identity and batching configuration common to all labelers."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """Download / load the model and tokenizer into memory."""

    @abstractmethod
    def predict(self, article_text: str) -> Dict[str, Any]:
        """
        Analyze a single article and return a bias prediction.

        Args:
            article_text: Full text of the article to analyze.

        Returns:
            dict with keys: lean (float), direction (str), reason (str), error (bool).
        """

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def label_articles_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of already-formatted prompt strings.

        The default implementation calls predict() sequentially. Override for
        true batched inference (e.g. with a transformers pipeline).

        Args:
            prompts: List of formatted prompt strings (from ArticleDataset).

        Returns:
            List of prediction dicts, one per prompt.

        Raises:
            Any exception raised by predict() - callers must handle errors explicitly.
        """
        return [self.predict(prompt) for prompt in prompts]

    def unload_model(self) -> None:
        """
        Free GPU/CPU memory by deleting model weights.

        The ensemble calls this after each model's turn to keep peak VRAM low.
        Override to add model-specific cleanup.
        """
        import gc
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate_discussion_challenge(
        self,
        article_content: str,
        conversation_history: str,
        own_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any],
    ):
        """
        Generate a structured challenge during collaborative discussion.

        Only needed if this labeler participates in the two-stage discussion.
        The default raises NotImplementedError; override if you want discussion support.

        Returns:
            Tuple of (prompt_str, response_str)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support collaborative discussion. "
            "Override generate_discussion_challenge() to enable it."
        )

    def generate_discussion_response(
        self,
        article_content: str,
        conversation_history: str,
        challenge: str,
        own_analysis: Dict[str, Any],
        challenger_analysis: Dict[str, Any],
    ):
        """
        Generate a structured response during collaborative discussion.

        Returns:
            Tuple of (prompt_str, result_dict)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support collaborative discussion. "
            "Override generate_discussion_response() to enable it."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def score_to_direction(score: float) -> str:
        """Convert a numeric bias score [-3, 3] to a direction label."""
        if score <= -1:
            return "Left"
        elif score >= 1:
            return "Right"
        return "Center"

    def __repr__(self) -> str:
        """Return a short description with class name, model, and batch size."""
        return f"{self.__class__.__name__}(model={self.model_name}, batch_size={self.batch_size})"

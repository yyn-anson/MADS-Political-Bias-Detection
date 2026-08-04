"""Reasoning embeddings for divergent-pair selection and stagnation checks."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


class HashingEmbedder:
    """Dependency-free semantic-lite fallback; deterministic and offline."""

    name = "hashing-384-fallback"

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        features = tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
        counts = Counter(features)
        vector = [0.0] * self.dimensions
        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class MiniLMEmbedder:
    """Exact embedding model named in the paper; loaded only when requested/available."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]


def create_embedder(backend: str, model_name: str) -> tuple[Embedder, str | None]:
    """Create the selected embedder and return an optional fallback warning."""
    if backend == "hashing":
        return HashingEmbedder(), None
    try:
        return MiniLMEmbedder(model_name), None
    except (ImportError, OSError, RuntimeError) as exc:
        if backend == "minilm":
            raise RuntimeError(
                "MiniLM was requested but could not be loaded. Install with "
                "`pip install -e '.[paper]'` and ensure the model is available."
            ) from exc
        warning = (
            "MiniLM is unavailable; using the deterministic hashing fallback. "
            "Install the `paper` extra for exact paper embeddings."
        )
        return HashingEmbedder(), warning


def most_divergent_pair(
    names: Sequence[str], reasonings: Sequence[str], embedder: Embedder
) -> tuple[str, str, float]:
    if len(names) != len(reasonings) or len(names) < 2:
        raise ValueError("names and reasonings must have equal length >= 2")
    vectors = embedder.encode(reasonings)
    candidates: list[tuple[float, str, str]] = []
    for i, first in enumerate(names):
        for j in range(i + 1, len(names)):
            candidates.append((cosine_similarity(vectors[i], vectors[j]), first, names[j]))
    similarity, first, second = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return first, second, similarity

"""
Shared utility for extracting ground-truth labels from dataset article dicts.

Supported dataset types: 'baly', 'budak', 'ad_fontes'.
Returns a 3-class label: 0 = Left, 1 = Center, 2 = Right.
"""

from typing import Tuple, Optional


def get_ground_truth_labels(
    article_data: dict, dataset_type: str
) -> Tuple[Optional[int], bool]:
    """Return (label, is_valid) for an article dict.

    label is 0 (Left), 1 (Center), or 2 (Right).
    is_valid is False when the field is missing or the dataset has no ground truth.
    """
    if dataset_type == "baly":
        raw = article_data.get("bias")
        if raw is None:
            return None, False
        return int(raw), True

    if dataset_type == "budak":
        text = str(article_data.get("bias_text", "")).strip().lower()
        mapping = {"left": 0, "center": 1, "right": 2}
        label = mapping.get(text)
        if label is None:
            return None, False
        return label, True

    if dataset_type == "ad_fontes":
        raw = article_data.get("Bias")
        if raw is None:
            return None, False
        score = float(raw)
        if score < -6:
            return 0, True   # Left
        if score > 6:
            return 2, True   # Right
        return 1, True       # Center

    return None, False

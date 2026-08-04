"""Small dependency-free evaluation metrics for optional user labels."""

from __future__ import annotations

from collections.abc import Iterable

from .types import ArticleResult, BiasLabel


def classification_metrics(results: Iterable[ArticleResult]) -> dict:
    pairs = [
        (result.article.label, result.final_label)
        for result in results
        if result.article.label is not None and result.error is None
    ]
    labels = list(BiasLabel)
    if not pairs:
        return {"labeled_articles": 0}

    confusion = {actual.value: {predicted.value: 0 for predicted in labels} for actual in labels}
    for actual, predicted in pairs:
        confusion[actual.value][predicted.value] += 1

    per_class = {}
    f1_values = []
    for label in labels:
        true_positive = confusion[label.value][label.value]
        false_positive = sum(
            confusion[actual.value][label.value] for actual in labels if actual is not label
        )
        false_negative = sum(
            confusion[label.value][predicted.value]
            for predicted in labels
            if predicted is not label
        )
        support = sum(confusion[label.value].values())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label.value] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    correct = sum(1 for actual, predicted in pairs if actual is predicted)
    return {
        "labeled_articles": len(pairs),
        "accuracy": round(correct / len(pairs), 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }

"""Outlet-level aggregation and dependency-free paper-style SVG figures."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from html import escape
from typing import Any

from .metrics import classification_metrics_from_pairs
from .types import ArticleResult, BiasLabel, score_to_label

COLORS = {
    BiasLabel.LEFT: "#3267c8",
    BiasLabel.CENTER: "#667085",
    BiasLabel.RIGHT: "#c74440",
}


@dataclass(frozen=True)
class OutletSummary:
    source: str
    article_count: int
    expected_label: BiasLabel | None
    predicted_label: BiasLabel
    mean_score: float
    median_score: float
    minimum_score: float
    maximum_score: float
    scores: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_label"] = self.expected_label.value if self.expected_label else None
        payload["predicted_label"] = self.predicted_label.value
        payload["scores"] = list(self.scores)
        return payload


def summarize_outlets(results: list[ArticleResult]) -> list[OutletSummary]:
    """Aggregate article scores by source and order outlets by mean score."""
    grouped: dict[str, list[ArticleResult]] = defaultdict(list)
    for result in results:
        if result.error is None and result.article.source:
            grouped[result.article.source].append(result)

    summaries = []
    for source, source_results in grouped.items():
        scores = tuple(sorted(result.final_score for result in source_results))
        expected = {result.article.label for result in source_results if result.article.label}
        expected_label = next(iter(expected)) if len(expected) == 1 else None
        mean_score = statistics.fmean(scores)
        summaries.append(
            OutletSummary(
                source=source,
                article_count=len(scores),
                expected_label=expected_label,
                predicted_label=score_to_label(mean_score),
                mean_score=round(mean_score, 4),
                median_score=round(statistics.median(scores), 4),
                minimum_score=round(min(scores), 4),
                maximum_score=round(max(scores), 4),
                scores=scores,
            )
        )
    return sorted(summaries, key=lambda item: (item.mean_score, item.source.lower()))


def outlet_metrics(summaries: list[OutletSummary]) -> dict[str, Any]:
    pairs = [
        (summary.expected_label, summary.predicted_label)
        for summary in summaries
        if summary.expected_label is not None
    ]
    return classification_metrics_from_pairs(pairs)


def _x(score: float, left: float, width: float) -> float:
    return left + (max(-3.0, min(3.0, score)) + 3.0) / 6.0 * width


def _svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        (
            "<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
            "fill:#14213d}.title{font-size:22px;font-weight:700}"
            ".subtitle{font-size:12px;fill:#667085}.label{font-size:12px;font-weight:600}"
            ".small{font-size:10px;fill:#667085}.tick{font-size:10px;fill:#667085}"
            ".grid{stroke:#98a2b3;stroke-opacity:.28;stroke-width:1}</style>"
        ),
        '<rect width="100%" height="100%" rx="12" fill="#ffffff"/>',
        f'<text class="title" x="24" y="34">{escape(title)}</text>',
        f'<text class="subtitle" x="24" y="55">{escape(description)}</text>',
    ]


def _background(parts: list[str], left: float, top: float, width: float, height: float) -> None:
    thirds = width / 3
    parts.extend(
        [
            f'<rect x="{left:.1f}" y="{top:.1f}" width="{thirds:.1f}" height="{height:.1f}" fill="#edf3fc"/>',
            f'<rect x="{left + thirds:.1f}" y="{top:.1f}" width="{thirds:.1f}" height="{height:.1f}" fill="#f2f4f7"/>',
            f'<rect x="{left + 2 * thirds:.1f}" y="{top:.1f}" width="{thirds:.1f}" height="{height:.1f}" fill="#fbeeed"/>',
        ]
    )
    for tick in range(-3, 4):
        position = _x(float(tick), left, width)
        parts.append(
            f'<line class="grid" x1="{position:.1f}" y1="{top:.1f}" '
            f'x2="{position:.1f}" y2="{top + height:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" text-anchor="middle" x="{position:.1f}" '
            f'y="{top + height + 19:.1f}">{tick:+d}</text>'
        )
    parts.extend(
        [
            f'<text class="small" text-anchor="middle" x="{left + thirds / 2:.1f}" y="{top - 8:.1f}">Left</text>',
            f'<text class="small" text-anchor="middle" x="{left + 1.5 * thirds:.1f}" y="{top - 8:.1f}">Center</text>',
            f'<text class="small" text-anchor="middle" x="{left + 2.5 * thirds:.1f}" y="{top - 8:.1f}">Right</text>',
        ]
    )


def _density_path(scores: tuple[float, ...], left: float, width: float, center_y: float) -> str:
    sample_points = [-3.0 + index * 0.075 for index in range(81)]
    deviation = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    bandwidth = max(0.18, min(0.65, 1.06 * max(deviation, 0.15) * len(scores) ** -0.2))
    densities = [
        sum(math.exp(-0.5 * ((point - score) / bandwidth) ** 2) for score in scores)
        for point in sample_points
    ]
    maximum = max(densities) or 1.0
    half_heights = [13.5 * density / maximum for density in densities]
    upper = [
        f"{_x(point, left, width):.1f},{center_y - height:.1f}"
        for point, height in zip(sample_points, half_heights, strict=True)
    ]
    lower = [
        f"{_x(point, left, width):.1f},{center_y + height:.1f}"
        for point, height in reversed(list(zip(sample_points, half_heights, strict=True)))
    ]
    return "M " + " L ".join(upper + lower) + " Z"


def score_distribution_svg(summaries: list[OutletSummary]) -> str:
    width, row_height = 1200, 48
    left, right, top, bottom = 320.0, 30.0, 90.0, 48.0
    plot_width = width - left - right
    plot_height = max(row_height, len(summaries) * row_height)
    height = int(top + plot_height + bottom)
    parts = _svg_start(
        width,
        height,
        "Article bias-score distributions by outlet",
        "Local MADS scores; blue = Left, gray = Center, red = Right supplied labels.",
    )
    _background(parts, left, top, plot_width, plot_height)
    for index, summary in enumerate(summaries):
        center_y = top + (index + 0.5) * row_height
        color = COLORS.get(summary.expected_label, "#667085")
        parts.append(
            f'<text class="label" text-anchor="end" x="{left - 12:.1f}" y="{center_y + 4:.1f}">'
            f"{escape(summary.source)}</text>"
        )
        parts.append(
            f'<path d="{_density_path(summary.scores, left, plot_width, center_y)}" '
            f'fill="{color}" fill-opacity=".42" stroke="{color}" stroke-width="1.3"/>'
        )
        mean_x = _x(summary.mean_score, left, plot_width)
        parts.append(
            f'<line x1="{mean_x:.1f}" y1="{center_y - 15:.1f}" x2="{mean_x:.1f}" '
            f'y2="{center_y + 15:.1f}" stroke="#14213d" stroke-width="2"/>'
        )
        parts.append(
            f'<text class="small" x="{left + plot_width - 3:.1f}" y="{center_y - 14:.1f}" '
            f'text-anchor="end">n={summary.article_count}, mean={summary.mean_score:+.2f}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def outlet_comparison_svg(summaries: list[OutletSummary]) -> str:
    width, row_height = 1200, 44
    left, right, top, bottom = 320.0, 190.0, 90.0, 48.0
    plot_width = width - left - right
    plot_height = max(row_height, len(summaries) * row_height)
    height = int(top + plot_height + bottom)
    parts = _svg_start(
        width,
        height,
        "Outlet-level mean score and classification",
        "Outlets are ordered by mean score; prediction uses the paper's -1 and +1 thresholds.",
    )
    _background(parts, left, top, plot_width, plot_height)
    for index, summary in enumerate(summaries):
        center_y = top + (index + 0.5) * row_height
        expected_text = summary.expected_label.value if summary.expected_label else "unlabeled"
        if summary.expected_label is None:
            status = "no expected label"
        elif summary.expected_label is summary.predicted_label:
            status = "correct"
        else:
            status = "differs"
        color = COLORS[summary.predicted_label]
        parts.extend(
            [
                (
                    f'<text class="label" text-anchor="end" x="{left - 12:.1f}" '
                    f'y="{center_y + 2:.1f}">{escape(summary.source)}</text>'
                ),
                (
                    f'<text class="small" text-anchor="end" x="{left - 12:.1f}" '
                    f'y="{center_y + 14:.1f}">Expected: {expected_text}</text>'
                ),
                (
                    f'<line x1="{_x(summary.minimum_score, left, plot_width):.1f}" '
                    f'y1="{center_y:.1f}" '
                    f'x2="{_x(summary.maximum_score, left, plot_width):.1f}" '
                    f'y2="{center_y:.1f}" stroke="#475467" stroke-width="2" '
                    'stroke-linecap="round"/>'
                ),
                (
                    f'<circle cx="{_x(summary.mean_score, left, plot_width):.1f}" '
                    f'cy="{center_y:.1f}" r="7" fill="{color}" stroke="#fff" '
                    'stroke-width="2"/>'
                ),
                (
                    f'<text class="label" x="{left + plot_width + 13:.1f}" '
                    f'y="{center_y + 1:.1f}">{summary.predicted_label.value} '
                    f"({summary.mean_score:+.2f})</text>"
                ),
                (
                    f'<text class="small" x="{left + plot_width + 13:.1f}" '
                    f'y="{center_y + 14:.1f}">{status}</text>'
                ),
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)

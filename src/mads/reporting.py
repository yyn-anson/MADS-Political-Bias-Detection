"""Durable JSON/JSONL/CSV outputs and a self-contained human-readable HTML report."""

from __future__ import annotations

import csv
import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from .config import AppConfig
from .metrics import classification_metrics
from .types import ArticleResult, BiasLabel


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _score_position(score: float) -> float:
    return max(0.0, min(100.0, (score + 3.0) / 6.0 * 100.0))


def _result_card(result: ArticleResult) -> str:
    label_class = result.final_label.value.lower()
    expected = result.article.label.value if result.article.label else "Not provided"
    correctness = ""
    if result.article.label:
        correctness = (
            '<span class="correct">matches label</span>'
            if result.article.label is result.final_label
            else '<span class="incorrect">differs from label</span>'
        )
    agents = "".join(
        f"<tr><td>{escape(name)}</td><td>{analysis.score:+.2f}</td>"
        f"<td>{analysis.label.value}</td><td>{analysis.entropy:.3f}</td>"
        f"<td>{escape(analysis.confidence_source)}</td></tr>"
        for name, analysis in result.initial_analyses.items()
    )
    debate_summary = (
        f"{len(result.debate)} exchange(s)" if result.debate else "No debate - unanimous early exit"
    )
    source_line = ""
    if result.article.url:
        source_line = (
            f'<a href="{escape(result.article.url, quote=True)}" rel="noreferrer">'
            f"{escape(result.article.source or 'Source article')}</a>"
        )
    return f"""
    <article class="card">
      <div class="card-head">
        <div><p class="eyebrow">{escape(result.article.id)} · {escape(result.route)}</p>
        <h2>{escape(result.article.title)}</h2><p>{source_line}</p></div>
        <div class="verdict {label_class}">{result.final_label.value}<small>{result.final_score:+.2f}</small></div>
      </div>
      <div class="scale"><span style="left:{_score_position(result.final_score):.2f}%"></span></div>
      <div class="facts"><span>Expected: {expected} {correctness}</span>
      <span>{debate_summary}</span><span>{result.elapsed_seconds:.1f}s</span></div>
      <details><summary>Agent evidence and decision trace</summary>
        <table><thead><tr><th>Agent</th><th>Score</th><th>Label</th><th>Entropy</th><th>Confidence</th></tr></thead>
        <tbody>{agents}</tbody></table>
        <pre>{escape(json.dumps(result.decision_trace, indent=2, ensure_ascii=False))}</pre>
      </details>
    </article>"""


def _html_report(
    results: Sequence[ArticleResult],
    errors: Sequence[dict[str, str]],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    counts = {label.value: sum(r.final_label is label for r in results) for label in BiasLabel}
    debated = sum(bool(result.debate) for result in results)
    accuracy = metrics.get("accuracy")
    metric_card = (
        f'<div class="stat"><b>{accuracy * 100:.1f}%</b><span>sample accuracy</span></div>'
        if accuracy is not None
        else '<div class="stat"><b>n/a</b><span>no labels supplied</span></div>'
    )
    error_html = ""
    if errors:
        items = "".join(
            f"<li><b>{escape(item['id'])}</b>: {escape(item['error'])}</li>" for item in errors
        )
        error_html = f"<section><h2>Errors</h2><ul>{items}</ul></section>"
    cards = "".join(_result_card(result) for result in results)
    generated = escape(str(manifest["generated_at"]))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MADS political bias report</title>
<style>
:root{{--ink:#14213d;--muted:#667085;--paper:#f6f3ed;--card:#fff;--left:#3267c8;--center:#667085;--right:#c74440;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
main{{max-width:1040px;margin:auto;padding:54px 24px 80px}} h1{{font:700 42px/1.05 ui-serif,Georgia,serif;margin:.15em 0}} h2{{font:700 23px/1.2 ui-serif,Georgia,serif;margin:.15em 0}}
.kicker,.eyebrow{{text-transform:uppercase;letter-spacing:.11em;font-size:11px;font-weight:700;color:var(--muted)}} .intro{{max-width:720px;color:var(--muted)}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:30px 0}} .stat{{background:var(--card);padding:18px;border:1px solid #e4dfd6;border-radius:12px}} .stat b{{display:block;font-size:27px}} .stat span{{color:var(--muted);font-size:12px}}
.card{{background:var(--card);border:1px solid #e4dfd6;border-radius:16px;padding:23px;margin:16px 0;box-shadow:0 7px 25px #14213d0b}} .card-head{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}}
.verdict{{min-width:106px;text-align:center;padding:12px;border-radius:11px;color:white;font-weight:800}} .verdict small{{display:block;font-weight:500}} .left{{background:var(--left)}} .center{{background:var(--center)}} .right{{background:var(--right)}}
.scale{{height:8px;border-radius:8px;margin:22px 0 10px;background:linear-gradient(90deg,var(--left),#dce4ef 40%,#ddd 50%,#ecd9d7 60%,var(--right));position:relative}} .scale span{{position:absolute;top:-5px;width:3px;height:18px;background:#111;border-radius:3px}}
.facts{{display:flex;gap:22px;flex-wrap:wrap;color:var(--muted);font-size:13px}} .correct{{color:#18794e}} .incorrect{{color:#b42318}} details{{margin-top:18px}} summary{{cursor:pointer;font-weight:700}} table{{width:100%;border-collapse:collapse;margin:13px 0}} th,td{{text-align:left;border-bottom:1px solid #eee;padding:8px}} pre{{white-space:pre-wrap;background:#f7f7f8;padding:14px;border-radius:9px;font-size:12px}} a{{color:#315da8}} footer{{margin-top:35px;color:var(--muted);font-size:12px}}
@media(max-width:720px){{.stats{{grid-template-columns:1fr 1fr}}.card-head{{display:block}}.verdict{{margin-top:12px;width:110px}}}}
</style></head><body><main>
<p class="kicker">Conditional multi-agent debate</p><h1>MADS analysis report</h1>
<p class="intro">Content-only Left / Center / Right analysis. Scores range from -3 to +3. Labels are model judgments, not verified facts; inspect the cited evidence before relying on them.</p>
<div class="stats"><div class="stat"><b>{len(results)}</b><span>articles completed</span></div><div class="stat"><b>{debated}</b><span>debates triggered</span></div><div class="stat"><b>{counts["Left"]} / {counts["Center"]} / {counts["Right"]}</b><span>Left / Center / Right</span></div>{metric_card}</div>
{cards}{error_html}<footer>Generated {generated}. Agent states, token-probability confidence, decision traces, and debate transcripts are preserved in results.json.</footer>
</main></body></html>"""


def write_report(
    output_root: str | Path,
    results: Sequence[ArticleResult],
    errors: Sequence[dict[str, str]],
    config: AppConfig,
    *,
    input_path: str,
    embedding_warning: str | None = None,
) -> Path:
    generated = datetime.now(UTC)
    run_dir = Path(output_root) / generated.strftime("run_%Y%m%d_%H%M%S")
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{run_dir}_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True)

    metrics = classification_metrics(results)
    manifest = {
        "schema_version": "2.0",
        "generated_at": generated.isoformat(),
        "input_path": input_path,
        "github": "https://github.com/yyn-anson/MADS-Political-Bias-Detection",
        "models": [agent.model for agent in config.agents],
        "agents": [asdict(agent) for agent in config.agents],
        "method": asdict(config.method),
        "runtime": asdict(config.runtime),
        "embedding_warning": embedding_warning,
        "completed": len(results),
        "failed": len(errors),
        "metrics": metrics,
    }
    payload = {
        "manifest": manifest,
        "results": [result.to_dict() for result in results],
        "errors": list(errors),
    }
    _atomic_text(run_dir / "results.json", json.dumps(payload, indent=2, ensure_ascii=False))
    _atomic_text(
        run_dir / "results.jsonl",
        "\n".join(json.dumps(result.to_dict(), ensure_ascii=False) for result in results) + "\n",
    )
    _atomic_text(run_dir / "report.html", _html_report(results, errors, metrics, manifest))

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=run_dir, delete=False
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "title",
                "source",
                "url",
                "expected_label",
                "predicted_label",
                "score",
                "route",
                "debate_exchanges",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "id": result.article.id,
                    "title": result.article.title,
                    "source": result.article.source,
                    "url": result.article.url,
                    "expected_label": result.article.label.value if result.article.label else "",
                    "predicted_label": result.final_label.value,
                    "score": result.final_score,
                    "route": result.route,
                    "debate_exchanges": len(result.debate),
                    "elapsed_seconds": result.elapsed_seconds,
                }
            )
        temporary_csv = Path(handle.name)
    temporary_csv.replace(run_dir / "summary.csv")
    return run_dir

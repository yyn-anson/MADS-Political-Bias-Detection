"""Friendly command-line entrypoint for local article analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .data import load_articles, validate_article_lengths
from .embeddings import create_embedder
from .engine import MADSAnalyzer
from .llm import OllamaOpenAIClient
from .reporting import write_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze news articles with the MADS three-agent debate method."
    )
    parser.add_argument("--input", default="data/articles", help="Article file or directory")
    parser.add_argument("--demo", action="store_true", help="Analyze the five bundled samples")
    parser.add_argument("--output", default="reports", help="Directory for timestamped reports")
    parser.add_argument("--config", default="mads.toml", help="TOML configuration file")
    parser.add_argument("--model", help="Use one installed Ollama model for all three agents")
    parser.add_argument("--endpoint", help="Override the OpenAI-compatible local endpoint")
    parser.add_argument(
        "--embedding", choices=["auto", "minilm", "hashing"], help="Reasoning embedder"
    )
    parser.add_argument("--max-rounds", type=int, help="Maximum rounds per pairwise debate")
    parser.add_argument("--serial", action="store_true", help="Run initial agents sequentially")
    parser.add_argument("--limit", type=int, help="Analyze only the first N articles")
    parser.add_argument("--check", action="store_true", help="Check the local model and exit")
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate data without calling a model"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config).with_overrides(
            endpoint=args.endpoint,
            model=args.model,
            embedding_backend=args.embedding,
            max_debate_rounds=args.max_rounds,
            parallel_initial_analysis=False if args.serial else None,
        )
        client = OllamaOpenAIClient(
            config.runtime.endpoint,
            timeout_seconds=config.runtime.request_timeout_seconds,
            retries=config.runtime.retries,
            max_output_tokens=config.runtime.max_output_tokens,
        )
        if args.check:
            models = client.available_models()
            required = sorted({agent.model for agent in config.agents})
            missing = [model for model in required if model not in models]
            print(f"Ollama endpoint: {config.runtime.endpoint}")
            print(f"Installed models: {', '.join(models) or '(none)'}")
            if missing:
                print(f"Missing required models: {', '.join(missing)}", file=sys.stderr)
                return 1
            print("Ready: all configured models are installed.")
            return 0

        input_path = "data/sample_articles" if args.demo else args.input
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be at least 1")
        articles = load_articles(input_path, limit=args.limit)
        validate_article_lengths(articles, config.method.max_article_characters)
        print(f"Validated {len(articles)} article(s) from {input_path}.")
        if args.validate_only:
            return 0

        models = client.available_models()
        missing = sorted({agent.model for agent in config.agents if agent.model not in models})
        if missing:
            raise RuntimeError(
                f"configured model(s) not installed in Ollama: {missing}. "
                "Use --model with a name shown by `ollama list`."
            )
        embedder, embedding_warning = create_embedder(
            config.method.embedding_backend, config.method.embedding_model
        )
        if embedding_warning:
            print(f"Note: {embedding_warning}")
        analyzer = MADSAnalyzer(config, client, embedder, progress=lambda message: print(message))
        results = []
        errors = []
        for index, article in enumerate(articles, 1):
            print(f"\n[{index}/{len(articles)}] {article.title}")
            try:
                result = analyzer.analyze_article(article)
                results.append(result)
                print(
                    f"  -> {result.final_label.value} ({result.final_score:+.2f}), "
                    f"route={result.route}, exchanges={len(result.debate)}"
                )
            # A malformed model response must not discard results from other articles.
            except Exception as exc:  # noqa: BLE001 - deliberate per-article boundary
                errors.append({"id": article.id, "error": str(exc)})
                print(f"  -> ERROR: {exc}", file=sys.stderr)

        run_dir = write_report(
            args.output,
            results,
            errors,
            config,
            input_path=str(Path(input_path)),
            embedding_warning=embedding_warning,
        )
        print(f"\nReport: {run_dir / 'report.html'}")
        print(f"Machine-readable results: {run_dir / 'results.json'}")
        return 0 if results and not errors else 1
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

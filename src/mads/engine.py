"""Faithful, testable implementation of the MADS conditional-debate algorithm."""

from __future__ import annotations

import copy
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import AgentConfig, AppConfig
from .embeddings import Embedder, cosine_similarity, most_divergent_pair
from .llm import Completion, LLMBackend
from .prompts import (
    ANALYSIS_SCHEMA,
    CHALLENGE_SCHEMA,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    analysis_prompt,
    challenge_prompt,
    response_prompt,
)
from .types import Analysis, Article, ArticleResult, DebateExchange, score_to_label

ProgressCallback = Callable[[str], None]


def _snapshot(analysis: Analysis) -> dict[str, Any]:
    return {
        "score": analysis.score,
        "label": analysis.label.value,
        "understanding": analysis.understanding,
        "reasoning": analysis.reasoning,
        "evidence": analysis.evidence,
        "probabilities": analysis.probabilities,
        "entropy": analysis.entropy,
        "confidence_source": analysis.confidence_source,
    }


def _history_for_prompt(exchanges: Sequence[DebateExchange]) -> list[dict[str, Any]]:
    return [
        {
            "stage": item.stage,
            "round": item.round,
            "challenger": item.challenger,
            "target": item.target,
            "challenge": item.challenge,
            "response": item.response,
            "challenger_after": {
                "score": item.challenger_after["score"],
                "label": item.challenger_after["label"],
            },
            "target_after": {
                "score": item.target_after["score"],
                "label": item.target_after["label"],
            },
        }
        for item in exchanges
    ]


class MADSAnalyzer:
    """Analyze articles with three independent agents and conditional debate."""

    def __init__(
        self,
        config: AppConfig,
        backend: LLMBackend,
        embedder: Embedder,
        *,
        progress: ProgressCallback | None = None,
    ):
        self.config = config.validate()
        self.backend = backend
        self.embedder = embedder
        self.progress = progress or (lambda _message: None)

    @property
    def threshold(self) -> float:
        return self.config.method.direction_threshold

    def _complete(
        self,
        agent: AgentConfig,
        user_prompt: str,
        schema: Mapping[str, Any],
        *,
        previous: Analysis | None = None,
        seed_offset: int = 0,
    ) -> Analysis:
        completion: Completion = self.backend.complete(
            model=agent.model,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=schema,
            temperature=agent.temperature,
            seed=agent.seed + seed_offset,
        )
        analysis = Analysis.from_payload(
            agent.name,
            completion.payload,
            threshold=self.threshold,
            probabilities=completion.class_probabilities,
            confidence_source=completion.confidence_source,
            latency_seconds=completion.latency_seconds,
            usage=completion.usage,
        )
        if previous and not analysis.understanding:
            analysis.understanding = previous.understanding
        return analysis

    def _initial(self, article: Article, agent: AgentConfig) -> Analysis:
        self.progress(f"{article.id}: {agent.name} initial analysis")
        return self._complete(agent, analysis_prompt(article.text), ANALYSIS_SCHEMA)

    def _run_initial_analyses(self, article: Article) -> dict[str, Analysis]:
        agents = self.config.agents
        if self.config.runtime.parallel_initial_analysis:
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="mads-agent") as pool:
                futures = {
                    pool.submit(self._initial, article, agent): agent.name for agent in agents
                }
                return {futures[future]: future.result() for future in futures}
        return {agent.name: self._initial(article, agent) for agent in agents}

    def _agent_config(self, name: str) -> AgentConfig:
        return next(agent for agent in self.config.agents if agent.name == name)

    @staticmethod
    def _panel_unanimous(states: Mapping[str, Analysis]) -> bool:
        return len({analysis.label for analysis in states.values()}) == 1

    @staticmethod
    def _more_confident(names: Sequence[str], states: Mapping[str, Analysis]) -> str:
        return min(names, key=lambda name: (states[name].entropy, name))

    def _debate_pair(
        self,
        article: Article,
        states: dict[str, Analysis],
        first: str,
        second: str,
        stage: str,
        exchanges: list[DebateExchange],
        trace: list[dict[str, Any]],
    ) -> str:
        previous_argument_vector: list[float] | None = None
        termination = "round_limit"
        for round_number in range(1, self.config.method.max_debate_rounds + 1):
            challenger, target = (first, second) if round_number % 2 else (second, first)
            challenger_before = _snapshot(states[challenger])
            target_before = _snapshot(states[target])
            history = _history_for_prompt(exchanges)

            self.progress(
                f"{article.id}: {stage} round {round_number}, {challenger} challenges {target}"
            )
            challenger_config = self._agent_config(challenger)
            challenge_completion = self.backend.complete(
                model=challenger_config.model,
                system=SYSTEM_PROMPT,
                user=challenge_prompt(article.text, challenger_before, target_before, history),
                schema=CHALLENGE_SCHEMA,
                temperature=challenger_config.temperature,
                seed=challenger_config.seed + 100 + round_number,
            )
            states[challenger] = Analysis.from_payload(
                challenger,
                challenge_completion.payload,
                threshold=self.threshold,
                probabilities=challenge_completion.class_probabilities,
                confidence_source=challenge_completion.confidence_source,
                latency_seconds=challenge_completion.latency_seconds,
                usage=challenge_completion.usage,
            )
            states[challenger].understanding = states[
                challenger
            ].understanding or challenger_before.get("understanding", "")
            challenge_text = str(
                challenge_completion.payload.get("argument")
                or challenge_completion.payload.get("reasoning")
                or ""
            )

            target_config = self._agent_config(target)
            response_completion = self.backend.complete(
                model=target_config.model,
                system=SYSTEM_PROMPT,
                user=response_prompt(
                    article.text,
                    target_before,
                    _snapshot(states[challenger]),
                    challenge_text,
                    history,
                ),
                schema=RESPONSE_SCHEMA,
                temperature=target_config.temperature,
                seed=target_config.seed + 200 + round_number,
            )
            states[target] = Analysis.from_payload(
                target,
                response_completion.payload,
                threshold=self.threshold,
                probabilities=response_completion.class_probabilities,
                confidence_source=response_completion.confidence_source,
                latency_seconds=response_completion.latency_seconds,
                usage=response_completion.usage,
            )
            states[target].understanding = states[target].understanding or str(
                target_before.get("understanding", "")
            )
            response_text = str(
                response_completion.payload.get("counterargument")
                or response_completion.payload.get("reasoning")
                or ""
            )

            combined_argument = f"{challenge_text}\n{response_text}"
            current_vector = self.embedder.encode([combined_argument])[0]
            similarity = (
                cosine_similarity(previous_argument_vector, current_vector)
                if previous_argument_vector is not None
                else None
            )
            previous_argument_vector = current_vector

            if states[challenger].label is states[target].label:
                termination = "pair_consensus"
            elif similarity is not None and similarity > self.config.method.stagnation_similarity:
                termination = "stagnation"
            elif round_number == self.config.method.max_debate_rounds:
                termination = "round_limit"
            else:
                termination = "continue"

            exchange = DebateExchange(
                stage=stage,
                round=round_number,
                challenger=challenger,
                target=target,
                challenge=challenge_text,
                response=response_text,
                challenger_before=challenger_before,
                challenger_after=_snapshot(states[challenger]),
                target_before=target_before,
                target_after=_snapshot(states[target]),
                similarity_to_previous=similarity,
                termination=None if termination == "continue" else termination,
            )
            exchanges.append(exchange)
            if termination != "continue":
                break

        winner = self._more_confident((first, second), states)
        trace.append(
            {
                "event": "pair_debate_complete",
                "stage": stage,
                "pair": [first, second],
                "termination": termination,
                "winner": winner,
                "winner_entropy": states[winner].entropy,
            }
        )
        return winner

    def analyze_article(self, article: Article) -> ArticleResult:
        if len(article.text) > self.config.method.max_article_characters:
            raise ValueError(
                f"{article.id} exceeds max_article_characters; full text was not truncated"
            )
        started = time.monotonic()
        states = self._run_initial_analyses(article)
        initial_states = copy.deepcopy(states)
        exchanges: list[DebateExchange] = []
        trace: list[dict[str, Any]] = []
        counts = Counter(analysis.label for analysis in states.values())

        if len(counts) == 1:
            route = "unanimous"
            winner = None
            trace.append({"event": "early_exit", "reason": "initial_unanimity"})
        elif len(counts) == 2:
            route = "majority_debate"
            majority_label, _ = counts.most_common(1)[0]
            majority_names = sorted(
                name for name, state in states.items() if state.label is majority_label
            )
            dissenter = next(
                name for name, state in states.items() if state.label is not majority_label
            )
            representative = self._more_confident(majority_names, states)
            trace.append(
                {
                    "event": "route",
                    "type": "majority_2v1",
                    "majority_label": majority_label.value,
                    "representative": representative,
                    "dissenter": dissenter,
                    "selection": "lowest_entropy_in_majority",
                }
            )
            winner = self._debate_pair(
                article,
                states,
                representative,
                dissenter,
                "majority",
                exchanges,
                trace,
            )
        else:
            route = "all_different_debate"
            names = sorted(states)
            first, second, similarity = most_divergent_pair(
                names, [states[name].reasoning for name in names], self.embedder
            )
            remaining = next(name for name in names if name not in {first, second})
            trace.append(
                {
                    "event": "route",
                    "type": "all_different_1v1v1",
                    "stage_1_pair": [first, second],
                    "pair_cosine_similarity": similarity,
                    "remaining_agent": remaining,
                    "selection": "minimum_reasoning_cosine_similarity",
                }
            )
            representative = self._debate_pair(
                article, states, first, second, "all_different_stage_1", exchanges, trace
            )
            if self._panel_unanimous(states):
                winner = representative
                trace.append({"event": "early_exit", "reason": "panel_unanimity_after_stage_1"})
            else:
                winner = self._debate_pair(
                    article,
                    states,
                    representative,
                    remaining,
                    "all_different_stage_2",
                    exchanges,
                    trace,
                )

        panel_unanimous = self._panel_unanimous(states)
        if panel_unanimous:
            final_score = sum(state.score for state in states.values()) / 3.0
            final_label = score_to_label(final_score, self.threshold)
            trace.append(
                {
                    "event": "resolution",
                    "type": "panel_mean",
                    "score": final_score,
                    "label": final_label.value,
                }
            )
        else:
            if winner is None:
                winner = self._more_confident(tuple(states), states)
            final_score = states[winner].score
            final_label = states[winner].label
            trace.append(
                {
                    "event": "resolution",
                    "type": "confidence_winner",
                    "winner": winner,
                    "score": final_score,
                    "label": final_label.value,
                    "entropy": states[winner].entropy,
                }
            )

        return ArticleResult(
            article=article,
            route=route,
            final_label=final_label,
            final_score=round(final_score, 4),
            winning_agent=winner,
            panel_unanimous=panel_unanimous,
            initial_analyses=initial_states,
            final_analyses=copy.deepcopy(states),
            debate=exchanges,
            decision_trace=trace,
            embedding_backend=self.embedder.name,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

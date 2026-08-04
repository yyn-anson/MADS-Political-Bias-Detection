"""Configuration loading using only Python's standard library."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    name: str
    model: str
    temperature: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class RuntimeConfig:
    endpoint: str = "http://127.0.0.1:11434/v1"
    request_timeout_seconds: int = 240
    retries: int = 2
    max_output_tokens: int = 700
    parallel_initial_analysis: bool = True


@dataclass(frozen=True)
class MethodConfig:
    direction_threshold: float = 1.0
    max_debate_rounds: int = 3
    stagnation_similarity: float = 0.90
    embedding_backend: str = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_article_characters: int = 60_000


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    agents: tuple[AgentConfig, ...] = (
        AgentConfig("analyst_a", "qwen2.5vl:3b", 0.15, 17),
        AgentConfig("analyst_b", "qwen2.5vl:3b", 0.25, 29),
        AgentConfig("analyst_c", "qwen2.5vl:3b", 0.35, 43),
    )

    def validate(self) -> AppConfig:
        if len(self.agents) != 3:
            raise ValueError("MADS requires exactly three agents")
        if len({agent.name for agent in self.agents}) != 3:
            raise ValueError("agent names must be unique")
        if not 0 < self.method.direction_threshold <= 3:
            raise ValueError("direction_threshold must be in (0, 3]")
        if self.method.max_debate_rounds < 1:
            raise ValueError("max_debate_rounds must be at least 1")
        if not 0 <= self.method.stagnation_similarity <= 1:
            raise ValueError("stagnation_similarity must be in [0, 1]")
        if self.method.embedding_backend not in {"auto", "minilm", "hashing"}:
            raise ValueError("embedding_backend must be auto, minilm, or hashing")
        return self

    def with_overrides(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        embedding_backend: str | None = None,
        max_debate_rounds: int | None = None,
        parallel_initial_analysis: bool | None = None,
    ) -> AppConfig:
        runtime = self.runtime
        method = self.method
        agents = self.agents
        if endpoint:
            runtime = replace(runtime, endpoint=endpoint.rstrip("/"))
        if parallel_initial_analysis is not None:
            runtime = replace(runtime, parallel_initial_analysis=parallel_initial_analysis)
        if model:
            agents = tuple(replace(agent, model=model) for agent in agents)
        if embedding_backend:
            method = replace(method, embedding_backend=embedding_backend)
        if max_debate_rounds is not None:
            method = replace(method, max_debate_rounds=max_debate_rounds)
        return replace(self, runtime=runtime, method=method, agents=agents).validate()


def load_config(path: str | Path = "mads.toml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig().validate()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    runtime = RuntimeConfig(**raw.get("runtime", {}))
    method = MethodConfig(**raw.get("method", {}))
    agents = tuple(AgentConfig(**item) for item in raw.get("agents", []))
    if not agents:
        agents = AppConfig().agents
    return AppConfig(runtime=runtime, method=method, agents=agents).validate()

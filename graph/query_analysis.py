from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from schemas import QueryPlan


ConversationHistory = Sequence[tuple[str, str]]
KeywordOutput = tuple[Sequence[str], Sequence[str]]


class QueryAnalyzer(Protocol):
    """Synchronous query-analysis implementation."""

    def analyze(self, query: str, history: ConversationHistory) -> QueryPlan:
        ...


class AsyncQueryAnalyzer(Protocol):
    """Asynchronous query-analysis implementation."""

    def analyze(
        self,
        query: str,
        history: ConversationHistory,
    ) -> Awaitable[QueryPlan]:
        ...


@dataclass(frozen=True)
class PassThroughQueryAnalyzer:
    """PathRAG-compatible analyzer that retains the original query."""

    strategy: str = "pathrag"
    mode: str | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)

    def analyze(self, query: str, history: ConversationHistory) -> QueryPlan:
        return QueryPlan(
            query=query,
            history=history,
            strategy=self.strategy,
            mode=self.mode,
            settings=dict(self.settings),
        )


@dataclass(frozen=True)
class KeywordQueryAnalyzer:
    """Adapter for synchronous LightRAG-style keyword extraction."""

    extract_keywords: Callable[[str, ConversationHistory], KeywordOutput]
    mode: str = "hybrid"
    strategy: str = "lightrag"
    settings: Mapping[str, Any] = field(default_factory=dict)

    def analyze(self, query: str, history: ConversationHistory) -> QueryPlan:
        high_level, low_level = self.extract_keywords(query, history)
        return _keyword_plan(
            query,
            history,
            high_level,
            low_level,
            strategy=self.strategy,
            mode=self.mode,
            settings=self.settings,
        )


@dataclass(frozen=True)
class AsyncKeywordQueryAnalyzer:
    """Adapter for asynchronous LightRAG-style keyword extraction."""

    extract_keywords: Callable[
        [str, ConversationHistory],
        Awaitable[KeywordOutput],
    ]
    mode: str = "hybrid"
    strategy: str = "lightrag"
    settings: Mapping[str, Any] = field(default_factory=dict)

    async def analyze(self, query: str, history: ConversationHistory) -> QueryPlan:
        high_level, low_level = await self.extract_keywords(query, history)
        return _keyword_plan(
            query,
            history,
            high_level,
            low_level,
            strategy=self.strategy,
            mode=self.mode,
            settings=self.settings,
        )


def analyze_query(
    query: str,
    history: ConversationHistory,
    analyzer: QueryAnalyzer,
) -> QueryPlan:
    """Analyze one query through a synchronous typed connector."""

    normalized_history = _validate_input(query, history)
    plan = analyzer.analyze(query, normalized_history)
    return _validate_plan(plan, query, normalized_history)


async def analyze_query_async(
    query: str,
    history: ConversationHistory,
    analyzer: AsyncQueryAnalyzer,
) -> QueryPlan:
    """Analyze one query through an asynchronous typed connector."""

    normalized_history = _validate_input(query, history)
    plan = await analyzer.analyze(query, normalized_history)
    return _validate_plan(plan, query, normalized_history)


def _validate_input(query, history):
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("history must be a sequence")
    normalized = tuple(tuple(turn) for turn in history)
    for turn in normalized:
        if len(turn) != 2 or not all(isinstance(value, str) for value in turn):
            raise TypeError("history must contain string (role, message) pairs")
    return normalized


def _validate_plan(plan, query, history):
    if not isinstance(plan, QueryPlan):
        raise TypeError("analyzer must return a QueryPlan")
    if plan.query != query:
        raise ValueError("query plan must preserve the input query")
    if tuple(plan.history) != history:
        raise ValueError("query plan must preserve the supplied history")
    return plan


def _keyword_plan(
    query,
    history,
    high_level,
    low_level,
    *,
    strategy,
    mode,
    settings,
):
    high_level = list(high_level)
    low_level = list(low_level)
    resolved_mode = mode
    if high_level or low_level:
        if not low_level and resolved_mode in {"local", "hybrid"}:
            resolved_mode = "global"
        if not high_level and resolved_mode in {"global", "hybrid"}:
            resolved_mode = "local"
    return QueryPlan(
        query=query,
        history=history,
        high_level_keywords=high_level,
        low_level_keywords=low_level,
        strategy=strategy,
        mode=resolved_mode,
        settings=dict(settings),
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from schemas import (
    QueryPlan,
    RetrievalCandidates,
    RetrievedContext,
    retrieval_candidates_from_legacy,
)


class CandidateRetriever(Protocol):
    """Query-plan to scored-candidates connector."""

    def retrieve(self, plan: QueryPlan) -> RetrievalCandidates:
        ...


class ContextBuilder(Protocol):
    """Scored-candidates to bounded-context connector."""

    def build(
        self,
        plan: QueryPlan,
        candidates: RetrievalCandidates,
    ) -> RetrievedContext:
        ...


class Reranker(Protocol):
    """Optional context-window reranking connector."""

    def rerank(self, context: RetrievedContext) -> RetrievedContext:
        ...


@dataclass(frozen=True)
class LegacyCandidateRetriever:
    """Adapter for dictionary/dataclass candidate lookup implementations."""

    retrieve_legacy: Callable[[QueryPlan], Mapping[str, Any]]

    def retrieve(self, plan: QueryPlan) -> RetrievalCandidates:
        return retrieval_candidates_from_legacy(
            plan,
            self.retrieve_legacy(plan),
        )


def retrieve_candidates(
    plan: QueryPlan,
    retriever: CandidateRetriever,
) -> RetrievalCandidates:
    """Retrieve scored candidates through one typed boundary."""

    _require_plan(plan)
    candidates = retriever.retrieve(plan)
    if not isinstance(candidates, RetrievalCandidates):
        raise TypeError("candidate retriever must return RetrievalCandidates")
    if candidates.query_plan != plan:
        raise ValueError("retrieval candidates must preserve the query plan")
    return candidates


def build_context(
    plan: QueryPlan,
    candidates: RetrievalCandidates,
    builder: ContextBuilder,
) -> RetrievedContext:
    """Build bounded answer context from scored candidates."""

    _require_plan(plan)
    if not isinstance(candidates, RetrievalCandidates):
        raise TypeError("candidates must be RetrievalCandidates")
    if candidates.query_plan != plan:
        raise ValueError("retrieval candidates must preserve the query plan")
    context = builder.build(plan, candidates)
    return _require_context(context, plan, "context builder")


def rerank_context(
    context: RetrievedContext,
    reranker: Reranker | None = None,
) -> RetrievedContext:
    """Return unchanged context or pass it through a typed reranker."""

    if not isinstance(context, RetrievedContext):
        raise TypeError("context must be a RetrievedContext")
    if reranker is None:
        return context
    reranked = reranker.rerank(context)
    return _require_context(reranked, context.query_plan, "reranker")


def _require_plan(plan):
    if not isinstance(plan, QueryPlan):
        raise TypeError("plan must be a QueryPlan")


def _require_context(context, plan, producer):
    if not isinstance(context, RetrievedContext):
        raise TypeError(f"{producer} must return a RetrievedContext")
    if context.query_plan != plan:
        raise ValueError(f"{producer} must preserve the query plan")
    return context

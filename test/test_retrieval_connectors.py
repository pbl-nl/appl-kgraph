from dataclasses import dataclass

import pytest

from retrieval import (
    LegacyCandidateRetriever,
    build_context,
    rerank_context,
    retrieve_candidates,
)
from schemas import (
    ContextWindow,
    EntityCandidate,
    QueryPlan,
    RetrievalCandidates,
    RetrievedContext,
)


def test_retrieval_connectors_keep_candidate_and_context_stages_separate():
    plan = QueryPlan(query="Question", strategy="pathrag")
    candidates = RetrievalCandidates(
        query_plan=plan,
        entities=[
            EntityCandidate(
                name="Alpha",
                type="Thing",
                description="Entity",
                score=0.8,
            )
        ],
    )
    context = RetrievedContext(
        query_plan=plan,
        context_windows=[
            ContextWindow(label="entity::Alpha", text="Evidence", score=0.8)
        ],
    )
    calls = []

    class Retriever:
        def retrieve(self, received_plan):
            calls.append(("retrieve", received_plan))
            return candidates

    class Builder:
        def build(self, received_plan, received_candidates):
            calls.append(("build", received_plan, received_candidates))
            return context

    retrieved = retrieve_candidates(plan, Retriever())
    built = build_context(plan, retrieved, Builder())

    assert retrieved is candidates
    assert built is context
    assert calls == [
        ("retrieve", plan),
        ("build", plan, candidates),
    ]


def test_legacy_candidate_adapter_returns_shared_contract():
    plan = QueryPlan(query="Question", strategy="pathrag")

    adapter = LegacyCandidateRetriever(
        lambda received_plan: {
            "entity_matches": [
                {
                    "name": "Alpha",
                    "type": "Thing",
                    "description": "Entity",
                    "score": 0.9,
                }
            ],
            "strategy": received_plan.strategy,
        }
    )

    candidates = retrieve_candidates(plan, adapter)

    assert candidates.entities[0].name == "Alpha"
    assert candidates.metadata == {"strategy": "pathrag"}


def test_rerank_connector_is_optional_and_preserves_plan():
    plan = QueryPlan(query="Question")
    low = ContextWindow(label="low", text="Low", score=0.1)
    high = ContextWindow(label="high", text="High", score=0.9)
    context = RetrievedContext(query_plan=plan, context_windows=[low, high])

    @dataclass
    class ScoreReranker:
        def rerank(self, received):
            return RetrievedContext(
                query_plan=received.query_plan,
                context_windows=sorted(
                    received.context_windows,
                    key=lambda window: window.score,
                    reverse=True,
                ),
            )

    assert rerank_context(context) is context
    assert rerank_context(context, ScoreReranker()).context_windows == [high, low]


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ("retrieve", "RetrievalCandidates"),
        ("build", "RetrievedContext"),
        ("rerank", "RetrievedContext"),
    ],
)
def test_retrieval_connectors_reject_invalid_outputs(stage, message):
    plan = QueryPlan(query="Question")
    candidates = RetrievalCandidates(query_plan=plan)
    context = RetrievedContext(query_plan=plan)

    class Invalid:
        def retrieve(self, plan):
            return {}

        def build(self, plan, candidates):
            return {}

        def rerank(self, context):
            return {}

    with pytest.raises(TypeError, match=message):
        if stage == "retrieve":
            retrieve_candidates(plan, Invalid())
        elif stage == "build":
            build_context(plan, candidates, Invalid())
        else:
            rerank_context(context, Invalid())


def test_retrieval_connectors_reject_changed_query_plan():
    plan = QueryPlan(query="Original")
    changed = QueryPlan(query="Changed")

    class ChangedRetriever:
        def retrieve(self, plan):
            return RetrievalCandidates(query_plan=changed)

    with pytest.raises(ValueError, match="preserve the query plan"):
        retrieve_candidates(plan, ChangedRetriever())


def test_context_window_requires_stable_source_references():
    with pytest.raises(ValueError, match="unique"):
        ContextWindow(
            label="window",
            text="Evidence",
            source_refs=("chunk-1", "chunk-1"),
        )

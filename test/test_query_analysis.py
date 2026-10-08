import asyncio

import pytest

from query_analysis import (
    AsyncKeywordQueryAnalyzer,
    KeywordQueryAnalyzer,
    PassThroughQueryAnalyzer,
    analyze_query,
    analyze_query_async,
)
from schemas import QueryPlan


def test_pathrag_analyzer_returns_explicit_pass_through_plan():
    history = [("user", "Earlier question"), ("assistant", "Earlier answer")]

    plan = analyze_query(
        "How are Alpha and Beta related?",
        history,
        PassThroughQueryAnalyzer(
            mode="hybrid",
            settings={"entity_top_k": 5},
        ),
    )

    assert plan == QueryPlan(
        query="How are Alpha and Beta related?",
        history=tuple(history),
        strategy="pathrag",
        mode="hybrid",
        settings={"entity_top_k": 5},
    )


@pytest.mark.parametrize(
    ("keywords", "expected_mode"),
    [
        ((["theme"], ["Alpha"]), "hybrid"),
        ((["theme"], []), "global"),
        (([], ["Alpha"]), "local"),
        (([], []), "hybrid"),
    ],
)
def test_keyword_analyzer_records_keywords_and_resolves_mode(
    keywords,
    expected_mode,
):
    calls = []

    def extract(query, history):
        calls.append((query, history))
        return keywords

    plan = analyze_query(
        "Question",
        [],
        KeywordQueryAnalyzer(extract, settings={"history_turns": 4}),
    )

    assert calls == [("Question", ())]
    assert plan.high_level_keywords == keywords[0]
    assert plan.low_level_keywords == keywords[1]
    assert plan.mode == expected_mode
    assert plan.strategy == "lightrag"
    assert plan.settings == {"history_turns": 4}


def test_async_keyword_analyzer_uses_same_output_contract():
    async def extract(query, history):
        return ["theme"], ["Alpha"]

    plan = asyncio.run(
        analyze_query_async(
            "Question",
            [],
            AsyncKeywordQueryAnalyzer(extract),
        )
    )

    assert isinstance(plan, QueryPlan)
    assert plan.high_level_keywords == ["theme"]
    assert plan.low_level_keywords == ["Alpha"]


def test_query_connector_rejects_analyzer_that_changes_query():
    class RewritingAnalyzer:
        def analyze(self, query, history):
            return QueryPlan(query="different", history=history)

    with pytest.raises(ValueError, match="preserve the input query"):
        analyze_query("original", [], RewritingAnalyzer())


@pytest.mark.parametrize(
    ("query", "history", "message"),
    [
        (None, [], "query"),
        ("Question", "history", "history"),
        ("Question", [("user",)], "pairs"),
        ("Question", [("user", None)], "pairs"),
    ],
)
def test_query_connector_rejects_invalid_inputs(query, history, message):
    with pytest.raises(TypeError, match=message):
        analyze_query(query, history, PassThroughQueryAnalyzer())

from dataclasses import replace
import json

import pytest

from test.rag_fixtures import corpus as corpus_fixture
from test.rag_fixtures import configure_rag, observe_stages
from test.production_fixtures import DeterministicChat
from schemas import AnswerResult
import lightrag


corpus = corpus_fixture
pytestmark = pytest.mark.storage_integration


def keywords(monkeypatch, value):
    async def extract(chat, query, history, turns, **kwargs):
        return value

    monkeypatch.setattr(lightrag, "extract_keywords", extract)


@pytest.mark.parametrize("mode", ["local", "global", "hybrid", "mix", "naive"])
def test_lightrag_public_entrypoint_uses_production_connectors(
    corpus, monkeypatch, mode
):
    configure_rag(lightrag, monkeypatch)
    monkeypatch.setattr(
        lightrag,
        "settings",
        replace(
            lightrag.settings,
            retrieval=replace(lightrag.settings.retrieval, light_mode=mode),
        ),
    )
    keywords(monkeypatch, (["partnership"], ["Alpha"]))
    calls = observe_stages(lightrag, monkeypatch)
    rag = lightrag.LightRAG(project_paths=corpus)
    try:
        result = rag.retrieve("How are Alpha and Beta related?", [("user", "Earlier")])
        assert isinstance(result, AnswerResult) and result.answer == "Grounded answer"
        assert result.context.query_plan.mode == mode
        assert result.context.chunks
        assert len({chunk.chunk_uuid for chunk in result.context.chunks}) == len(
            result.context.chunks
        )
        assert calls == [
            "retrieve_candidates",
            "build_context",
            "rerank_context",
            "generate_answer",
            "audit",
        ]
        system_prompt, query, _, _ = DeterministicChat.calls[0]
        assert query == "How are Alpha and Beta related?"
        assert "Earlier" in system_prompt
        audit = json.loads(
            next(corpus.audit_logs_dir.glob("*lightrag*.json")).read_text()
        )
        assert [row["chunk_uuid"] for row in audit["retrieved_chunks"]] == [
            chunk.chunk_uuid for chunk in result.context.chunks
        ]
    finally:
        rag.close()


@pytest.mark.parametrize(
    "value,mode", [(([], ["Alpha"]), "local"), ((["partnership"], []), "global")]
)
def test_keyword_fallback(corpus, monkeypatch, value, mode):
    configure_rag(lightrag, monkeypatch)
    keywords(monkeypatch, value)
    monkeypatch.setattr(
        lightrag,
        "settings",
        replace(
            lightrag.settings,
            retrieval=replace(lightrag.settings.retrieval, light_mode="hybrid"),
        ),
    )
    rag = lightrag.LightRAG(project_paths=corpus)
    try:
        assert rag.retrieve("Alpha").context.query_plan.mode == mode
    finally:
        rag.close()


def test_empty_keywords_avoids_answer_model_but_naive_still_retrieves(
    corpus, monkeypatch
):
    configure_rag(lightrag, monkeypatch)
    keywords(monkeypatch, ([], []))
    rag = lightrag.LightRAG(project_paths=corpus)
    try:
        result = rag.retrieve("Unknown")
        assert (
            result.answer == lightrag.PROMPTS["fail_response"]
            and result.context.chunks == []
        )
        assert DeterministicChat.calls == []
        monkeypatch.setattr(
            lightrag,
            "settings",
            replace(
                lightrag.settings,
                retrieval=replace(lightrag.settings.retrieval, light_mode="naive"),
            ),
        )
        assert rag.retrieve("Alpha").context.chunks
    finally:
        rag.close()


def test_selection_and_reranking_use_final_context(corpus, monkeypatch):
    configure_rag(lightrag, monkeypatch)
    keywords(monkeypatch, (["partnership"], ["Alpha"]))
    monkeypatch.setattr(
        lightrag,
        "settings",
        replace(
            lightrag.settings,
            retrieval=replace(
                lightrag.settings.retrieval, enable_rerank=True, chunk_top_k=1
            ),
        ),
    )
    monkeypatch.setattr(
        lightrag,
        "rerank",
        lambda query, docs, top_n, **kwargs: list(reversed(range(len(docs)))),
    )
    rag = lightrag.LightRAG(project_paths=corpus)
    try:
        result = rag.retrieve("Alpha", allowed_document_names=["a.txt"])
        assert (
            len(result.context.chunks) == 1
            and result.context.chunks[0].filename == "a.txt"
        )
        assert result.context.chunks[0].text in DeterministicChat.calls[0][0]
    finally:
        rag.close()

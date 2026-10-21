import pytest
from test.rag_fixtures import corpus as corpus_fixture
from test.rag_fixtures import configure_rag, observe_stages
from test.production_fixtures import DeterministicChat
from schemas import AnswerResult
import pathrag
import json


corpus = corpus_fixture
pytestmark = pytest.mark.storage_integration


def test_pathrag_public_entrypoint_uses_production_connectors(corpus, monkeypatch):
    configure_rag(pathrag, monkeypatch)
    calls = observe_stages(pathrag, monkeypatch)
    rag = pathrag.PathRAG(project_paths=corpus)
    history = [("user", "Earlier question"), ("assistant", "Earlier answer")]
    try:
        result = rag.retrieve("How are Alpha and Beta related?", history)
        assert isinstance(result, AnswerResult)
        assert result.answer == "Grounded answer"
        assert result.context.query_plan.strategy == "pathrag"
        assert result.context.query_plan.history == tuple(history)
        assert result.context.entities and result.context.chunks
        assert result.context.context_windows
        assert calls == [
            "retrieve_candidates",
            "build_context",
            "rerank_context",
            "generate_answer",
            "audit",
        ]
        prompt = DeterministicChat.calls[0][1]
        assert "Earlier question" in prompt and "Earlier answer" in prompt
        assert "Alpha" in prompt and "chunk_uuid" in prompt
        audit = json.loads(
            next(corpus.audit_logs_dir.glob("*pathrag*.json")).read_text()
        )
        assert audit["answer"] == result.answer
        assert (
            audit["retrieved_chunks"][0]["chunk_uuid"]
            == result.context.chunks[0].chunk_uuid
        )
        assert audit["conversation_history"] == [list(turn) for turn in history]
    finally:
        rag.close()


def test_pathrag_selection_filters_graph_expansion_and_sources(corpus, monkeypatch):
    configure_rag(pathrag, monkeypatch)
    rag = pathrag.PathRAG(project_paths=corpus)
    try:
        result = rag.retrieve("Alpha", allowed_document_names=["a.txt"])
        assert result.context.chunks
        assert {chunk.filename for chunk in result.context.chunks} == {"a.txt"}
    finally:
        rag.close()


def test_path_context_builder_runs_from_candidates_without_storage(corpus, monkeypatch):
    configure_rag(pathrag, monkeypatch)
    rag = pathrag.PathRAG(project_paths=corpus)
    try:
        from query_analysis import analyze_query, PassThroughQueryAnalyzer
        from dataclasses import asdict

        plan = analyze_query(
            "Alpha",
            [],
            PassThroughQueryAnalyzer(settings=asdict(pathrag.settings.retrieval)),
        )
        candidates = pathrag.PathCandidateRetriever(rag._storage).retrieve(plan)
    finally:
        rag.close()
    context = pathrag.PathContextBuilder().build(plan, candidates)
    assert context.chunks and context.context_windows

"""Exercise actual prompts, response parsing, caching and async model bridges."""

from dataclasses import replace

import pytest

from test.production_fixtures import DeterministicEmbedder
from db_storage import Storage
import ingestion
import llm
import lightrag
import pathrag
from project_paths import resolve_project_paths


@pytest.mark.storage_integration
@pytest.mark.parametrize("strategy", ["pathrag", "lightrag"])
def test_complete_pipeline_substitutes_only_model_and_embedding_calls(
    tmp_path, monkeypatch, strategy
):
    calls = []

    class Chat:
        model = "fixture-model"

        def generate(self, prompt, system=None, **kwargs):
            calls.append((system, prompt))
            if system and "extract entities" in system:
                return '("entity"<|>"Alpha"<|>"Organization"<|>"An organization")##("relationship"<|>"Alpha"<|>"Beta"<|>"Works with"<|>"partnership"<|>1)<|COMPLETE|>'
            if system and "validate extraction" in system:
                return '{"missing_entities": [], "missing_relationships": [], "summary": "complete"}'
            if "high_level_keywords" in prompt and "low_level_keywords" in prompt:
                return '{"high_level_keywords": ["partnership"], "low_level_keywords": ["Alpha"]}'
            return "A grounded answer from the actual prompt pipeline"

    monkeypatch.setattr(llm.Chat, "singleton", lambda: Chat())
    make_storage = lambda paths=None: Storage(paths, DeterministicEmbedder())
    monkeypatch.setattr(ingestion, "Storage", make_storage)
    module = pathrag if strategy == "pathrag" else lightrag
    monkeypatch.setattr(module, "Storage", make_storage)
    monkeypatch.setattr(
        module,
        "settings",
        replace(
            module.settings,
            retrieval=replace(module.settings.retrieval, enable_rerank=False),
        ),
    )
    document = tmp_path / "report.txt"
    document.write_text("Alpha works with Beta.")
    first = ingestion.ingest_paths([document], documents_root=tmp_path)
    assert first["entity_count"] == 2 and first["relation_count"] == 1
    assert (
        ingestion.ingest_paths([document], documents_root=tmp_path)["skipped_files"]
        == 1
    )
    assert (
        sum(bool(system and "extract entities" in system) for system, _ in calls) == 1
    )
    paths = resolve_project_paths(tmp_path)
    rag = (pathrag.PathRAG if strategy == "pathrag" else lightrag.LightRAG)(
        project_paths=paths
    )
    try:
        result = rag.retrieve("Explain the partnership")
        assert result.answer.startswith("A grounded answer")
        assert result.context.entities and result.context.chunks
        if strategy == "lightrag":
            assert result.context.query_plan.low_level_keywords == ["Alpha"]
        assert result.context.chunks[0].doc_id
    finally:
        rag.close()


def test_reranker_preserves_distinct_chunks_with_identical_text(monkeypatch):
    class Ranker:
        def __init__(self, **kwargs):
            pass

        def rerank(self, request):
            return list(reversed(request.passages))

    monkeypatch.setattr(lightrag, "Ranker", Ranker)
    assert lightrag.rerank("Question", [{"text": "same"}, {"text": "same"}], 2) == [
        1,
        0,
    ]


@pytest.mark.parametrize(
    "response", ["[]", "null", '{"low_level_keywords": "Alpha"}', "invalid JSON"]
)
def test_keyword_parser_handles_malformed_model_output(response):
    import asyncio

    class Chat:
        async def generate(self, *args, **kwargs):
            return response

    assert asyncio.run(lightrag.extract_keywords(Chat(), "Question")) == ([], [])

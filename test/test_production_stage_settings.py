"""Production adapters honor supplied contracts without opening a RAG store."""

import asyncio
from dataclasses import asdict, replace

import networkx as nx
import pytest

from answer_generation import generate_answer_async
import lightrag
import pathrag
from schemas import Chunk, QueryPlan, RetrievedContext


@pytest.mark.parametrize(
    "module, retriever",
    [
        (pathrag, pathrag.PathCandidateRetriever),
        (lightrag, lightrag.LightCandidateRetriever),
    ],
)
def test_candidate_lookup_uses_plan_settings(module, retriever, monkeypatch):
    config = replace(
        module.settings.retrieval, entity_top_k=1, relation_top_k=2, chunk_top_k=3
    )
    plan = QueryPlan(
        "Alpha",
        strategy=module.__name__,
        mode="mix",
        high_level_keywords=["topic"],
        low_level_keywords=["Alpha"],
        settings=asdict(config),
    )
    monkeypatch.setattr(
        module,
        "settings",
        replace(
            module.settings,
            retrieval=replace(config, entity_top_k=9, relation_top_k=9, chunk_top_k=9),
        ),
    )
    calls = []

    class Store:
        graph = nx.Graph()

        def query_entities(self, text, limit):
            calls.append(("entities", limit))
            return []

        def query_relations(self, text, limit):
            calls.append(("relations", limit))
            return []

        def query_chunks(self, text, limit):
            calls.append(("chunks", limit))
            return []

    candidates = retriever(Store()).retrieve(plan)
    assert candidates.query_plan == plan
    assert calls == [("entities", 1), ("relations", 2), ("chunks", 3)]


@pytest.mark.parametrize("module", [pathrag, lightrag])
def test_supplied_context_answer_uses_snapshot_and_needs_no_storage(
    module, monkeypatch
):
    config = replace(
        module.settings.retrieval,
        history_turns=1,
        answer_max_tokens=73,
        llm_temperature=0.2,
        response_type="Bullet list",
    )
    history = [("user", "Discard this older message"), ("assistant", "Latest message")]
    plan = QueryPlan(
        "Alpha?",
        strategy=module.__name__,
        mode="naive",
        settings=asdict(config),
        history=history,
    )
    context = RetrievedContext(
        plan, chunks=[Chunk("source", "doc", 0, "report.txt", "Evidence", 8, 0, 0)]
    )
    calls = []

    class Chat:
        def __init__(self, system_prompt=None):
            self.system_prompt = system_prompt

        async def generate(self, prompt, **kwargs):
            calls.append((self.system_prompt, prompt, kwargs))
            return "Answer"

    def forbidden_storage(*args, **kwargs):
        raise AssertionError("answer generation must not open storage")

    monkeypatch.setattr(module, "Storage", forbidden_storage)
    monkeypatch.setattr(
        module,
        "settings",
        replace(
            module.settings,
            retrieval=replace(config, history_turns=10, answer_max_tokens=1),
        ),
    )
    generator = module.make_answer_generator(Chat() if module is pathrag else Chat)
    result = asyncio.run(generate_answer_async(plan.query, context, history, generator))
    assert result.answer == "Answer" and result.context == context
    system, prompt, options = calls[0]
    text = (system or "") + prompt
    assert "Evidence" in text and "Latest message" in text
    assert "Discard this older message" not in text
    assert options == {"max_tokens": 73, "temperature": 0.2}


def test_native_vector_results_retain_page_metadata_for_typed_context():
    from db_storage import Storage

    class VectorStore:
        def query(self, **kwargs):
            return [
                {
                    "ids": ["chunk"],
                    "documents": ["Evidence"],
                    "distances": [0.1],
                    "metadatas": [
                        {
                            "doc_id": "doc",
                            "filename": "report.pdf",
                            "chunk_id": 7,
                            "start_page": 2,
                            "end_page": 3,
                            "char_count": 8,
                        }
                    ],
                }
            ]

    storage = Storage.__new__(Storage)
    storage.chunk_vectors = VectorStore()
    row = storage._native_chunk_search("Question", 1)[0]
    assert (row["start_page"], row["end_page"], row["chunk_id"]) == (2, 3, 7)
    assert row["chunk_uuid"] == "chunk" and row["score"] == pytest.approx(0.9)


def test_selected_document_endpoint_does_not_admit_an_excluded_relation():
    from schemas import RelationCandidate

    graph = nx.Graph()
    graph.add_node("Alpha", filepath="a.txt", chunk_uuids=["a"])
    graph.add_node("Hidden", filepath="b.txt", chunk_uuids=["b"])
    graph.add_edge(
        "Alpha",
        "Hidden",
        filepath="b.txt",
        description="Excluded fact",
        chunk_uuids=["b"],
    )

    class Store:
        def __init__(self):
            self.graph = graph

        def query_entities(self, *args, **kwargs):
            return []

        def query_chunks(self, *args, **kwargs):
            return []

        def query_relations(self, *args, **kwargs):
            return [RelationCandidate("Alpha", "Hidden", "Excluded fact", score=0.9)]

        def get_node(self, name):
            return graph.nodes.get(name)

    plan = QueryPlan(
        "Alpha", strategy="pathrag", metadata={"allowed_document_names": ["a.txt"]}
    )
    candidates = pathrag.PathCandidateRetriever(Store()).retrieve(plan)
    assert candidates.relations == []
    copied = nx.node_link_graph(candidates.metadata["graph_data"], edges="edges")
    assert list(copied.nodes) == ["Alpha"] and list(copied.edges) == []

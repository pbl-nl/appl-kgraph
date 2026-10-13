import os
import sys
from pathlib import Path
from types import SimpleNamespace


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")
sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

import graph.pathrag as pathrag


class _StorageAdapterFake:
    def __init__(self, *, entities=(), relations=(), chunks=()):
        self.entities = list(entities)
        self.relations = list(relations)
        self.chunks = list(chunks)
        self.calls = []

    def query_entities(self, question, *, limit):
        self.calls.append(("entities", question, limit))
        return self.entities

    def query_relations(self, question, *, limit):
        self.calls.append(("relations", question, limit))
        return self.relations

    def query_chunks(self, question, *, limit):
        self.calls.append(("chunks", question, limit))
        return self.chunks


class _ChatFake:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def generate(self, prompt, *, max_tokens, temperature):
        self.calls.append((prompt, max_tokens, temperature))
        return self.answer


def _retrieval_settings():
    return SimpleNamespace(
        entity_top_k=3,
        relation_top_k=4,
        chunk_top_k=5,
        hybrid_use_paths_for_global=True,
        path_use_top_entities=2,
        path_max_depth=3,
        path_threshold=0.2,
        path_alpha=0.7,
        path_max_windows=2,
        path_window_tokens=200,
        tiktoken_model="test-tokenizer",
        use_local_chunks=True,
        chunk_windows=2,
        chunk_window_tokens=150,
        use_local_graph=True,
        graph_depth=1,
        graph_windows=2,
        graph_window_tokens=150,
        local_max_windows=4,
        path_history_turns=2,
        response_type="concise",
        llm_max_tokens=321,
        llm_temperature=0.25,
    )


def _configure_retrieval_fakes(monkeypatch, storage, chat):
    adapter_initialization = []

    def make_adapter(**kwargs):
        adapter_initialization.append(kwargs)
        return storage

    monkeypatch.setattr(pathrag, "StorageAdapter", make_adapter)
    monkeypatch.setattr(pathrag, "RetrieveChat", lambda system_prompt=None: chat)
    monkeypatch.setattr(pathrag, "set_logger", lambda log_file: None)
    monkeypatch.setattr(
        pathrag,
        "settings",
        SimpleNamespace(
            retrieval=_retrieval_settings(),
            logging=SimpleNamespace(audit_enabled=False),
        ),
    )
    monkeypatch.setattr(
        pathrag,
        "PROMPTS",
        {
            "pathrag_response": (
                "CONTEXT\n{context_data}\nHISTORY\n{history}\n"
                "TYPE\n{response_type}\nQUESTION\n{user_prompt}"
            )
        },
    )
    return adapter_initialization


def test_pathrag_retrieve_returns_structured_matches_and_ordered_context(monkeypatch):
    entity = pathrag.EntityMatch(
        name="Alpha",
        type="Organization",
        description="Primary entity",
        score=0.9,
    )
    relation = pathrag.RelationMatch(
        source_name="Alpha",
        target_name="Beta",
        description="Works with",
        keywords="collaboration",
        score=0.8,
    )
    chunk = pathrag.ChunkMatch(
        chunk_uuid="chunk-1",
        document_id="doc-1",
        filename="report.txt",
        text="Supporting text",
        score=0.7,
    )
    global_window = pathrag.ContextWindow(
        label="global::path::Alpha-Beta",
        text="Global evidence",
        score=0.9,
    )
    chunk_window = pathrag.ContextWindow(
        label="local::chunk::report.txt",
        text="Chunk evidence",
        score=0.7,
    )
    graph_window = pathrag.ContextWindow(
        label="local::graph::Alpha",
        text="Graph evidence",
        score=0.9,
    )
    storage = _StorageAdapterFake(
        entities=[entity],
        relations=[relation],
        chunks=[chunk],
    )
    chat = _ChatFake("Grounded answer")
    adapter_initialization = _configure_retrieval_fakes(monkeypatch, storage, chat)
    window_calls = []

    def fake_path_windows(adapter, seeds, **settings):
        window_calls.append(("global", adapter, seeds, settings))
        return [global_window]

    def fake_chunk_windows(chunks, **settings):
        window_calls.append(("chunks", chunks, settings))
        return [chunk_window]

    def fake_graph_windows(adapter, seeds, **settings):
        window_calls.append(("graph", adapter, seeds, settings))
        return [graph_window]

    monkeypatch.setattr(pathrag, "build_path_windows", fake_path_windows)
    monkeypatch.setattr(pathrag, "build_chunk_windows", fake_chunk_windows)
    monkeypatch.setattr(pathrag, "build_graph_windows", fake_graph_windows)

    rag = pathrag.PathRAG(storage_paths="storage-config", system_prompt="system")
    result = rag.retrieve(
        "How are Alpha and Beta related?",
        conversation_history=[
            ("user", "Earlier question"),
            ("assistant", "Earlier answer"),
        ],
    )

    assert isinstance(result, pathrag.RetrievalResult)
    assert result.answer == "Grounded answer"
    assert result.entity_matches == [entity]
    assert result.relation_matches == [relation]
    assert result.chunk_matches == [chunk]
    assert result.context_windows == [global_window, chunk_window, graph_window]
    assert storage.calls == [
        ("entities", "How are Alpha and Beta related?", 3),
        ("relations", "How are Alpha and Beta related?", 4),
        ("chunks", "How are Alpha and Beta related?", 5),
    ]
    assert adapter_initialization == [
        {"paths": "storage-config", "graph_pickle_path": None}
    ]
    assert [call[0] for call in window_calls] == ["global", "chunks", "graph"]

    prompt, max_tokens, temperature = chat.calls[0]
    assert max_tokens == 321
    assert temperature == 0.25
    assert '"name": "Alpha"' in prompt
    assert '"source_name": "Alpha"' in prompt
    assert '"chunk_uuid": "chunk-1"' in prompt
    assert "Earlier question" in prompt
    assert "Earlier answer" in prompt
    assert "TYPE\nconcise" in prompt
    assert "QUESTION\nHow are Alpha and Beta related?" in prompt


def test_pathrag_retrieve_preserves_empty_result_shape(monkeypatch):
    storage = _StorageAdapterFake()
    chat = _ChatFake("No supporting evidence")
    _configure_retrieval_fakes(monkeypatch, storage, chat)

    rag = pathrag.PathRAG(system_prompt="system")
    result = rag.retrieve("Unknown topic")

    assert result == pathrag.RetrievalResult(
        answer="No supporting evidence",
        context_windows=[],
        entity_matches=[],
        relation_matches=[],
        chunk_matches=[],
    )
    prompt, max_tokens, temperature = chat.calls[0]
    assert '"context_windows": []' in prompt
    assert '"entity_matches": []' in prompt
    assert '"relation_matches": []' in prompt
    assert '"chunk_matches": []' in prompt
    assert "HISTORY\n\nTYPE" in prompt
    assert max_tokens == 321
    assert temperature == 0.25

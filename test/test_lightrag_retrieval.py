import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")
sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

import graph.lightrag as lightrag


class _ChatFake:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def generate(self, prompt, *, max_tokens, temperature):
        self.calls.append((prompt, max_tokens, temperature))
        return self.answer


def _retrieval_settings():
    return SimpleNamespace(
        history_turns=3,
        light_mode="hybrid",
        entity_top_k=4,
        relation_top_k=5,
        chunk_top_k=6,
        top_k_chunk_per_entity=2,
        llm_max_tokens=432,
        llm_temperature=0.15,
    )


def _configure_retrieval_fakes(
    monkeypatch,
    *,
    keywords,
    contexts=((), (), ()),
    answer="Grounded answer",
):
    storage = object()
    keyword_chat = _ChatFake("unused keyword response")
    answer_chat = _ChatFake(answer)
    state = SimpleNamespace(
        adapter_initialization=[],
        chat_initialization=[],
        keyword_calls=[],
        context_calls=[],
        prompt_calls=[],
        storage=storage,
        keyword_chat=keyword_chat,
        answer_chat=answer_chat,
    )

    def make_adapter(**kwargs):
        state.adapter_initialization.append(kwargs)
        return storage

    def make_chat(system_prompt=None):
        state.chat_initialization.append(system_prompt)
        if len(state.chat_initialization) == 1:
            return keyword_chat
        return answer_chat

    async def fake_extract_keywords(chat, query, history, history_turns):
        state.keyword_calls.append((chat, query, history, history_turns))
        return keywords

    def fake_build_context(adapter, query, hl_keywords, ll_keywords, **settings):
        state.context_calls.append(
            (adapter, query, hl_keywords, ll_keywords, settings)
        )
        return tuple(list(items) for items in contexts)

    def fake_lightrag_prompt(**kwargs):
        state.prompt_calls.append(kwargs)
        return "formatted LightRAG system prompt"

    monkeypatch.setattr(lightrag, "StorageAdapter", make_adapter)
    monkeypatch.setattr(lightrag, "RetrieveChat", make_chat)
    monkeypatch.setattr(lightrag, "extract_keywords", fake_extract_keywords)
    monkeypatch.setattr(lightrag, "build_context", fake_build_context)
    monkeypatch.setattr(lightrag, "lightrag_prompt", fake_lightrag_prompt)
    monkeypatch.setattr(lightrag, "set_logger", lambda log_file: None)
    monkeypatch.setattr(
        lightrag,
        "settings",
        SimpleNamespace(
            retrieval=_retrieval_settings(),
            logging=SimpleNamespace(audit_enabled=False),
        ),
    )
    monkeypatch.setattr(lightrag, "PROMPTS", {"fail_response": "No keywords found"})
    return state


def test_lightrag_retrieve_returns_context_and_keywords(monkeypatch):
    entities = [
        {
            "id": "1",
            "entity": "Alpha",
            "type": "Organization",
            "description": "Primary entity",
        }
    ]
    relations = [
        {
            "id": "1",
            "entity1": "Alpha",
            "entity2": "Beta",
            "description": "Works with",
        }
    ]
    chunks = [
        {
            "id": "chunk-1",
            "text": "Supporting text",
            "source_type": "entity",
        }
    ]
    history = [("user", "Earlier question"), ("assistant", "Earlier answer")]
    state = _configure_retrieval_fakes(
        monkeypatch,
        keywords=(["partnership"], ["Alpha", "Beta"]),
        contexts=(entities, relations, chunks),
    )

    rag = lightrag.LightRAG(storage_paths="storage-config", system_prompt="initial")
    result = rag.retrieve("How are Alpha and Beta related?", history)

    assert result == lightrag.RetrievalResult(
        answer="Grounded answer",
        entities_context=entities,
        relations_context=relations,
        all_chunks=chunks,
        hl_keywords=["partnership"],
        ll_keywords=["Alpha", "Beta"],
    )
    assert state.adapter_initialization == [
        {"paths": "storage-config", "graph_pickle_path": None}
    ]
    assert state.keyword_calls == [
        (
            state.keyword_chat,
            "How are Alpha and Beta related?",
            history,
            3,
        )
    ]
    assert state.context_calls == [
        (
            state.storage,
            "How are Alpha and Beta related?",
            ["partnership"],
            ["Alpha", "Beta"],
            {
                "retrieval_mode": "hybrid",
                "top_k_entities": 4,
                "top_k_relations": 5,
                "top_k_chunks": 6,
                "top_k_chunk_per_entity": 2,
            },
        )
    ]
    assert state.prompt_calls == [
        {
            "entities_context": entities,
            "relations_context": relations,
            "all_chunks": chunks,
            "history": history,
        }
    ]
    assert state.chat_initialization == [
        "initial",
        "formatted LightRAG system prompt",
    ]
    assert state.answer_chat.calls == [
        ("How are Alpha and Beta related?", 432, 0.15)
    ]


@pytest.mark.parametrize(
    ("keywords", "expected_mode"),
    [
        (([], ["Alpha"]), "local"),
        ((["partnership"], []), "global"),
    ],
)
def test_lightrag_retrieve_falls_back_when_one_keyword_class_is_empty(
    monkeypatch,
    keywords,
    expected_mode,
):
    state = _configure_retrieval_fakes(monkeypatch, keywords=keywords)

    result = lightrag.LightRAG().retrieve("Question")

    assert result.hl_keywords == keywords[0]
    assert result.ll_keywords == keywords[1]
    assert state.context_calls[0][4]["retrieval_mode"] == expected_mode
    assert result.answer == "Grounded answer"


def test_lightrag_retrieve_returns_failure_result_for_empty_keywords(monkeypatch):
    state = _configure_retrieval_fakes(monkeypatch, keywords=([], []))

    result = lightrag.LightRAG().retrieve("Unknown topic")

    assert result == lightrag.RetrievalResult(
        answer="No keywords found",
        entities_context=[],
        relations_context=[],
        all_chunks=[],
        hl_keywords=[],
        ll_keywords=[],
    )
    assert state.context_calls == []
    assert state.prompt_calls == []
    assert state.chat_initialization == [None]
    assert state.answer_chat.calls == []

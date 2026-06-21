"""
Unit tests for LLM calling integration.

Covers three layers:
  1. Chat.generate — the base LLM client sends the right request and returns the response
  2. extract_from_chunks — the ingest path calls the LLM and parses entities/relationships
  3. PathRAG.aretrieve — the query path calls the LLM to produce an answer

All tests use mocks so no real API calls are made.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Must be set before any graph module is imported so that settings.load_settings()
# finds valid credentials and does not raise.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from graph.db_storage import StoragePaths
from graph.extractor import extract_from_chunks
from graph.llm import Chat
from graph.pathrag import PathRAG


# ─── 1. Chat.generate ────────────────────────────────────────────────────────


def test_chat_generate_calls_openai_and_returns_response(monkeypatch):
    """Chat.generate forwards the prompt to the OpenAI client and surfaces the reply."""
    fake_message = MagicMock()
    fake_message.content = "42 is the answer"
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_completion = MagicMock()
    fake_completion.choices = [fake_choice]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion

    monkeypatch.setattr("graph.llm.OpenAI", MagicMock(return_value=fake_client))

    chat = Chat()
    result = chat.generate("What is the answer?", system="Be concise.")

    assert result == "42 is the answer"
    fake_client.chat.completions.create.assert_called_once()
    messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
    assert any(m["role"] == "system" and "Be concise" in m["content"] for m in messages)
    assert any(m["role"] == "user" and "What is the answer?" in m["content"] for m in messages)


# ─── 2. Ingestion — extract_from_chunks ──────────────────────────────────────

# Minimal LLM extraction response in the expected delimiter format.
_EXTRACTION_RESPONSE = (
    '("entity"<|>"Alice"<|>"person"<|>"A researcher in AI")##'
    '("entity"<|>"Bob"<|>"person"<|>"A software developer")##'
    '("relationship"<|>"Alice"<|>"Bob"<|>"They co-author papers"<|>"collaboration"<|>0.9)##'
    "<|COMPLETE|>"
)


def test_extract_from_chunks_calls_llm_and_parses_entities():
    """extract_from_chunks calls the LLM once per chunk and parses entities and relationships."""
    fake_chat = MagicMock(spec=Chat)
    fake_chat.generate.return_value = _EXTRACTION_RESPONSE
    fake_chat.model = "test-model"

    # Fake storage: always report a cache miss so the LLM is actually called.
    fake_storage = MagicMock()
    fake_storage.get_llm_cache.return_value = None
    fake_storage.put_llm_cache.return_value = None

    chunk = {
        "chunk_uuid": "chunk-001",
        "text": "Alice and Bob co-author AI research papers.",
        "doc_id": "doc-001",
    }

    result = extract_from_chunks([chunk], client=fake_chat, storage=fake_storage)

    assert fake_chat.generate.called, "LLM was not called during extraction"

    entity_names = {e["name"] for e in result["entities"]}
    assert "Alice" in entity_names
    assert "Bob" in entity_names

    relations = result["relationships"]
    assert any(
        r["source_name"] == "Alice" and r["target_name"] == "Bob" for r in relations
    )


# ─── 3. Query — PathRAG.aretrieve ────────────────────────────────────────────


def test_pathrag_aretrieve_calls_llm_for_answer(tmp_path, monkeypatch):
    """PathRAG.aretrieve calls the LLM to generate an answer and returns it."""
    # Use Python-only vector search to avoid ChromaDB on any platform.
    monkeypatch.setenv("APPL_KGRAPH_FORCE_PYTHON_VECTOR_SEARCH", "1")

    paths = StoragePaths(
        documents_db=str(tmp_path / "docs.sqlite"),
        chunks_db=str(tmp_path / "chunks.sqlite"),
        graph_db=str(tmp_path / "graph.sqlite"),
        chroma_chunks=str(tmp_path / "chroma_chunks"),
        chroma_entities=str(tmp_path / "chroma_entities"),
        chroma_relations=str(tmp_path / "chroma_relations"),
    )

    rag = PathRAG(storage_paths=paths, system_prompt="You are helpful.")

    # Replace the async LLM call with a deterministic stub.
    async def _fake_generate(prompt, *, temperature=None, max_tokens=None):
        return "LLM-generated answer"

    rag._chat.generate = _fake_generate

    result = asyncio.run(rag.aretrieve("What is machine learning?"))

    assert result.answer == "LLM-generated answer"

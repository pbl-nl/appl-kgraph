import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from graph.pathrag import StorageAdapter


class _FakeStorage:
    def __init__(self, responses):
        self._responses = responses

    def init(self):
        return None

    def search_entities(self, *, text, n_results):
        return self._responses["entities"][:n_results]

    def search_relations(self, *, text, n_results):
        return self._responses["relations"][:n_results]

    def search_chunks(self, *, text, n_results):
        return self._responses["chunks"][:n_results]


@pytest.fixture
def fake_storage(monkeypatch):
    responses = {
        "entities": [
            {
                "name": "Entity One",
                "type": "person",
                "description": "Desc 1",
                "score": 0.8,
            },
            {
                "name": "Entity Two",
                "type": "place",
                "description": "Desc 2",
                "score": 0.4,
            },
        ],
        "relations": [
            {
                "source_name": "Entity One",
                "target_name": "Entity Two",
                "description": "Relates",
                "keywords": "friend",
                "score": 0.9,
            },
            {
                "source_name": "Entity Two",
                "target_name": "Entity Three",
                "description": "Connects",
                "keywords": "ally",
                "score": 0.6,
            },
        ],
        "chunks": [
            {
                "chunk_uuid": "chunk-1",
                "doc_id": "doc-1",
                "filename": "file-1.txt",
                "text": "text one",
                "score": 0.7,
            },
            {
                "chunk_uuid": "chunk-2",
                "doc_id": "doc-2",
                "filename": "file-2.txt",
                "text": "text two",
                "score": 0.5,
            },
        ],
    }
    storage = _FakeStorage(responses)
    monkeypatch.setattr("graph.pathrag.Storage", lambda *args, **kwargs: storage)
    return storage


def test_storage_adapter_query_helpers_expand_matches(fake_storage):
    adapter = StorageAdapter()

    entity_matches = adapter.query_entities("query", limit=5)
    assert [match.name for match in entity_matches] == ["Entity One", "Entity Two"]
    assert all(match.score > 0 for match in entity_matches)

    relation_matches = adapter.query_relations("query", limit=5)
    assert [match.source_name for match in relation_matches] == ["Entity One", "Entity Two"]
    assert all(match.score > 0 for match in relation_matches)

    chunk_matches = adapter.query_chunks("query", limit=5)
    assert [match.chunk_uuid for match in chunk_matches] == ["chunk-1", "chunk-2"]
    assert [match.text for match in chunk_matches] == ["text one", "text two"]
    assert all(match.score > 0 for match in chunk_matches)

import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from graph.pathrag import StorageAdapter


class _FakeVector:
    def __init__(self, response):
        self._response = response

    def query(self, *args, **kwargs):
        return self._response


class _FakeStorage:
    def __init__(self, responses):
        self.entity_vectors = _FakeVector(responses["entities"])
        self.relation_vectors = _FakeVector(responses["relations"])
        self.chunk_vectors = _FakeVector(responses["chunks"])

    def init(self):
        return None


@pytest.fixture
def fake_storage(monkeypatch):
    responses = {
        "entities": [
            {
                "ids": ["entity-1", "entity-2"],
                "metadatas": [
                    {"name": "Entity One", "type": "person", "description": "Desc 1"},
                    {"name": "Entity Two", "type": "place", "description": "Desc 2"},
                ],
                "distances": [0.2, 0.6],
            }
        ],
        "relations": [
            {
                "metadatas": [
                    {
                        "source_name": "Entity One",
                        "target_name": "Entity Two",
                        "description": "Relates",
                        "keywords": "friend",
                    },
                    {
                        "source_name": "Entity Two",
                        "target_name": "Entity Three",
                        "description": "Connects",
                        "keywords": "ally",
                    },
                ],
                "distances": [0.1, 0.4],
            }
        ],
        "chunks": [
            {
                "ids": ["chunk-1", "chunk-2"],
                "metadatas": [
                    {"doc_id": "doc-1", "filename": "file-1.txt"},
                    {"doc_id": "doc-2", "filename": "file-2.txt"},
                ],
                "documents": ["text one", "text two"],
                "distances": [0.3, 0.5],
            }
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

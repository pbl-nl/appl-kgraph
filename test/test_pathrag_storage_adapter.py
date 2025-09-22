import sys
import types

import pytest


_storage_module = types.ModuleType("storage")


class _ImportStubStorage:
    def __init__(self, *args, **kwargs):
        pass

    def init(self):  # pragma: no cover - simple stub
        return None


class _ImportStubPaths:
    pass


_storage_module.Storage = _ImportStubStorage
_storage_module.StoragePaths = _ImportStubPaths
sys.modules.setdefault("storage", _storage_module)

_llm_module = types.ModuleType("llm")


class _ImportStubChat:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):  # pragma: no cover - simple stub
        return ""


_llm_module.Chat = _ImportStubChat
sys.modules.setdefault("llm", _llm_module)

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

    def init(self):  # pragma: no cover - simple stub
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
    monkeypatch.setattr("graph.pathrag.IngestionStorage", lambda *args, **kwargs: storage)
    return storage


def test_storage_adapter_query_helpers_expand_matches(fake_storage):
    adapter = StorageAdapter()

    entity_matches = adapter.query_entities("query", limit=5)
    assert [m.name for m in entity_matches] == ["Entity One", "Entity Two"]
    assert all(m.score > 0 for m in entity_matches)

    relation_matches = adapter.query_relations("query", limit=5)
    assert [m.source_name for m in relation_matches] == ["Entity One", "Entity Two"]
    assert all(m.score > 0 for m in relation_matches)

    chunk_matches = adapter.query_chunks("query", limit=5)
    assert [m.chunk_uuid for m in chunk_matches] == ["chunk-1", "chunk-2"]
    assert [m.text for m in chunk_matches] == ["text one", "text two"]
    assert all(m.score > 0 for m in chunk_matches)

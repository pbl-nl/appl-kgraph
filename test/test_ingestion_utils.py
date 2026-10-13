import os
import sys
import types
import json
import pytest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.modules.setdefault("fitz", types.ModuleType("fitz"))

langdetect_stub = types.ModuleType("langdetect")


class _DummyLangDetectException(Exception):
    pass


langdetect_stub.detect = lambda text: "en"
langdetect_stub.LangDetectException = _DummyLangDetectException
sys.modules.setdefault("langdetect", langdetect_stub)

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

import ingestion
from ingestion import (
    _resolve_type,
    _write_extraction_diagnostics,
    _write_raw_document_snapshot,
    dedupe_entities_for_vectors,
)
from project_paths import resolve_project_paths


def test_resolve_type_prefers_non_unknown():
    votes = Counter({"unknown": 1, "Person": 1})
    result = _resolve_type(votes, "unknown")
    assert result == "Person"


def test_resolve_type_uses_majority_ignoring_unknown():
    votes = Counter({"Person": 2, "Organization": 1, "unknown": 5})
    result = _resolve_type(votes, "Organization")
    assert result == "Person"


def test_resolve_type_prefers_existing_on_tie():
    votes = Counter({"Person": 1, "Organization": 1})
    result = _resolve_type(votes, "Organization")
    assert result == "Organization"


def test_dedupe_entities_prefers_typed_over_unknown():
    entities = [
        {"name": "Alice", "type": "unknown", "description": "from edge"},
        {"name": "Alice", "type": "Person", "description": "from extraction"},
        {"name": "Bob", "type": "Company"},
        {"name": "Bob", "type": "unknown"},
    ]

    deduped = dedupe_entities_for_vectors(entities)

    assert len(deduped) == 2
    names_to_types = {e["name"]: e.get("type") for e in deduped}
    assert names_to_types["Alice"] == "Person"
    assert names_to_types["Bob"] == "Company"


def test_write_raw_document_snapshot_writes_canonical_storage_payload(tmp_path):
    documents_root = tmp_path / "docs"
    documents_root.mkdir()
    project_paths = resolve_project_paths(documents_root)

    doc_meta = {
        "doc_id": "doc-123",
        "filepath": str((documents_root / "example.txt").resolve()),
        "content_hash": "abc123",
        "language": "en",
    }

    _write_raw_document_snapshot(
        project_paths,
        filename="example.txt",
        doc_meta=doc_meta,
        raw_text="",
    )

    target = project_paths.raw_documents_dir / "example.txt.raw.json"
    assert target.exists()

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["filename"] == "example.txt"
    assert payload["doc_id"] == "doc-123"
    assert payload["filepath"] == doc_meta["filepath"]
    assert payload["content_hash"] == "abc123"
    assert payload["language"] == "en"
    assert payload["char_count"] == 0
    assert payload["raw_text"] == ""
    assert payload["extracted_at"].endswith("+00:00")


def test_extraction_diagnostics_respect_internal_logging_gate(tmp_path, monkeypatch):
    documents_root = tmp_path / "docs"
    documents_root.mkdir()
    project_paths = resolve_project_paths(documents_root)
    validation_results = [{"chunk_uuid": "chunk-1", "summary": "missing nothing"}]

    monkeypatch.setattr(
        ingestion,
        "settings",
        SimpleNamespace(logging=SimpleNamespace(internal_logging_enabled=False)),
    )

    _write_extraction_diagnostics(project_paths, "report.md", validation_results)

    assert not any(project_paths.extraction_diagnostics_dir.glob("*.validation.json"))

    monkeypatch.setattr(
        ingestion,
        "settings",
        SimpleNamespace(logging=SimpleNamespace(internal_logging_enabled=True)),
    )

    _write_extraction_diagnostics(project_paths, "report.md", validation_results)

    target = project_paths.extraction_diagnostics_dir / "report.validation.json"
    assert target.exists()


class _EmptyStorage:
    def __init__(self, *args, **kwargs):
        pass

    def init(self):
        return None

    def get_all_documents(self):
        return []


def test_ingestion_progress_callback_respects_verbosity_gate(monkeypatch):
    messages = []
    monkeypatch.setattr(ingestion, "Storage", _EmptyStorage)
    monkeypatch.setattr(
        ingestion,
        "settings",
        SimpleNamespace(
            logging=SimpleNamespace(
                internal_logging_enabled=False,
                internal_log_level="INFO",
                verbosity_enabled=False,
            )
        ),
    )

    ingestion.ingest_paths([], progress_callback=messages.append)

    assert messages == []


def test_ingestion_resumes_without_reprocessing_completed_files(monkeypatch, tmp_path):
    doc1 = tmp_path / "a.txt"
    doc2 = tmp_path / "b.txt"
    doc1.write_text("alpha", encoding="utf-8")
    doc2.write_text("beta", encoding="utf-8")

    parse_calls = []
    extract_calls = []

    class _FakeStorage:
        persisted_docs = {}
        persisted_chunks = {}
        persisted_chunk_vectors = set()
        _vector_mutations_disabled = False

        def __init__(self, *args, **kwargs):
            pass

        def init(self):
            return None

        def get_all_documents(self):
            return [dict(doc) for doc in self.persisted_docs.values()]

        def get_document_by_filename(self, filename):
            doc = self.persisted_docs.get(filename)
            if doc is None:
                return None
            return dict(doc)

        def add_document(self, metadata, full_text):
            self.persisted_docs[metadata["filename"]] = {
                "filename": metadata["filename"],
                "content_hash": metadata["content_hash"],
                "doc_id": metadata["doc_id"],
            }

        def upsert_document(self, doc_id, updates):
            for filename, doc in self.persisted_docs.items():
                if doc.get("doc_id") == doc_id:
                    doc.update(updates)
                    self.persisted_docs[filename] = doc

        def add_chunks(self, chunks):
            if not chunks:
                return None
            doc_id = chunks[0]["doc_id"]
            self.persisted_chunks[doc_id] = [dict(c) for c in chunks]
            return None

        def upsert_chunk_vector(self, chunks):
            for c in chunks:
                self.persisted_chunk_vectors.add(c["chunk_uuid"])
            return None

        def get_chunks_by_doc_id(self, doc_id):
            return [dict(c) for c in self.persisted_chunks.get(doc_id, [])]

        def get_chunk_vector(self, chunk_uuid):
            return [{"ids": [chunk_uuid]}] if chunk_uuid in self.persisted_chunk_vectors else []

        def delete_document(self, doc_id):
            for filename, doc in list(self.persisted_docs.items()):
                if doc.get("doc_id") == doc_id:
                    del self.persisted_docs[filename]
            self.persisted_chunks.pop(doc_id, None)

        def upsert_nodes(self, nodes):
            return None

        def upsert_edges(self, edges):
            return None

        def upsert_entity_vector(self, entities):
            return None

        def upsert_relation_vector(self, relations):
            return None

        def get_nodes(self, names):
            return []

        def get_edges(self, pairs):
            return []

    def _fake_parse(path):
        parse_calls.append(Path(path).name)
        return [(1, "text")], {"language": "en", "mime_type": "text/plain"}

    def _fake_build_chunks(pages, doc_id, filename, **kwargs):
        return [{
            "chunk_uuid": f"chunk-{filename}",
            "doc_id": doc_id,
            "chunk_id": 0,
            "filename": filename,
            "filepath": kwargs.get("filepath"),
            "document_language": kwargs.get("document_language"),
            "text": "text",
            "char_count": 4,
            "start_page": 1,
            "end_page": 1,
        }]

    def _fake_extract(chunks, storage):
        filename = chunks[0]["filename"]
        extract_calls.append(filename)
        return {
            "entities": [{
                "name": f"Entity-{filename}",
                "type": "Thing",
                "description": "d",
                "source_id": chunks[0]["chunk_uuid"],
                "filepath": chunks[0].get("filepath", ""),
            }],
            "relationships": [{
                "source_name": f"Entity-{filename}",
                "target_name": "Shared",
                "description": "r",
                "keywords": "k",
                "weight": 1.0,
                "source_id": chunks[0]["chunk_uuid"],
                "filepath": chunks[0].get("filepath", ""),
            }],
            "validation_results": [],
        }

    def _fake_merge(storage, entities, relations):
        return entities, relations

    def _fake_ensure_endpoints(storage, edges):
        return []

    class _StopAfterSecondParse(Exception):
        pass

    state = {"parse_count": 0}

    def _crashing_parse(path):
        state["parse_count"] += 1
        if state["parse_count"] == 2:
            raise _StopAfterSecondParse("simulated interruption")
        return _fake_parse(path)

    monkeypatch.setattr(ingestion, "Storage", _FakeStorage)
    monkeypatch.setattr(ingestion, "parse_to_pages", _crashing_parse)
    monkeypatch.setattr(ingestion, "build_chunks", _fake_build_chunks)
    monkeypatch.setattr(ingestion, "extract_from_chunks", _fake_extract)
    monkeypatch.setattr(ingestion, "merge_graph_data", _fake_merge)
    monkeypatch.setattr(ingestion, "ensure_edge_endpoints", _fake_ensure_endpoints)
    monkeypatch.setattr(ingestion, "_write_retrieval_graph_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingestion, "_write_raw_document_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingestion, "_write_extraction_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ingestion,
        "settings",
        SimpleNamespace(
            logging=SimpleNamespace(
                internal_logging_enabled=False,
                internal_log_level="INFO",
                verbosity_enabled=False,
            ),
            ingestion=SimpleNamespace(
                delimiter="||",
                description_segment_limit=999,
            ),
        ),
    )

    with pytest.raises(_StopAfterSecondParse):
        ingestion.ingest_paths([doc1, doc2])

    # Second run should skip doc1 (already persisted) and only process doc2.
    monkeypatch.setattr(ingestion, "parse_to_pages", _fake_parse)
    ingest_result = ingestion.ingest_paths([doc1, doc2])

    assert parse_calls == ["a.txt", "b.txt"]
    assert extract_calls == ["a.txt", "b.txt"]
    assert ingest_result["processed_files"] == 1
    assert ingest_result["skipped_files"] == 1


def test_ingestion_does_not_skip_file_after_crash_post_chunk_vectors(monkeypatch, tmp_path):
    doc1 = tmp_path / "only.txt"
    doc1.write_text("alpha", encoding="utf-8")

    parse_calls = []

    class _FakeStorage:
        persisted_docs = {}
        persisted_chunks = {}
        _vector_mutations_disabled = False
        first_run = True

        def __init__(self, *args, **kwargs):
            self.graphdb = SimpleNamespace(
                list_nodes=lambda: [],
                list_edges=lambda: [],
            )

        def init(self):
            return None

        def get_all_documents(self):
            return [dict(doc) for doc in self.persisted_docs.values()]

        def get_document_by_filename(self, filename):
            doc = self.persisted_docs.get(filename)
            return dict(doc) if doc else None

        def add_document(self, metadata, full_text):
            self.persisted_docs[metadata["filename"]] = {
                "filename": metadata["filename"],
                "content_hash": metadata["content_hash"],
                "doc_id": metadata["doc_id"],
            }

        def upsert_document(self, doc_id, updates):
            for filename, doc in self.persisted_docs.items():
                if doc.get("doc_id") == doc_id:
                    doc.update(updates)
                    self.persisted_docs[filename] = doc

        def add_chunks(self, chunks):
            if not chunks:
                return None
            self.persisted_chunks[chunks[0]["doc_id"]] = [dict(c) for c in chunks]
            return None

        def get_chunks_by_doc_id(self, doc_id):
            return [dict(c) for c in self.persisted_chunks.get(doc_id, [])]

        def get_chunk_vector(self, chunk_uuid):
            # Crash scenario test relies on pending content_hash marker, not vector checks.
            return []

        def upsert_chunk_vector(self, chunks):
            if self.first_run:
                raise _StopAfterChunkVectors("simulated crash right after chunk vectors")
            return None

        def upsert_nodes(self, nodes):
            return None

        def upsert_edges(self, edges):
            return None

        def upsert_entity_vector(self, entities):
            return None

        def upsert_relation_vector(self, relations):
            return None

        def get_nodes(self, names):
            return []

        def get_edges(self, pairs):
            return []

        def delete_document(self, doc_id):
            for filename, doc in list(self.persisted_docs.items()):
                if doc.get("doc_id") == doc_id:
                    del self.persisted_docs[filename]
            self.persisted_chunks.pop(doc_id, None)

        def delete_chunk_vector(self, chunk_uuid):
            return None

        def delete_chunks_by_uuids(self, chunk_uuids):
            return None

        def get_nodes_by_chunk_uuids(self, chunk_uuids):
            return []

        def get_edges_by_chunk_uuids(self, chunk_uuids):
            return []

        def delete_nodes(self, names):
            return None

        def delete_edges(self, pairs):
            return None

        def delete_entity_vector(self, names):
            return None

        def delete_relation_vector(self, pairs):
            return None

    def _fake_parse(path):
        parse_calls.append(Path(path).name)
        return [(1, "text")], {"language": "en", "mime_type": "text/plain"}

    def _fake_build_chunks(pages, doc_id, filename, **kwargs):
        return [{
            "chunk_uuid": f"chunk-{filename}",
            "doc_id": doc_id,
            "chunk_id": 0,
            "filename": filename,
            "filepath": kwargs.get("filepath"),
            "document_language": kwargs.get("document_language"),
            "text": "text",
            "char_count": 4,
            "start_page": 1,
            "end_page": 1,
        }]

    def _fake_extract(chunks, storage):
        return {
            "entities": [{
                "name": "Entity-only",
                "type": "Thing",
                "description": "d",
                "source_id": chunks[0]["chunk_uuid"],
                "filepath": chunks[0].get("filepath", ""),
            }],
            "relationships": [{
                "source_name": "Entity-only",
                "target_name": "Shared",
                "description": "r",
                "keywords": "k",
                "weight": 1.0,
                "source_id": chunks[0]["chunk_uuid"],
                "filepath": chunks[0].get("filepath", ""),
            }],
            "validation_results": [{"ok": True}],
        }

    def _fake_merge(storage, entities, relations):
        return entities, relations

    def _fake_ensure_endpoints(storage, edges):
        return []

    class _StopAfterChunkVectors(Exception):
        pass

    monkeypatch.setattr(ingestion, "Storage", _FakeStorage)
    monkeypatch.setattr(ingestion, "parse_to_pages", _fake_parse)
    monkeypatch.setattr(ingestion, "build_chunks", _fake_build_chunks)
    monkeypatch.setattr(ingestion, "extract_from_chunks", _fake_extract)
    monkeypatch.setattr(ingestion, "merge_graph_data", _fake_merge)
    monkeypatch.setattr(ingestion, "ensure_edge_endpoints", _fake_ensure_endpoints)
    monkeypatch.setattr(ingestion, "_write_retrieval_graph_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingestion, "_write_raw_document_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingestion, "_write_extraction_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ingestion,
        "settings",
        SimpleNamespace(
            logging=SimpleNamespace(
                internal_logging_enabled=False,
                internal_log_level="INFO",
                verbosity_enabled=False,
            ),
            ingestion=SimpleNamespace(
                delimiter="||",
                description_segment_limit=999,
            ),
        ),
    )

    fake_storage = _FakeStorage()

    monkeypatch.setattr(ingestion, "Storage", lambda *args, **kwargs: fake_storage)

    with pytest.raises(_StopAfterChunkVectors):
        ingestion.ingest_paths([doc1])

    # The first run should leave a pending content hash marker in the document row.
    pending_doc = fake_storage.get_document_by_filename("only.txt")
    assert pending_doc is not None
    assert pending_doc["content_hash"].startswith(ingestion.PENDING_CONTENT_HASH_PREFIX)

    # Second run should reprocess (not skip), then finalize content hash.
    fake_storage.first_run = False
    result = ingestion.ingest_paths([doc1])

    assert parse_calls == ["only.txt", "only.txt"]
    finalized_doc = fake_storage.get_document_by_filename("only.txt")
    assert finalized_doc is not None
    assert not finalized_doc["content_hash"].startswith(ingestion.PENDING_CONTENT_HASH_PREFIX)
    assert result["processed_files"] == 1
    assert result["skipped_files"] == 0

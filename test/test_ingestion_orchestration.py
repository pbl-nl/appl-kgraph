import os
import sys
from pathlib import Path
from types import SimpleNamespace


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")
sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

import ingestion


class _StorageSpy:
    def __init__(self, existing_documents=()):
        self.existing_documents = list(existing_documents)
        self.calls = []

    def init(self):
        self.calls.append(("init",))

    def get_all_documents(self):
        self.calls.append(("get_all_documents",))
        return self.existing_documents

    def get_document_by_filename(self, filename):
        self.calls.append(("get_document_by_filename", filename))
        return None

    def add_document(self, metadata, full_text):
        self.calls.append(("add_document", metadata, full_text))

    def add_chunks(self, chunks):
        self.calls.append(("add_chunks", chunks))

    def upsert_nodes(self, nodes):
        self.calls.append(("upsert_nodes", nodes))

    def upsert_edges(self, edges):
        self.calls.append(("upsert_edges", edges))

    def upsert_chunk_vector(self, chunks):
        self.calls.append(("upsert_chunk_vector", chunks))

    def upsert_entity_vector(self, entities):
        self.calls.append(("upsert_entity_vector", entities))

    def upsert_relation_vector(self, relations):
        self.calls.append(("upsert_relation_vector", relations))


def _configure_orchestration_fakes(monkeypatch, storage):
    monkeypatch.setattr(ingestion, "Storage", lambda paths=None: storage)
    monkeypatch.setattr(ingestion, "_configure_ingestion_logger", lambda paths: None)
    monkeypatch.setattr(
        ingestion,
        "settings",
        SimpleNamespace(
            logging=SimpleNamespace(
                verbosity_enabled=True,
                internal_logging_enabled=False,
            )
        ),
    )


def test_ingest_paths_orchestrates_writes_and_returns_summary(tmp_path, monkeypatch):
    document = tmp_path / "report.txt"
    document.write_text("source text", encoding="utf-8")
    storage = _StorageSpy()
    progress = []
    chunks = [
        {
            "chunk_uuid": "chunk-1",
            "doc_id": "doc-1",
            "filename": document.name,
            "text": "First",
        }
    ]
    placeholder = {"name": "Missing endpoint", "type": "unknown"}
    node = {"name": "Named entity", "type": "Organization"}
    edge = {
        "source_name": "Named entity",
        "target_name": "Missing endpoint",
        "description": "references",
    }
    extraction_calls = []
    chunk_calls = []

    _configure_orchestration_fakes(monkeypatch, storage)
    monkeypatch.setattr(ingestion, "file_sha256", lambda path: "content-hash")
    monkeypatch.setattr(ingestion, "should_skip_ingestion", lambda *args: False)
    monkeypatch.setattr(
        ingestion,
        "parse_to_pages",
        lambda path: ([(1, "First"), (2, "Second")], {"language": "en", "mime_type": "text/plain"}),
    )

    def fake_build_chunks(pages, doc_id, filename, **metadata):
        chunk_calls.append((pages, doc_id, filename, metadata))
        return chunks

    def fake_extract(chunks_to_extract, *, storage, validation_enabled):
        extraction_calls.append((chunks_to_extract, storage, validation_enabled))
        return {
            "entities": [node],
            "relationships": [edge],
            "validation_results": [],
        }

    monkeypatch.setattr(ingestion, "build_chunks", fake_build_chunks)
    monkeypatch.setattr(ingestion, "extract_from_chunks", fake_extract)
    monkeypatch.setattr(ingestion, "ensure_edge_endpoints", lambda storage, edges: [placeholder])
    monkeypatch.setattr(ingestion, "merge_graph_data", lambda storage, entities, edges: ([node], [edge]))
    monkeypatch.setattr(ingestion.uuid, "uuid4", lambda: "doc-1")
    monkeypatch.setattr(
        ingestion,
        "_write_retrieval_graph_snapshot",
        lambda storage, project_paths: Path("retrieval.pkl"),
    )

    summary = ingestion.ingest_paths(
        [document],
        storage_paths="storage-paths",
        audit_enabled=True,
        progress_callback=progress.append,
    )

    assert summary == {
        "documents_root": None,
        "project_root": None,
        "retrieval_graph_pickle": "retrieval.pkl",
        "processed_files": 1,
        "skipped_files": 0,
        "removed_files": 0,
        "chunk_count": 1,
        "entity_count": 2,
        "relation_count": 1,
    }
    assert [call[0] for call in storage.calls] == [
        "init",
        "get_all_documents",
        "get_document_by_filename",
        "add_document",
        "add_chunks",
        "upsert_nodes",
        "upsert_edges",
        "upsert_chunk_vector",
        "upsert_entity_vector",
        "upsert_relation_vector",
    ]
    stored_metadata = storage.calls[3][1]
    assert stored_metadata["doc_id"] == "doc-1"
    assert stored_metadata["filename"] == document.name
    assert stored_metadata["content_hash"] == "content-hash"
    assert stored_metadata["language"] == "en"
    assert storage.calls[3][2] == "First\nSecond"
    assert chunk_calls == [
        (
            [(1, "First"), (2, "Second")],
            "doc-1",
            document.name,
            {
                "filepath": str(document.resolve()),
                "document_language": "en",
            },
        )
    ]
    assert extraction_calls == [(chunks, storage, True)]
    assert progress[0] == "Initializing project storage"
    assert progress[-1] == "Completed ingestion"


def test_ingest_paths_counts_skips_and_removes_stale_documents(tmp_path, monkeypatch):
    missing = tmp_path / "missing.txt"
    unchanged = tmp_path / "unchanged.txt"
    temporary = tmp_path / "~$draft.docx"
    broken = tmp_path / "broken.txt"
    for path in (unchanged, temporary, broken):
        path.write_text("content", encoding="utf-8")

    storage = _StorageSpy(existing_documents=[{"filename": "stale.txt"}])
    removed = []
    parsed = []
    progress = []

    _configure_orchestration_fakes(monkeypatch, storage)
    monkeypatch.setattr(ingestion, "file_sha256", lambda path: f"hash-{path.name}")
    monkeypatch.setattr(
        ingestion,
        "should_skip_ingestion",
        lambda storage, path, content_hash: path == unchanged,
    )
    monkeypatch.setattr(
        ingestion,
        "remove_document_from_storage",
        lambda storage, filename: removed.append((storage, filename)),
    )

    def fake_parse(path):
        parsed.append(path)
        return None, {}

    monkeypatch.setattr(ingestion, "parse_to_pages", fake_parse)
    monkeypatch.setattr(
        ingestion,
        "_write_retrieval_graph_snapshot",
        lambda storage, project_paths: None,
    )

    summary = ingestion.ingest_paths(
        [missing, unchanged, temporary, broken],
        storage_paths="storage-paths",
        progress_callback=progress.append,
    )

    assert summary["processed_files"] == 0
    assert summary["skipped_files"] == 4
    assert summary["removed_files"] == 1
    assert summary["chunk_count"] == 0
    assert summary["entity_count"] == 0
    assert summary["relation_count"] == 0
    assert removed == [(storage, "stale.txt")]
    assert parsed == [broken]
    assert "1/4 missing.txt - skipped (path missing or not a file)" in progress
    assert "2/4 unchanged.txt - skipped (unchanged)" in progress
    assert "3/4 ~$draft.docx - skipped (temporary file)" in progress
    assert "4/4 broken.txt - parsing failed" in progress
    assert not any(call[0].startswith("upsert_") for call in storage.calls)

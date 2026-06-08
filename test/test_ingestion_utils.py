import os
import sys
import types
import json
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

"""Public ingestion regressions using real parsing, canonical storage and graph adapters."""

import json
from dataclasses import replace

import pytest

from test.production_fixtures import configure_ingestion, DeterministicEmbedder
import ingestion
from db_storage import Storage
from ingestion_pipeline import IngestionStageError
from project_paths import resolve_project_paths

pytestmark = pytest.mark.storage_integration


def test_ingest_paths_uses_all_typed_stages_and_canonical_snapshots(
    tmp_path, monkeypatch
):
    document = tmp_path / "report.txt"
    document.write_text("Alpha works with Beta.", encoding="utf-8")
    extractor, constructed = configure_ingestion(monkeypatch)
    progress = []
    result = ingestion.ingest_paths(
        [document], documents_root=tmp_path, progress_callback=progress.append
    )
    assert (
        result["processed_files"],
        result["chunk_count"],
        result["entity_count"],
        result["relation_count"],
    ) == (1, 1, 2, 1)
    assert len(extractor.calls) == 1
    for stage in (
        "parse",
        "enrich",
        "chunk",
        "persist_document",
        "persist_chunks",
        "extract",
        "normalize",
        "transform_graph",
        "persist_graph",
        "index_chunks",
        "index_graph",
        "diagnostics",
        "finalize",
    ):
        assert any(message.endswith(" - " + stage) for message in progress), stage
    paths = resolve_project_paths(tmp_path)
    raw = json.loads((paths.raw_documents_dir / "report.txt.raw.json").read_text())
    enriched = json.loads(
        (paths.raw_documents_dir / "report.txt.enriched.json").read_text()
    )
    assert raw["text"] == enriched["text"] == "Alpha works with Beta."
    assert enriched["transformations"] == ["normalize_metadata"]
    assert paths.retrieval_graph_pickle_file.exists()
    check = Storage(paths.storage, DeterministicEmbedder())
    try:
        check.init()
        doc = check.get_document_by_filename("report.txt")
        assert doc["content_hash"] == ingestion.file_sha256(document)
        nodes = check.get_nodes(["Alpha", "Beta"])
        assert all(
            node["source_id"] for node in nodes
        )  # placeholder has real provenance
    finally:
        check.close()


def test_skip_partial_selection_replacement_and_removal(tmp_path, monkeypatch):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("Alpha one.")
    b.write_text("Alpha two.")
    extractor, _ = configure_ingestion(monkeypatch)
    assert (
        ingestion.ingest_paths([a, b], documents_root=tmp_path)["processed_files"] == 2
    )
    assert (
        ingestion.ingest_paths(
            [a], documents_root=tmp_path, prune_missing_documents=False
        )["skipped_files"]
        == 1
    )
    assert len(extractor.calls) == 2
    a.write_text("Alpha revised.")
    changed = ingestion.ingest_paths(
        [a], documents_root=tmp_path, prune_missing_documents=False
    )
    assert changed["processed_files"] == 1 and changed["removed_files"] == 0
    pruned = ingestion.ingest_paths([a], documents_root=tmp_path)
    assert pruned["removed_files"] == 1
    assert ingestion.ingest_paths([], documents_root=tmp_path)["removed_files"] == 1
    paths = resolve_project_paths(tmp_path)
    check = Storage(paths.storage, DeterministicEmbedder())
    try:
        check.init()
        assert check.get_all_documents() == []
        assert check.graphdb.list_nodes() == []
        assert check.graphdb.list_edges() == []
    finally:
        check.close()


@pytest.mark.parametrize(
    "failed_stage",
    [
        "extract",
        "persist_graph",
        "index_chunks",
        "index_graph",
        "diagnostics",
        "finalize",
    ],
)
def test_pending_marker_and_retry_preserve_completed_files(
    tmp_path, monkeypatch, failed_stage
):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("Alpha one.")
    b.write_text("Alpha two.")
    extractor, _ = configure_ingestion(monkeypatch)
    factory = ingestion.create_ingestion_pipeline

    def broken(storage, paths, **kwargs):
        pipeline = factory(storage, paths, **kwargs)
        original = getattr(pipeline, failed_stage)
        count = 0

        def fail_second(*args):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("injected failure")
            return original(*args)

        return replace(pipeline, **{failed_stage: fail_second})

    monkeypatch.setattr(ingestion, "create_ingestion_pipeline", broken)
    with pytest.raises(IngestionStageError, match=failed_stage):
        ingestion.ingest_paths([a, b], documents_root=tmp_path)
    paths = resolve_project_paths(tmp_path)
    check = Storage(paths.storage, DeterministicEmbedder())
    try:
        check.init()
        assert check.get_document_by_filename("a.txt")[
            "content_hash"
        ] == ingestion.file_sha256(a)
        assert check.get_document_by_filename("b.txt")["content_hash"].startswith(
            ingestion.PENDING_CONTENT_HASH_PREFIX
        )
    finally:
        check.close()
    monkeypatch.setattr(ingestion, "create_ingestion_pipeline", factory)
    summary = ingestion.ingest_paths([a, b], documents_root=tmp_path)
    assert summary["processed_files"] == 1 and summary["skipped_files"] == 1


def test_empty_invalid_and_duplicate_inputs(tmp_path, monkeypatch):
    configure_ingestion(monkeypatch)
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    temporary = tmp_path / "~$word.docx"
    temporary.write_text("invalid")
    broken = tmp_path / "broken.pdf"
    broken.write_text("invalid PDF")
    summary = ingestion.ingest_paths(
        [empty, temporary, broken, tmp_path / "missing.txt"], documents_root=tmp_path
    )
    assert summary["processed_files"] == 0 and summary["skipped_files"] == 4
    monkeypatch.setattr(
        ingestion,
        "Storage",
        lambda **kwargs: pytest.fail("duplicates must be rejected before writes"),
    )
    with pytest.raises(ValueError, match="Duplicate document paths"):
        ingestion.ingest_paths([empty, empty], documents_root=tmp_path)
    with pytest.raises(ValueError, match="Duplicate document filenames"):
        ingestion.ingest_paths(
            [empty, tmp_path / "nested" / "empty.txt"], documents_root=tmp_path
        )


def test_failed_replacement_cannot_leave_a_stale_retrieval_snapshot(
    tmp_path, monkeypatch
):
    configure_ingestion(monkeypatch)
    source = tmp_path / "report.txt"
    source.write_text("Alpha original.")
    ingestion.ingest_paths([source], documents_root=tmp_path)
    paths = resolve_project_paths(tmp_path)
    assert paths.retrieval_graph_pickle_file.exists()
    source.write_text("Alpha revised.")
    factory = ingestion.create_ingestion_pipeline

    def failed(*args, **kwargs):
        pipeline = factory(*args, **kwargs)

        def extract(_):
            raise OSError("model unavailable")

        return replace(pipeline, extract=extract)

    monkeypatch.setattr(ingestion, "create_ingestion_pipeline", failed)
    with pytest.raises(IngestionStageError, match="extract"):
        ingestion.ingest_paths([source], documents_root=tmp_path)
    assert not paths.retrieval_graph_pickle_file.exists()

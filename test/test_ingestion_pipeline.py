from pathlib import Path

import pytest

from ingestion_pipeline import (
    IngestionPipeline,
    IngestionStageError,
    ingest_documents,
)
from schemas import (
    Chunk,
    DocumentRef,
    EnrichedDocument,
    Entity,
    ExtractionResult,
    GraphDelta,
    IngestedDocument,
    IngestionSummary,
    RawDocument,
)


def _pipeline(document, *, chunk_doc_id="doc-1", fail_stage=None):
    calls = []
    raw = RawDocument(
        ref=document,
        doc_id="doc-1",
        pages=((0, "Alpha"),),
        text="Alpha",
    )
    enriched = EnrichedDocument(raw=raw, text="Alpha")
    chunks = (
        Chunk(
            chunk_uuid="chunk-1",
            doc_id=chunk_doc_id,
            chunk_id=0,
            filename=document.path.name,
            text="Alpha",
            char_count=5,
            start_page=0,
            end_page=0,
        ),
    )
    extraction = ExtractionResult(
        entities=[Entity(name="Alpha", type="Thing", description="Entity")]
    )
    delta = GraphDelta(entities=extraction.entities)

    def stage(name, output=None):
        def run(*args):
            calls.append((name, args))
            if fail_stage == name:
                raise OSError("stage failed")
            return output

        return run

    pipeline = IngestionPipeline(
        parse=stage("parse", raw),
        enrich=stage("enrich", enriched),
        chunk=stage("chunk", chunks),
        extract=stage("extract", extraction),
        normalize=stage("normalize", delta),
        persist_document=stage("persist_document"),
        persist_chunks=stage("persist_chunks"),
        persist_graph=stage("persist_graph"),
        index_chunks=stage("index_chunks"),
        index_graph=stage("index_graph"),
    )
    return pipeline, calls, raw, enriched, chunks, extraction, delta


def test_ingest_documents_composes_typed_stages_and_returns_summary():
    document = DocumentRef(path=Path("report.txt"))
    pipeline, calls, raw, enriched, chunks, extraction, delta = _pipeline(document)
    progress = []

    summary = ingest_documents(
        [document],
        pipeline,
        progress=lambda stage, ref: progress.append((stage, ref)),
    )

    assert summary == IngestionSummary(
        documents=(
            IngestedDocument(
                ref=document,
                doc_id="doc-1",
                chunk_count=1,
                entity_count=1,
                relation_count=0,
            ),
        )
    )
    assert summary.processed_files == 1
    assert summary.chunk_count == 1
    assert [name for name, _ in calls] == [
        "parse",
        "enrich",
        "persist_document",
        "chunk",
        "persist_chunks",
        "extract",
        "normalize",
        "persist_graph",
        "index_chunks",
        "index_graph",
    ]
    assert calls[2][1] == (raw, enriched)
    assert calls[4][1] == (chunks,)
    assert calls[5][1] == (chunks,)
    assert calls[6][1] == (extraction,)
    assert calls[7][1] == (delta,)
    assert calls[8][1] == (chunks,)
    assert calls[9][1] == (delta,)
    assert [stage for stage, _ in progress] == [name for name, _ in calls]


def test_ingest_documents_returns_empty_summary_for_empty_input():
    document = DocumentRef(path=Path("unused.txt"))
    pipeline, calls, *_ = _pipeline(document)

    assert ingest_documents([], pipeline) == IngestionSummary()
    assert calls == []


def test_ingest_documents_identifies_stage_and_document_on_failure():
    document = DocumentRef(path=Path("broken.txt"))
    pipeline, calls, *_ = _pipeline(document, fail_stage="extract")

    with pytest.raises(IngestionStageError, match="extract") as caught:
        ingest_documents([document], pipeline)

    assert caught.value.stage == "extract"
    assert caught.value.document is document
    assert isinstance(caught.value.__cause__, OSError)
    assert [name for name, _ in calls][-1] == "extract"


def test_ingest_documents_rejects_cross_document_chunks_before_writes():
    document = DocumentRef(path=Path("report.txt"))
    pipeline, calls, *_ = _pipeline(document, chunk_doc_id="other-doc")

    with pytest.raises(IngestionStageError, match="matching document identity"):
        ingest_documents([document], pipeline)

    assert [name for name, _ in calls][-1] == "chunk"


def test_ingest_documents_rejects_duplicate_source_paths():
    document = DocumentRef(path=Path("report.txt"))
    pipeline, calls, *_ = _pipeline(document)

    with pytest.raises(ValueError, match="duplicate paths"):
        ingest_documents([document, document], pipeline)

    assert calls == []

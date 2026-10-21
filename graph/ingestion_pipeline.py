from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from schemas import (
    Chunk,
    DocumentRef,
    EnrichedDocument,
    ExtractionResult,
    GraphDelta,
    IngestedDocument,
    IngestionSummary,
    RawDocument,
)


@dataclass(frozen=True)
class IngestionPipeline:
    """All typed stage connectors required to ingest supplied documents."""

    parse: Callable[[DocumentRef], RawDocument]
    enrich: Callable[[RawDocument], EnrichedDocument]
    chunk: Callable[[EnrichedDocument], Sequence[Chunk]]
    extract: Callable[[Sequence[Chunk]], ExtractionResult]
    normalize: Callable[[ExtractionResult], GraphDelta]
    persist_document: Callable[[RawDocument, EnrichedDocument], None]
    persist_chunks: Callable[[Sequence[Chunk]], None]
    persist_graph: Callable[[GraphDelta], None]
    index_chunks: Callable[[Sequence[Chunk]], None]
    index_graph: Callable[[GraphDelta], None]


class IngestionStageError(RuntimeError):
    """A pipeline failure identified by stage and source document."""

    def __init__(self, stage: str, document: DocumentRef, cause: Exception) -> None:
        self.stage = stage
        self.document = document
        super().__init__(f"Ingestion failed during {stage} for {document.path}: {cause}")


def ingest_documents(
    documents: Sequence[DocumentRef],
    pipeline: IngestionPipeline,
    *,
    progress: Callable[[str, DocumentRef], None] | None = None,
) -> IngestionSummary:
    """Run supplied documents through each typed ingestion connector."""

    document_batch = _document_batch(documents)
    if not isinstance(pipeline, IngestionPipeline):
        raise TypeError("pipeline must be an IngestionPipeline")
    completed = []

    for document in document_batch:
        raw = _run_stage(
            "parse",
            document,
            pipeline.parse,
            document,
            progress=progress,
        )
        if not isinstance(raw, RawDocument):
            _invalid_output("parse", document, "RawDocument")
        if raw.ref.path != document.path:
            _invalid_output("parse", document, "RawDocument for the requested path")

        enriched = _run_stage(
            "enrich",
            document,
            pipeline.enrich,
            raw,
            progress=progress,
        )
        if not isinstance(enriched, EnrichedDocument):
            _invalid_output("enrich", document, "EnrichedDocument")
        if enriched.raw.doc_id != raw.doc_id:
            _invalid_output("enrich", document, "matching document identity")

        _run_stage(
            "persist_document",
            document,
            pipeline.persist_document,
            raw,
            enriched,
            progress=progress,
        )
        chunks = _run_stage(
            "chunk",
            document,
            pipeline.chunk,
            enriched,
            progress=progress,
        )
        chunks = _validated_chunks(chunks, raw, document)
        _run_stage(
            "persist_chunks",
            document,
            pipeline.persist_chunks,
            chunks,
            progress=progress,
        )

        extraction = _run_stage(
            "extract",
            document,
            pipeline.extract,
            chunks,
            progress=progress,
        )
        if not isinstance(extraction, ExtractionResult):
            _invalid_output("extract", document, "ExtractionResult")
        delta = _run_stage(
            "normalize",
            document,
            pipeline.normalize,
            extraction,
            progress=progress,
        )
        if not isinstance(delta, GraphDelta):
            _invalid_output("normalize", document, "GraphDelta")

        _run_stage(
            "persist_graph",
            document,
            pipeline.persist_graph,
            delta,
            progress=progress,
        )
        _run_stage(
            "index_chunks",
            document,
            pipeline.index_chunks,
            chunks,
            progress=progress,
        )
        _run_stage(
            "index_graph",
            document,
            pipeline.index_graph,
            delta,
            progress=progress,
        )
        completed.append(
            IngestedDocument(
                ref=document,
                doc_id=raw.doc_id,
                chunk_count=len(chunks),
                entity_count=len(delta.entities),
                relation_count=len(delta.relations),
            )
        )

    return IngestionSummary(documents=tuple(completed))


def _run_stage(stage, document, function, *args, progress=None):
    if progress is not None:
        progress(stage, document)
    try:
        return function(*args)
    except IngestionStageError:
        raise
    except Exception as exc:
        raise IngestionStageError(stage, document, exc) from exc


def _invalid_output(stage, document, expected):
    cause = TypeError(f"stage must return {expected}")
    raise IngestionStageError(stage, document, cause) from cause


def _document_batch(documents):
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise TypeError("documents must be a sequence of DocumentRef objects")
    batch = tuple(documents)
    if not all(isinstance(document, DocumentRef) for document in batch):
        raise TypeError("documents must contain only DocumentRef objects")
    resolved_paths = [document.path.resolve() for document in batch]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("documents must not contain duplicate paths")
    return batch


def _validated_chunks(chunks, raw, document):
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        _invalid_output("chunk", document, "a sequence of Chunk objects")
    batch = tuple(chunks)
    if not all(isinstance(chunk, Chunk) for chunk in batch):
        _invalid_output("chunk", document, "only Chunk objects")
    if any(chunk.doc_id != raw.doc_id for chunk in batch):
        _invalid_output("chunk", document, "chunks with matching document identity")
    return batch

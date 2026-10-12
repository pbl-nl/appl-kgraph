"""Production adapters for the typed document pipeline.

SQLite/model dictionary serialization belongs here, outside transformation stages.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone

from chunker import ChunkingConfig, chunk_document
from document_parser import DocumentParseError, parse_document
from enrichment import MetadataNormalizer, enrich_document
from extraction import LegacyGraphExtractor, chunk_to_legacy, extract_graph
from extractor import extract_from_chunks
from fileparser import FileParser
from graph_normalization import GraphView, normalize_graph
from ingestion_pipeline import IngestionPipeline
from llm import llm_summarize_text
from schemas import (
    extraction_result_from_legacy,
    extraction_result_to_legacy,
    ExtractionResult,
)
from settings import settings
from storage_connectors import (
    persist_document,
    persist_chunks,
    persist_graph,
    index_chunks,
    index_graph,
)

PENDING_CONTENT_HASH_PREFIX = "__PENDING_INGEST__::"


class EmptyDocumentError(DocumentParseError):
    """A successfully parsed source contains no usable text."""


class FileDocumentParser:
    def __init__(self, parser=None):
        self.parser = parser if parser is not None else FileParser()

    def parse(self, path):
        pages, metadata = self.parser.parse_file(path)
        if not any(text.strip() for _, text in pages):
            raise EmptyDocumentError(path, "document contains no text")
        return pages, metadata


def graph_rows(delta):
    payload = extraction_result_to_legacy(
        ExtractionResult(entities=delta.entities, relations=delta.relations)
    )
    return payload["entities"], payload["relationships"]


class SQLiteDocumentStore:
    def __init__(self, storage, project_paths=None):
        self.storage = storage
        self.project_paths = project_paths

    def _snapshot(self, document, suffix):
        if self.project_paths is None:
            return
        raw = document.raw if suffix == "enriched" else document
        folder = self.project_paths.raw_documents_dir
        folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_id": raw.doc_id,
            "filename": raw.ref.path.name,
            "filepath": str(raw.ref.path.resolve()),
            "metadata": document.metadata,
            "pages": document.pages,
            "text": document.text,
            "char_count": len(document.text),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        if suffix == "enriched":
            payload["transformations"] = document.transformations
        (folder / f"{raw.ref.path.name}.{suffix}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def put_raw(self, document):
        self._snapshot(document, "raw")

    def put_enriched(self, document):
        raw = document.raw
        metadata = dict(document.metadata)
        metadata.update(
            doc_id=raw.doc_id,
            filename=raw.ref.path.name,
            filepath=str(raw.ref.path.resolve()),
            full_char_count=len(document.text),
            content_hash=PENDING_CONTENT_HASH_PREFIX + raw.ref.metadata["content_hash"],
        )
        self.storage.add_document(metadata, document.text)
        self._snapshot(document, "enriched")


class SQLiteChunkStore:
    def __init__(self, storage):
        self.storage = storage

    def put_many(self, chunks):
        self.storage.add_chunks([chunk_to_legacy(chunk) for chunk in chunks])


class SQLiteGraphStore:
    def __init__(self, storage):
        self.storage = storage

    def apply(self, delta):
        if delta.deleted_relation_pairs:
            self.storage.delete_edges(list(delta.deleted_relation_pairs))
        if delta.deleted_entity_names:
            self.storage.delete_nodes(list(delta.deleted_entity_names))
        nodes, edges = graph_rows(delta)
        if nodes:
            self.storage.upsert_nodes(nodes)
        if edges:
            self.storage.upsert_edges(edges)

    def view(self, extraction):
        names = {entity.name for entity in extraction.entities}
        names.update(
            name
            for edge in extraction.relations
            for name in (edge.source_name, edge.target_name)
        )
        pairs = sorted(
            {
                tuple(sorted((edge.source_name, edge.target_name)))
                for edge in extraction.relations
            }
        )
        result = extraction_result_from_legacy(
            {
                "entities": self.storage.get_nodes(sorted(names)) if names else [],
                "relationships": self.storage.get_edges(pairs) if pairs else [],
            }
        )
        return GraphView(result.entities, result.relations)


class StorageVectorIndex:
    def __init__(self, storage):
        self.storage = storage

    def index_chunks(self, chunks):
        if chunks:
            self.storage.upsert_chunk_vector(
                [chunk_to_legacy(chunk) for chunk in chunks]
            )

    def index_graph(self, delta):
        if delta.deleted_relation_pairs:
            self.storage.delete_relation_vector(list(delta.deleted_relation_pairs))
        if delta.deleted_entity_names:
            self.storage.delete_entity_vector(list(delta.deleted_entity_names))
        nodes, edges = graph_rows(delta)
        if nodes:
            self.storage.upsert_entity_vector(nodes)
        if edges:
            self.storage.upsert_relation_vector(edges)


def summarize_graph(delta, summarize=llm_summarize_text):
    """Optional model side effect, separate from pure graph normalization."""

    def update(item):
        if (
            len(item.description.split(settings.ingestion.delimiter))
            > settings.ingestion.description_segment_limit
        ):
            return replace(item, description=summarize(item.description))
        return item

    return replace(
        delta,
        entities=[update(item) for item in delta.entities],
        relations=[update(item) for item in delta.relations],
    )


def remove_document_from_storage(storage, filename: str) -> None:
    """Remove document evidence through the same graph store and index adapters."""
    from graph_normalization import GraphView, remove_graph_sources
    from schemas import extraction_result_from_legacy
    from storage_connectors import persist_graph, index_graph
    document = storage.get_document_by_filename(filename)
    if not document:
        return
    chunks = storage.get_chunks_by_doc_id(document["doc_id"])
    chunk_ids = [chunk["chunk_uuid"] for chunk in chunks]
    rows = extraction_result_from_legacy({"entities": storage.graphdb.list_nodes(),
                                          "relationships": storage.graphdb.list_edges()})
    delta = remove_graph_sources(GraphView(rows.entities, rows.relations), chunk_ids,
                                 filepath=document.get("filepath", ""), delimiter=settings.ingestion.delimiter)
    persist_graph(delta, SQLiteGraphStore(storage))
    index_graph(delta, StorageVectorIndex(storage))
    for chunk_id in chunk_ids:
        storage.delete_chunk_vector(chunk_id)
    storage.delete_chunks_by_doc_id(document["doc_id"])
    storage.delete_document(document["doc_id"])



def create_ingestion_pipeline(
    storage,
    project_paths=None,
    *,
    parser=None,
    extractor=None,
    enrichers=None,
    remove_existing=None,
    diagnostics=None,
    summarize=None,
):
    """Bind real adapters once; every stage remains independently replaceable."""
    parser = FileDocumentParser(parser)
    extractor = (
        extractor
        if extractor is not None
        else LegacyGraphExtractor(
            lambda chunks: extract_from_chunks(chunks, storage=storage)
        )
    )
    enrichers = [MetadataNormalizer()] if enrichers is None else enrichers
    documents = SQLiteDocumentStore(storage, project_paths)
    chunks = SQLiteChunkStore(storage)
    graph = SQLiteGraphStore(storage)
    vectors = StorageVectorIndex(storage)
    config = ChunkingConfig(**asdict(settings.chunking))

    def store_document(raw, enriched):
        if storage.get_document_by_filename(raw.ref.path.name):
            (remove_existing or remove_document_from_storage)(storage, raw.ref.path.name)
        persist_document(raw, enriched, documents)

    def finalize(raw):
        storage.upsert_document(
            raw.doc_id, {"content_hash": raw.ref.metadata["content_hash"]}
        )

    def parse_source(ref):
        if "content_hash" not in ref.metadata:
            with ref.path.open("rb") as handle:
                content_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            ref = replace(ref, metadata={**ref.metadata, "content_hash": content_hash})
        return parse_document(ref, parser)

    return IngestionPipeline(
        parse=parse_source,
        enrich=lambda raw: enrich_document(raw, enrichers),
        chunk=lambda document: chunk_document(document, config),
        extract=lambda batch: extract_graph(batch, extractor),
        normalize=lambda findings: normalize_graph(
            findings, graph.view(findings), delimiter=settings.ingestion.delimiter
        ),
        persist_document=store_document,
        persist_chunks=lambda batch: persist_chunks(batch, chunks),
        persist_graph=lambda delta: persist_graph(delta, graph),
        index_chunks=lambda batch: index_chunks(batch, vectors),
        index_graph=lambda delta: index_graph(delta, vectors),
        transform_graph=lambda delta: summarize_graph(
            delta, summarize or llm_summarize_text
        ),
        diagnostics=diagnostics,
        finalize=finalize,
    )

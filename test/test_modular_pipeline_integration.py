import os


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

from chunker import ChunkingConfig, chunk_document
from document_parser import parse_document
from enrichment import MetadataNormalizer, enrich_document
from extraction import extract_graph
from graph_normalization import GraphView, normalize_graph
from ingestion_pipeline import IngestionPipeline, ingest_documents
from schemas import DocumentRef, Entity, ExtractionResult
from storage_connectors import (
    index_chunks,
    index_graph,
    persist_chunks,
    persist_document,
    persist_graph,
)


def test_typed_ingestion_connectors_compose_without_ui_or_external_services(tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("Alpha is an organization.", encoding="utf-8")
    document = DocumentRef(path=source)
    calls = []

    class Parser:
        def parse(self, path):
            return [(0, path.read_text(encoding="utf-8"))], {
                "doc_id": "doc-1",
                "Language": "en",
            }

    class Extractor:
        def extract(self, chunks):
            assert chunks[0].text == "Alpha is an organization."
            return ExtractionResult(
                entities=[
                    Entity(
                        name="Alpha",
                        type="Organization",
                        description="An organization",
                        source_ids=(chunks[0].chunk_uuid,),
                    )
                ]
            )

    class Stores:
        def put_raw(self, raw):
            calls.append(("raw", raw.doc_id))

        def put_enriched(self, enriched):
            calls.append(("enriched", enriched.transformations))

        def put_many(self, chunks):
            calls.append(("chunks", tuple(chunk.chunk_uuid for chunk in chunks)))

        def apply(self, delta):
            calls.append(("graph", tuple(entity.name for entity in delta.entities)))

        def index_chunks(self, chunks):
            calls.append(("chunk_index", len(chunks)))

        def index_graph(self, delta):
            calls.append(("graph_index", len(delta.entities)))

    stores = Stores()
    parser = Parser()
    extractor = Extractor()
    pipeline = IngestionPipeline(
        parse=lambda ref: parse_document(ref, parser),
        enrich=lambda raw: enrich_document(raw, [MetadataNormalizer()]),
        chunk=lambda enriched: chunk_document(
            enriched,
            ChunkingConfig(max_chars=100, overlap_chars=0),
            id_factory=lambda: "chunk-1",
        ),
        extract=lambda chunks: extract_graph(chunks, extractor),
        normalize=lambda extraction: normalize_graph(extraction, GraphView()),
        persist_document=lambda raw, enriched: persist_document(
            raw,
            enriched,
            stores,
        ),
        persist_chunks=lambda chunks: persist_chunks(chunks, stores),
        persist_graph=lambda delta: persist_graph(delta, stores),
        index_chunks=lambda chunks: index_chunks(chunks, stores),
        index_graph=lambda delta: index_graph(delta, stores),
    )

    summary = ingest_documents([document], pipeline)

    assert summary.processed_files == 1
    assert summary.chunk_count == 1
    assert summary.entity_count == 1
    assert summary.relation_count == 0
    assert calls == [
        ("raw", "doc-1"),
        ("enriched", ("normalize_metadata",)),
        ("chunks", ("chunk-1",)),
        ("graph", ("Alpha",)),
        ("chunk_index", 1),
        ("graph_index", 1),
    ]

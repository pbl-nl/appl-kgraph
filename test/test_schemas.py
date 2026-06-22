import os
import sys
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from schemas import (
    AnswerResult,
    Chunk,
    DocumentRef,
    EnrichedDocument,
    Entity,
    ExtractionResult,
    GraphDelta,
    QueryPlan,
    RawDocument,
    Relation,
    RetrievedContext,
)
from extractor import parse_model_output
from ingestion import build_chunks


def test_pipeline_schema_objects_form_standalone_boundaries(tmp_path):
    doc_ref = DocumentRef(path=tmp_path / "report.md", root=tmp_path)
    raw = RawDocument(ref=doc_ref, pages=[(0, "Alpha")], text="Alpha")
    enriched = EnrichedDocument(raw=raw, text="Alpha", metadata={"language": "en"})
    chunk = Chunk(
        chunk_uuid="chunk-1",
        doc_id="doc-1",
        chunk_id=0,
        filename="report.md",
        text="Alpha",
        char_count=5,
        start_page=0,
        end_page=0,
    )
    entity = Entity(name="Alpha", type="category", description="A test entity")
    graph_delta = GraphDelta(entities=[entity])
    query_plan = QueryPlan(query="What is Alpha?")
    context = RetrievedContext(query_plan=query_plan, entities=graph_delta.entities, chunks=[chunk])
    answer = AnswerResult(answer="Alpha is a test entity.", context=context)

    assert answer.context.query_plan.query == "What is Alpha?"
    assert answer.context.entities[0].name == "Alpha"
    assert answer.context.chunks[0].chunk_uuid == "chunk-1"
    assert enriched.raw.ref.path == tmp_path / "report.md"


def test_schema_defaults_are_not_shared():
    first = GraphDelta()
    second = GraphDelta()
    first_extraction = ExtractionResult()
    second_extraction = ExtractionResult()

    first.entities.append(Entity(name="Alpha", type="category", description="A"))
    first_extraction.content_keywords.append("alpha")
    first_extraction.chunk_results.append({"chunk_uuid": "chunk-1"})
    first_extraction.diagnostics.append({"summary": "complete"})

    assert second.entities == []
    assert second_extraction.content_keywords == []
    assert second_extraction.chunk_results == []
    assert second_extraction.diagnostics == []


def test_chunk_contract_accepts_current_ingestion_payload():
    payloads = build_chunks(
        [(0, "Alpha is the first sentence. Beta is the second sentence.")],
        doc_id="doc-1",
        filename="report.md",
        filepath="/documents/report.md",
        document_language="en",
    )

    chunks = [Chunk(**payload) for payload in payloads]

    assert len(chunks) == 1
    assert chunks[0].chunk_uuid
    assert chunks[0].doc_id == "doc-1"
    assert chunks[0].chunk_id == 0
    assert chunks[0].char_count == len(chunks[0].text)
    assert chunks[0].start_page == 0
    assert chunks[0].end_page == 0
    assert chunks[0].filepath == "/documents/report.md"
    assert chunks[0].document_language == "en"


def test_graph_contracts_accept_current_extractor_payloads():
    parsed = parse_model_output(
        "(entity|Alpha|Organization|Primary entity)##"
        "(relationship|Alpha|Beta|Works with|collaboration|0.75)<done>",
        tuple_delim="|",
        record_delim="##",
        completion_delim="<done>",
    )

    entities = [Entity(**payload) for payload in parsed.entities]
    relations = [Relation(**payload) for payload in parsed.relationships]
    result = ExtractionResult(
        entities=entities,
        relations=relations,
        content_keywords=["partnership"],
        chunk_results=[{"chunk_uuid": "chunk-1", "raw_output": "raw"}],
        diagnostics=[{"chunk_uuid": "chunk-1", "summary": "complete"}],
    )

    assert result.entities == [
        Entity(
            name="Alpha",
            type="Organization",
            description="Primary entity",
        )
    ]
    assert result.relations == [
        Relation(
            source_name="Alpha",
            target_name="Beta",
            description="Works with",
            keywords="collaboration",
            weight=0.75,
        )
    ]
    assert result.content_keywords == ["partnership"]
    assert result.chunk_results[0]["raw_output"] == "raw"
    assert result.diagnostics[0]["summary"] == "complete"

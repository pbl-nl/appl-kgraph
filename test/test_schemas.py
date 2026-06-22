import os
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from schemas import (
    AnswerResult,
    Chunk,
    ChunkCandidate,
    DocumentRef,
    EnrichedDocument,
    Entity,
    EntityCandidate,
    ExtractionResult,
    GraphDelta,
    QueryPlan,
    RawDocument,
    Relation,
    RelationCandidate,
    RetrievalCandidates,
    RetrievedContext,
    extraction_result_from_legacy,
    extraction_result_to_legacy,
    retrieval_candidates_from_legacy,
    retrieval_candidates_to_legacy,
)
from extractor import parse_model_output
from ingestion import build_chunks
from pathrag import ChunkMatch, EntityMatch, RelationMatch


def test_pipeline_schema_objects_form_standalone_boundaries(tmp_path):
    doc_ref = DocumentRef(path=tmp_path / "report.md", root=tmp_path)
    raw = RawDocument(
        ref=doc_ref,
        doc_id="doc-1",
        pages=[(0, "Alpha")],
        text="Alpha",
    )
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
    assert enriched.raw.doc_id == answer.context.chunks[0].doc_id


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


def test_retrieval_candidates_accept_current_pathrag_matches():
    query_plan = QueryPlan(
        query="How are Alpha and Beta related?",
        high_level_keywords=["partnership"],
        low_level_keywords=["Alpha", "Beta"],
    )
    entity_match = EntityMatch(
        name="Alpha",
        type="Organization",
        description="Primary entity",
        score=0.9,
    )
    relation_match = RelationMatch(
        source_name="Alpha",
        target_name="Beta",
        description="Works with",
        keywords="collaboration",
        score=0.8,
    )
    chunk_match = ChunkMatch(
        chunk_uuid="chunk-1",
        document_id="doc-1",
        filename="report.md",
        text="Supporting text",
        score=0.7,
    )

    candidates = RetrievalCandidates(
        query_plan=query_plan,
        entities=[EntityCandidate(**asdict(entity_match))],
        relations=[RelationCandidate(**asdict(relation_match))],
        chunks=[ChunkCandidate(**asdict(chunk_match))],
    )

    assert candidates.query_plan is query_plan
    assert candidates.entities[0].name == "Alpha"
    assert candidates.entities[0].score == 0.9
    assert candidates.relations[0].keywords == "collaboration"
    assert candidates.relations[0].score == 0.8
    assert candidates.chunks[0].chunk_uuid == "chunk-1"
    assert candidates.chunks[0].document_id == "doc-1"
    assert candidates.chunks[0].score == 0.7


def test_retrieval_candidate_defaults_are_not_shared():
    first = RetrievalCandidates(query_plan=QueryPlan(query="first"))
    second = RetrievalCandidates(query_plan=QueryPlan(query="second"))

    first.entities.append(
        EntityCandidate(
            name="Alpha",
            type=None,
            description="",
            score=0.5,
        )
    )
    first.metadata["strategy"] = "pathrag"

    assert second.entities == []
    assert second.metadata == {}


def test_pathrag_candidate_payload_round_trip():
    query_plan = QueryPlan(query="How are Alpha and Beta related?")
    entity_match = EntityMatch(
        name="Alpha",
        type="Organization",
        description="Primary entity",
        score=0.9,
    )
    relation_match = RelationMatch(
        source_name="Alpha",
        target_name="Beta",
        description="Works with",
        keywords="collaboration",
        score=0.8,
    )
    chunk_match = ChunkMatch(
        chunk_uuid="chunk-1",
        document_id="doc-1",
        filename="report.md",
        text="Supporting text",
        score=0.7,
    )
    legacy = {
        "entity_matches": [entity_match],
        "relation_matches": [relation_match],
        "chunk_matches": [chunk_match],
        "strategy": "pathrag",
    }

    candidates = retrieval_candidates_from_legacy(query_plan, legacy)

    assert candidates.query_plan is query_plan
    assert candidates.metadata == {"strategy": "pathrag"}
    assert retrieval_candidates_to_legacy(candidates) == {
        "entity_matches": [asdict(entity_match)],
        "relation_matches": [asdict(relation_match)],
        "chunk_matches": [asdict(chunk_match)],
        "strategy": "pathrag",
    }


def test_lightrag_candidate_payload_preserves_ids_and_input_ownership():
    query_plan = QueryPlan(query="Question")
    legacy = {
        "entities": [
            {
                "id": "Alpha",
                "name": "Alpha",
                "type": None,
                "description": "Entity description",
                "score": 0.9,
            }
        ],
        "relations": [
            {
                "id": "Alpha::Beta",
                "source_name": "Alpha",
                "target_name": "Beta",
                "description": "Related",
                "keywords": "connection",
                "score": 0.8,
            }
        ],
        "chunks": [
            {
                "chunk_uuid": "chunk-1",
                "document_id": "doc-1",
                "filename": "report.md",
                "text": "Supporting text",
                "score": 0.7,
            }
        ],
    }

    candidates = retrieval_candidates_from_legacy(query_plan, legacy)

    assert candidates.entities[0].metadata == {"id": "Alpha"}
    assert candidates.relations[0].metadata == {"id": "Alpha::Beta"}
    assert candidates.chunks[0].document_id == "doc-1"
    emitted = retrieval_candidates_to_legacy(candidates)
    assert emitted["entity_matches"][0]["id"] == "Alpha"
    assert emitted["relation_matches"][0]["id"] == "Alpha::Beta"

    legacy["entities"][0]["id"] = "Changed"
    legacy["chunks"][0]["text"] = "Changed"
    assert candidates.entities[0].metadata == {"id": "Alpha"}
    assert candidates.chunks[0].text == "Supporting text"


def test_retrieval_candidate_conversion_accepts_empty_payload():
    query_plan = QueryPlan(query="Question")

    candidates = retrieval_candidates_from_legacy(query_plan, {})

    assert candidates == RetrievalCandidates(query_plan=query_plan)
    assert retrieval_candidates_to_legacy(candidates) == {
        "entity_matches": [],
        "relation_matches": [],
        "chunk_matches": [],
    }


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


def test_empty_raw_document_preserves_stable_identity():
    raw = RawDocument(
        ref=DocumentRef(path=Path("empty.txt")),
        doc_id="doc-empty",
        pages=[],
        text="",
    )

    assert raw.doc_id == "doc-empty"
    assert raw.pages == ()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"doc_id": ""}, "doc_id must not be empty"),
        ({"pages": [(1, "Alpha")]}, "zero-based and contiguous"),
        ({"pages": [(0, "Alpha"), (2, "Beta")]}, "zero-based and contiguous"),
        ({"text": "Different"}, "newline-joined page text"),
    ],
)
def test_raw_document_rejects_invalid_identity_and_pages(overrides, message):
    values = {
        "ref": DocumentRef(path=Path("report.md")),
        "doc_id": "doc-1",
        "pages": [(0, "Alpha")],
        "text": "Alpha",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        RawDocument(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_uuid": ""}, "chunk_uuid must not be empty"),
        ({"doc_id": ""}, "doc_id must not be empty"),
        ({"chunk_id": -1}, "chunk_id must be zero or greater"),
        ({"char_count": 4}, "char_count must equal"),
        ({"start_page": -1}, "start_page must be zero or greater"),
        ({"start_page": 2, "end_page": 1}, "end_page must be greater"),
    ],
)
def test_chunk_rejects_invalid_identity_range_and_character_count(
    overrides,
    message,
):
    values = {
        "chunk_uuid": "chunk-1",
        "doc_id": "doc-1",
        "chunk_id": 0,
        "filename": "report.md",
        "text": "Alpha",
        "char_count": 5,
        "start_page": 0,
        "end_page": 0,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        Chunk(**values)


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


def test_extraction_result_legacy_payload_round_trip():
    legacy = {
        "entities": [
            {
                "name": "Alpha",
                "type": "Organization",
                "description": "Primary entity",
                "source_id": "chunk-1",
                "filepath": "/documents/report.md",
                "confidence": 0.95,
            }
        ],
        "relationships": [
            {
                "source_name": "Alpha",
                "target_name": "Beta",
                "description": "Works with",
                "keywords": "collaboration",
                "weight": 0.75,
                "source_id": "chunk-1",
                "filepath": "/documents/report.md",
                "provenance": "model",
            }
        ],
        "content_keywords": ["partnership"],
        "chunk_results": [
            {
                "chunk_uuid": "chunk-1",
                "entities": [{"name": "Alpha"}],
                "raw_output": "raw extraction",
            }
        ],
        "validation_results": [
            {"chunk_uuid": "chunk-1", "summary": "complete"}
        ],
        "audits": [{"chunk_uuid": "chunk-1", "summary": "complete"}],
        "request_id": "request-1",
    }
    expected = deepcopy(legacy)

    result = extraction_result_from_legacy(legacy)

    assert result.entities[0].metadata == {"confidence": 0.95}
    assert result.relations[0].metadata == {"provenance": "model"}
    assert result.metadata == {"request_id": "request-1"}
    assert extraction_result_to_legacy(result) == expected

    legacy["entities"][0]["name"] = "Changed"
    legacy["chunk_results"][0]["entities"][0]["name"] = "Changed"
    assert result.entities[0].name == "Alpha"
    assert result.chunk_results[0]["entities"][0]["name"] == "Alpha"


def test_extraction_result_legacy_conversion_accepts_aliases_and_empty_payloads():
    empty = extraction_result_from_legacy({})
    aliased = extraction_result_from_legacy(
        {
            "relations": [
                {
                    "source_name": "Alpha",
                    "target_name": "Beta",
                    "description": "Related",
                }
            ],
            "audits": [{"summary": "legacy diagnostic"}],
        }
    )

    assert empty == ExtractionResult()
    assert aliased.relations[0].source_name == "Alpha"
    assert aliased.diagnostics == [{"summary": "legacy diagnostic"}]
    legacy = extraction_result_to_legacy(aliased)
    assert legacy["relationships"][0]["target_name"] == "Beta"
    assert legacy["validation_results"] == [{"summary": "legacy diagnostic"}]
    assert legacy["audits"] == [{"summary": "legacy diagnostic"}]

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
    Entity,
    GraphDelta,
    QueryPlan,
    RawDocument,
    RetrievedContext,
)


def test_pipeline_schema_objects_form_standalone_boundaries(tmp_path):
    doc_ref = DocumentRef(path=tmp_path / "report.md", root=tmp_path)
    raw = RawDocument(ref=doc_ref, pages=[(0, "Alpha")], text="Alpha")
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


def test_schema_defaults_are_not_shared():
    first = GraphDelta()
    second = GraphDelta()

    first.entities.append(Entity(name="Alpha", type="category", description="A"))

    assert second.entities == []

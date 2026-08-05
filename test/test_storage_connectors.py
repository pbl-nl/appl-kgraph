from pathlib import Path

import pytest

from schemas import (
    Chunk,
    DocumentRef,
    EnrichedDocument,
    Entity,
    GraphDelta,
    RawDocument,
)
from storage_connectors import (
    CanonicalStorageError,
    VectorIndexError,
    index_chunks,
    index_graph,
    persist_chunks,
    persist_document,
    persist_graph,
)


def _documents(doc_id="doc-1"):
    raw = RawDocument(
        ref=DocumentRef(path=Path("report.txt")),
        doc_id=doc_id,
        pages=((0, "Alpha"),),
        text="Alpha",
    )
    return raw, EnrichedDocument(raw=raw, text="Alpha")


def _chunk(doc_id="doc-1", chunk_uuid="chunk-1"):
    return Chunk(
        chunk_uuid=chunk_uuid,
        doc_id=doc_id,
        chunk_id=0,
        filename="report.txt",
        text="Alpha",
        char_count=5,
        start_page=0,
        end_page=0,
    )


def test_storage_and_index_connectors_forward_typed_inputs():
    calls = []

    class StoreAndIndex:
        def put_raw(self, document):
            calls.append(("put_raw", document))

        def put_enriched(self, document):
            calls.append(("put_enriched", document))

        def put_many(self, chunks):
            calls.append(("put_many", chunks))

        def apply(self, delta):
            calls.append(("apply", delta))

        def index_chunks(self, chunks):
            calls.append(("index_chunks", chunks))

        def index_graph(self, delta):
            calls.append(("index_graph", delta))

    connector = StoreAndIndex()
    raw, enriched = _documents()
    chunks = [_chunk()]
    delta = GraphDelta(
        entities=[Entity(name="Alpha", type="Thing", description="Entity")]
    )

    assert persist_document(raw, enriched, connector) is None
    assert persist_chunks(chunks, connector) is None
    assert persist_graph(delta, connector) is None
    assert index_chunks(chunks, connector) is None
    assert index_graph(delta, connector) is None

    assert calls == [
        ("put_raw", raw),
        ("put_enriched", enriched),
        ("put_many", tuple(chunks)),
        ("apply", delta),
        ("index_chunks", tuple(chunks)),
        ("index_graph", delta),
    ]


def test_document_connector_rejects_mismatched_document_identity():
    raw, _ = _documents("raw-id")
    _, enriched = _documents("enriched-id")

    with pytest.raises(ValueError, match="same doc_id"):
        persist_document(raw, enriched, object())


def test_chunk_connector_rejects_mixed_document_batch():
    with pytest.raises(ValueError, match="one document"):
        persist_chunks([_chunk("doc-1", "one"), _chunk("doc-2", "two")], object())


@pytest.mark.parametrize(
    ("method", "error_type", "operation"),
    [
        (
            lambda connector, chunk, delta: persist_chunks([chunk], connector),
            CanonicalStorageError,
            "put_many_chunks",
        ),
        (
            lambda connector, chunk, delta: persist_graph(delta, connector),
            CanonicalStorageError,
            "apply_graph_delta",
        ),
        (
            lambda connector, chunk, delta: index_chunks([chunk], connector),
            VectorIndexError,
            "index_chunks",
        ),
        (
            lambda connector, chunk, delta: index_graph(delta, connector),
            VectorIndexError,
            "index_graph",
        ),
    ],
)
def test_connector_failures_identify_operation(method, error_type, operation):
    class FailingConnector:
        def __getattr__(self, name):
            def fail(*args):
                raise OSError("disk unavailable")

            return fail

    with pytest.raises(error_type, match=operation) as caught:
        method(FailingConnector(), _chunk(), GraphDelta())

    assert caught.value.operation == operation
    assert isinstance(caught.value.__cause__, OSError)


def test_document_connector_identifies_enriched_write_failure():
    class PartiallyFailingStore:
        def put_raw(self, document):
            pass

        def put_enriched(self, document):
            raise OSError("snapshot unavailable")

    raw, enriched = _documents()

    with pytest.raises(CanonicalStorageError, match="put_enriched") as caught:
        persist_document(raw, enriched, PartiallyFailingStore())

    assert caught.value.operation == "put_enriched"

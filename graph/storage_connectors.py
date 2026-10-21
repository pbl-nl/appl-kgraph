from __future__ import annotations

from typing import Protocol, Sequence

from schemas import Chunk, EnrichedDocument, GraphDelta, RawDocument


class DocumentStore(Protocol):
    """Canonical raw/enriched document persistence boundary."""

    def put_raw(self, document: RawDocument) -> None:
        ...

    def put_enriched(self, document: EnrichedDocument) -> None:
        ...


class ChunkStore(Protocol):
    """Canonical chunk persistence boundary."""

    def put_many(self, chunks: Sequence[Chunk]) -> None:
        ...


class GraphStore(Protocol):
    """Canonical graph-delta persistence boundary."""

    def apply(self, delta: GraphDelta) -> None:
        ...


class VectorIndex(Protocol):
    """Chunk and graph vector-index boundary."""

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        ...

    def index_graph(self, delta: GraphDelta) -> None:
        ...


class CanonicalStorageError(RuntimeError):
    """Failure in one named canonical-storage operation."""

    def __init__(self, operation: str, cause: Exception) -> None:
        self.operation = operation
        super().__init__(f"Canonical storage operation {operation} failed: {cause}")


class VectorIndexError(RuntimeError):
    """Failure in one named vector-index operation."""

    def __init__(self, operation: str, cause: Exception) -> None:
        self.operation = operation
        super().__init__(f"Vector index operation {operation} failed: {cause}")


def persist_document(
    raw: RawDocument,
    enriched: EnrichedDocument,
    store: DocumentStore,
) -> None:
    """Persist both canonical document representations in stage order."""

    if not isinstance(raw, RawDocument):
        raise TypeError("raw must be a RawDocument")
    if not isinstance(enriched, EnrichedDocument):
        raise TypeError("enriched must be an EnrichedDocument")
    if enriched.raw.doc_id != raw.doc_id:
        raise ValueError("raw and enriched documents must have the same doc_id")
    try:
        store.put_raw(raw)
    except Exception as exc:
        raise CanonicalStorageError("put_raw", exc) from exc
    try:
        store.put_enriched(enriched)
    except Exception as exc:
        raise CanonicalStorageError("put_enriched", exc) from exc


def persist_chunks(chunks: Sequence[Chunk], store: ChunkStore) -> None:
    """Persist one validated chunk batch."""

    chunk_batch = _chunk_batch(chunks)
    try:
        store.put_many(chunk_batch)
    except Exception as exc:
        raise CanonicalStorageError("put_many_chunks", exc) from exc


def persist_graph(delta: GraphDelta, store: GraphStore) -> None:
    """Apply one validated graph delta."""

    if not isinstance(delta, GraphDelta):
        raise TypeError("delta must be a GraphDelta")
    try:
        store.apply(delta)
    except Exception as exc:
        raise CanonicalStorageError("apply_graph_delta", exc) from exc


def index_chunks(chunks: Sequence[Chunk], index: VectorIndex) -> None:
    """Index one validated chunk batch."""

    chunk_batch = _chunk_batch(chunks)
    try:
        index.index_chunks(chunk_batch)
    except Exception as exc:
        raise VectorIndexError("index_chunks", exc) from exc


def index_graph(delta: GraphDelta, index: VectorIndex) -> None:
    """Index one validated graph delta."""

    if not isinstance(delta, GraphDelta):
        raise TypeError("delta must be a GraphDelta")
    try:
        index.index_graph(delta)
    except Exception as exc:
        raise VectorIndexError("index_graph", exc) from exc


def _chunk_batch(chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise TypeError("chunks must be a sequence of Chunk objects")
    batch = tuple(chunks)
    if not all(isinstance(chunk, Chunk) for chunk in batch):
        raise TypeError("chunks must contain only Chunk objects")
    doc_ids = {chunk.doc_id for chunk in batch}
    if len(doc_ids) > 1:
        raise ValueError("one chunk batch must belong to one document")
    chunk_ids = [chunk.chunk_uuid for chunk in batch]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk batch identifiers must be unique")
    return batch

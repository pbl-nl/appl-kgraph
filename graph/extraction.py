from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from schemas import Chunk, ExtractionResult, extraction_result_from_legacy


class GraphExtractor(Protocol):
    """Replaceable entity/relation extraction implementation."""

    def extract(self, chunks: Sequence[Chunk]) -> ExtractionResult:
        ...


def chunk_to_legacy(chunk: Chunk) -> dict[str, Any]:
    """Convert a canonical chunk at the legacy extractor boundary."""

    payload = dict(chunk.metadata)
    payload.update(
        {
            "chunk_uuid": chunk.chunk_uuid,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "filename": chunk.filename,
            "filepath": chunk.filepath,
            "document_language": chunk.document_language,
            "text": chunk.text,
            "char_count": chunk.char_count,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
        }
    )
    return payload


@dataclass(frozen=True)
class LegacyGraphExtractor:
    """Adapter for the current dictionary-based extraction implementation."""

    extractor: Callable[[Sequence[dict[str, Any]]], Mapping[str, Any]]

    def extract(self, chunks: Sequence[Chunk]) -> ExtractionResult:
        payload = self.extractor([chunk_to_legacy(chunk) for chunk in chunks])
        return extraction_result_from_legacy(payload)


def extract_graph(
    chunks: Sequence[Chunk],
    extractor: GraphExtractor,
) -> ExtractionResult:
    """Extract graph findings from chunks through one typed connector."""

    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise TypeError("chunks must be a sequence of Chunk objects")
    if not all(isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("chunks must contain only Chunk objects")
    result = extractor.extract(tuple(chunks))
    if not isinstance(result, ExtractionResult):
        raise TypeError("extractor must return an ExtractionResult")
    return result

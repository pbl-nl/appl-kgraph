from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, Tuple

from schemas import EnrichedDocument, RawDocument


DocumentPages = Sequence[Tuple[int, str]]
EnrichmentOutput = Tuple[DocumentPages, Mapping[str, Any]]


class TextEnricher(Protocol):
    """One named text/metadata transformation in the enrichment stage."""

    name: str

    def enrich(
        self,
        pages: DocumentPages,
        metadata: Mapping[str, Any],
    ) -> EnrichmentOutput:
        ...


def normalize_document_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the legacy-compatible, lower-case metadata representation."""

    return {str(key).lower(): value for key, value in (metadata or {}).items()}


@dataclass(frozen=True)
class MetadataNormalizer:
    """Behavior-preserving metadata normalization enrichment."""

    name: str = "normalize_metadata"

    def enrich(
        self,
        pages: DocumentPages,
        metadata: Mapping[str, Any],
    ) -> EnrichmentOutput:
        return pages, normalize_document_metadata(metadata)


def enrich_document(
    raw: RawDocument,
    enrichers: Sequence[TextEnricher] = (),
) -> EnrichedDocument:
    """Apply ordered enrichers and return the canonical enriched document."""

    if not isinstance(raw, RawDocument):
        raise TypeError("raw must be a RawDocument")

    pages: DocumentPages = raw.pages
    metadata: Mapping[str, Any] = raw.metadata
    transformations = []

    for enricher in enrichers:
        name = getattr(enricher, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("enricher name must not be empty")
        output = enricher.enrich(pages, metadata)
        if (
            isinstance(output, (str, bytes))
            or not isinstance(output, Sequence)
            or len(output) != 2
        ):
            raise TypeError("enricher output must be a (pages, metadata) pair")
        pages, metadata = output
        if isinstance(pages, (str, bytes)) or not isinstance(pages, Sequence):
            raise TypeError("enricher pages must be a sequence")
        if not isinstance(metadata, Mapping):
            raise TypeError("enricher metadata must be a mapping")
        transformations.append(name)

    return EnrichedDocument(
        raw=raw,
        pages=tuple(pages),
        text="\n".join(page_text for _, page_text in pages),
        metadata=dict(metadata),
        transformations=tuple(transformations),
    )

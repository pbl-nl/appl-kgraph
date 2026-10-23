from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, Tuple

from schemas import DocumentRef, RawDocument


ParserOutput = Tuple[Sequence[Tuple[int, str]], Mapping[str, Any]]


class DocumentParser(Protocol):
    """Format adapter used by the typed parser stage."""

    def parse(self, path: Path) -> ParserOutput:
        ...


class DocumentParseError(Exception):
    """A parser-stage failure associated with one source document."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse {path}: {reason}")


def parse_document(document: DocumentRef, parser: DocumentParser) -> RawDocument:
    """Parse one reference into a canonical raw document."""

    try:
        pages, parsed_metadata = parser.parse(document.path)
        if not isinstance(parsed_metadata, Mapping):
            raise TypeError("parser metadata must be a mapping")

        metadata = dict(document.metadata)
        metadata.update(parsed_metadata)
        doc_id = metadata.pop("doc_id", None)
        return RawDocument(
            ref=document,
            doc_id=doc_id,
            pages=pages,
            text="\n".join(page_text for _, page_text in pages),
            metadata=metadata,
        )
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(document.path, str(exc)) from exc

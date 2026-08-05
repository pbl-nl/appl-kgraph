import os
from pathlib import Path

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

from chunker import ChunkingConfig, chunk_document
from schemas import Chunk, DocumentRef, EnrichedDocument, RawDocument


def _document(pages=((0, "Alpha. Beta."),), *, enriched_pages=None):
    raw = RawDocument(
        ref=DocumentRef(path=Path("nested/report.txt")),
        doc_id="doc-1",
        pages=pages,
        text="\n".join(text for _, text in pages),
        metadata={"language": "nl"},
    )
    output_pages = pages if enriched_pages is None else enriched_pages
    return EnrichedDocument(
        raw=raw,
        pages=output_pages,
        text="\n".join(text for _, text in output_pages),
        metadata={"language": "en"},
        transformations=("translate",),
    )


def test_chunk_document_returns_typed_chunks_with_owned_metadata():
    ids = iter(["chunk-1", "chunk-2"])
    document = _document(
        pages=((0, "Raw one."), (1, "Raw two.")),
        enriched_pages=((0, "First sentence."), (1, "Second sentence.")),
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(max_chars=16, overlap_chars=0),
        id_factory=lambda: next(ids),
    )

    assert chunks == [
        Chunk(
            chunk_uuid="chunk-1",
            doc_id="doc-1",
            chunk_id=0,
            filename="report.txt",
            filepath=str(Path("nested/report.txt").resolve()),
            document_language="en",
            text="First sentence.",
            char_count=15,
            start_page=0,
            end_page=0,
            metadata={
                "sentence_span": [(0, 0, "First sentence.")],
                "overlap_from_previous": False,
                "overlap_chars_effective": 0,
                "included_new_sentence_count": 1,
                "include_overlap_in_limit": True,
                "max_chars_target": 16,
                "exceeds_target": False,
            },
        ),
        Chunk(
            chunk_uuid="chunk-2",
            doc_id="doc-1",
            chunk_id=1,
            filename="report.txt",
            filepath=str(Path("nested/report.txt").resolve()),
            document_language="en",
            text="Second sentence.",
            char_count=16,
            start_page=1,
            end_page=1,
            metadata={
                "sentence_span": [(1, 1, "Second sentence.")],
                "overlap_from_previous": False,
                "overlap_chars_effective": 0,
                "included_new_sentence_count": 1,
                "include_overlap_in_limit": True,
                "max_chars_target": 16,
                "exceeds_target": False,
            },
        ),
    ]


def test_chunk_document_returns_empty_for_empty_document():
    assert chunk_document(
        _document(pages=()),
        ChunkingConfig(max_chars=20, overlap_chars=0),
    ) == []


def test_chunk_document_rejects_duplicate_generated_ids():
    with pytest.raises(ValueError, match="duplicate"):
        chunk_document(
            _document(),
            ChunkingConfig(max_chars=7, overlap_chars=0),
            id_factory=lambda: "duplicate",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_chars": 0}, "greater than zero"),
        ({"max_chars": 10, "overlap_chars": -1}, "zero or greater"),
        ({"max_chars": 10, "overlap_chars": 10}, "smaller"),
        ({"max_chars": 10, "overlap_chars": 0, "join_with": None}, "string"),
    ],
)
def test_chunking_config_rejects_invalid_boundaries(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ChunkingConfig(**kwargs)

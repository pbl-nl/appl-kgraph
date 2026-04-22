import os
import sys
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from graph.chunker import chunk_text


def test_chunk_overlap_does_not_cascade_previous_overlap():
    text = "A1. B2. C3. D4. E5. F6."
    chunks = chunk_text(
        text,
        max_chars=11,
        overlap_chars=7,
        include_overlap_in_limit=True,
    )

    assert [chunk["text"] for chunk in chunks] == [
        "A1. B2. C3.",
        "B2. C3. D4.",
        "D4. E5. F6.",
    ]


def test_chunker_keeps_oversized_sentence_intact():
    text = "This sentence is deliberately much longer than the configured chunk size."
    chunks = chunk_text(
        text,
        max_chars=10,
        overlap_chars=0,
        include_overlap_in_limit=True,
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["char_count"] > 10
    assert chunks[0]["exceeds_target"] is True

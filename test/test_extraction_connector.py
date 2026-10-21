from pathlib import Path

import pytest

from extraction import LegacyGraphExtractor, chunk_to_legacy, extract_graph
from schemas import Chunk, Entity, ExtractionResult


def _chunk():
    return Chunk(
        chunk_uuid="chunk-1",
        doc_id="doc-1",
        chunk_id=0,
        filename="report.txt",
        filepath=str(Path("report.txt").resolve()),
        document_language="en",
        text="Alpha",
        char_count=5,
        start_page=0,
        end_page=0,
        metadata={"sentence_span": [(0, 0, "Alpha")]},
    )


def test_extract_graph_passes_typed_chunks_and_returns_typed_result():
    expected = ExtractionResult(
        entities=[Entity(name="Alpha", type="Thing", description="Entity")]
    )

    class FakeExtractor:
        def __init__(self):
            self.calls = []

        def extract(self, chunks):
            self.calls.append(chunks)
            return expected

    extractor = FakeExtractor()
    chunk = _chunk()

    result = extract_graph([chunk], extractor)

    assert result is expected
    assert extractor.calls == [(chunk,)]


def test_legacy_extractor_adapter_preserves_current_chunk_payload():
    calls = []

    def legacy_extract(chunks):
        calls.append(chunks)
        return {
            "entities": [
                {
                    "name": "Alpha",
                    "type": "Thing",
                    "description": "Entity",
                    "source_id": "chunk-1",
                }
            ],
            "relationships": [],
            "content_keywords": ["alpha"],
            "validation_results": [{"chunk_uuid": "chunk-1", "ok": True}],
        }

    chunk = _chunk()
    result = extract_graph([chunk], LegacyGraphExtractor(legacy_extract))

    assert calls == [[chunk_to_legacy(chunk)]]
    assert result.entities[0].source_ids == ("chunk-1",)
    assert result.content_keywords == ["alpha"]
    assert result.diagnostics == [{"chunk_uuid": "chunk-1", "ok": True}]


def test_extract_graph_accepts_empty_chunk_sequence():
    class EmptyExtractor:
        def extract(self, chunks):
            assert chunks == ()
            return ExtractionResult()

    assert extract_graph([], EmptyExtractor()) == ExtractionResult()


@pytest.mark.parametrize("chunks", [None, "chunk", [{}]])
def test_extract_graph_rejects_invalid_inputs_before_calling_extractor(chunks):
    class UnexpectedExtractor:
        def extract(self, chunks):
            raise AssertionError("extractor should not be called")

    with pytest.raises(TypeError, match="chunks"):
        extract_graph(chunks, UnexpectedExtractor())


def test_extract_graph_rejects_invalid_extractor_output():
    class InvalidExtractor:
        def extract(self, chunks):
            return {"entities": []}

    with pytest.raises(TypeError, match="ExtractionResult"):
        extract_graph([_chunk()], InvalidExtractor())

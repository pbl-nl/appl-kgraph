import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")
sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from document_parser import DocumentParseError, parse_document
from schemas import DocumentRef


class _FakeParser:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.paths = []

    def parse(self, path):
        self.paths.append(path)
        if self.error is not None:
            raise self.error
        return self.output


def test_parse_document_builds_raw_contract_and_assigns_metadata(tmp_path):
    path = tmp_path / "report.txt"
    document = DocumentRef(
        path=path,
        root=tmp_path,
        metadata={
            "doc_id": "discovery-id",
            "classification": "public",
            "language": "unknown",
        },
    )
    parser = _FakeParser(
        (
            [(0, "First page"), (1, "Second page")],
            {"doc_id": "parser-id", "language": "en", "mime_type": "text/plain"},
        )
    )

    raw = parse_document(document, parser)

    assert parser.paths == [path]
    assert raw.ref is document
    assert raw.doc_id == "parser-id"
    assert raw.pages == ((0, "First page"), (1, "Second page"))
    assert raw.text == "First page\nSecond page"
    assert raw.metadata == {
        "classification": "public",
        "language": "en",
        "mime_type": "text/plain",
    }


def test_parse_document_accepts_explicit_empty_parser_output(tmp_path):
    document = DocumentRef(path=tmp_path / "empty.txt", root=tmp_path)
    parser = _FakeParser(([], {"doc_id": "doc-empty", "language": "unknown"}))

    raw = parse_document(document, parser)

    assert raw.doc_id == "doc-empty"
    assert raw.pages == ()
    assert raw.text == ""


def test_parse_document_wraps_adapter_failures_with_source_path(tmp_path):
    document = DocumentRef(path=tmp_path / "broken.pdf", root=tmp_path)
    parser = _FakeParser(error=RuntimeError("decoder failed"))

    with pytest.raises(DocumentParseError, match="decoder failed") as caught:
        parse_document(document, parser)

    assert caught.value.path == document.path
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (([(0, "Text")], {}), "doc_id must be a string"),
        (([(1, "Text")], {"doc_id": "doc-1"}), "zero-based and contiguous"),
        (([(0, "Text")], []), "parser metadata must be a mapping"),
    ],
)
def test_parse_document_wraps_invalid_adapter_outputs(tmp_path, output, message):
    document = DocumentRef(path=tmp_path / "invalid.txt", root=tmp_path)

    with pytest.raises(DocumentParseError, match=message):
        parse_document(document, _FakeParser(output))

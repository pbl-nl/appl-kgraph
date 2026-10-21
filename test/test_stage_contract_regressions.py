from dataclasses import replace
from pathlib import Path

import pytest

from schemas import DocumentRef, RawDocument
from enrichment import enrich_document
from document_parser import DocumentParseError, parse_document
from production_ingestion import FileDocumentParser


def test_enrichment_cannot_mutate_raw_snapshot_metadata():
    raw = RawDocument(
        DocumentRef(Path("a.txt")),
        "doc",
        ((0, "Alpha"),),
        "Alpha",
        {"nested": {"values": ["original"]}},
    )

    class MutatingEnricher:
        name = "mutating"

        def enrich(self, pages, metadata):
            metadata["nested"]["values"].append("changed")
            return pages, metadata

    enriched = enrich_document(raw, [MutatingEnricher()])
    assert raw.metadata["nested"]["values"] == ["original"]
    assert enriched.metadata["nested"]["values"] == ["original", "changed"]


@pytest.mark.parametrize("extension", ["txt", "md", "html", "pdf"])
def test_real_format_adapter_returns_raw_contract(tmp_path, extension):
    path = tmp_path / ("report." + extension)
    if extension == "pdf":
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Alpha works with Beta.")
        doc.save(path)
        doc.close()
    else:
        path.write_text(
            "<html><body>Alpha works with Beta.</body></html>"
            if extension == "html"
            else "Alpha works with Beta.",
            encoding="utf-8",
        )
    raw = parse_document(DocumentRef(path), FileDocumentParser())
    assert raw.doc_id and "Alpha" in raw.text
    assert [page for page, _ in raw.pages] == list(range(len(raw.pages)))


def test_real_parser_failure_has_document_context(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_text("invalid PDF")
    with pytest.raises(DocumentParseError, match="broken.pdf"):
        parse_document(DocumentRef(path), FileDocumentParser())


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_pure_document_stages_import_without_model_credentials(tmp_path, provider):
    import os, subprocess, sys

    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("OPENAI_", "AZURE_OPENAI_")):
            del env[key]
    env["LLM_PROVIDER"] = provider
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "graph")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import chunker, document_parser, enrichment, production_ingestion",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_chat_still_requires_model_credentials(monkeypatch):
    import llm

    provider = replace(
        llm.settings.provider,
        provider="openai",
        openai_api_key=None,
        openai_llm_model=None,
    )
    monkeypatch.setattr(llm, "settings", replace(llm.settings, provider=provider))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm.Chat()


def test_html_cp1252_encoding_is_preserved(tmp_path):
    path = tmp_path / "report.html"
    path.write_bytes(
        "<html><body>caf\u00e9 works with Alpha.</body></html>".encode("cp1252")
    )
    raw = parse_document(DocumentRef(path), FileDocumentParser())
    assert "caf\u00e9" in raw.text


@pytest.mark.parametrize("broken_pdf", [False, True])
def test_docx_conversion_preserves_source_neighbors_and_cleans_temporary_files(
    tmp_path, monkeypatch, broken_pdf
):
    import fileparser
    import fitz

    source = tmp_path / "report.docx"
    source.write_bytes(b"conversion fixture")
    neighbor = tmp_path / "report.docx.pdf"
    neighbor.write_bytes(b"existing user file")
    outputs = []

    def convert(input_path, output_path, keep_active):
        assert Path(input_path) == source and keep_active is True
        pdf_path = Path(output_path)
        assert pdf_path.parent != source.parent
        outputs.append(pdf_path)
        if broken_pdf:
            pdf_path.write_bytes(b"invalid PDF")
        else:
            with fitz.open() as pdf:
                pdf.new_page().insert_text((72, 72), "Alpha works with Beta.")
                pdf.save(pdf_path)

    monkeypatch.setattr(fileparser, "convert", convert)
    if broken_pdf:
        with pytest.raises(DocumentParseError):
            parse_document(DocumentRef(source), FileDocumentParser())
    else:
        assert "Alpha" in parse_document(DocumentRef(source), FileDocumentParser()).text
    assert neighbor.read_bytes() == b"existing user file"
    assert outputs and not outputs[0].parent.exists()

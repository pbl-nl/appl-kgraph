import subprocess
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

import fileparser
from fileparser import FileParser


@pytest.fixture
def linux_converter(monkeypatch):
    monkeypatch.setattr(fileparser, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(fileparser.shutil, "which", lambda name: "/usr/bin/libreoffice")


def test_linux_docx_conversion_handles_spaces_and_unicode(tmp_path, monkeypatch, linux_converter):
    source = tmp_path / "résumé with spaces.DOCX"
    source.write_bytes(b"docx")
    output = tmp_path / "output folder"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        (output / "résumé with spaces.pdf").write_bytes(b"pdf")

    monkeypatch.setattr(fileparser.subprocess, "run", run)
    result = FileParser().convert_docx_to_pdf(str(source), output)

    assert Path(result) == output / "résumé with spaces.pdf"
    command, options = calls[0]
    assert command[0] == "/usr/bin/libreoffice"
    assert command[1].startswith("-env:UserInstallation=file://")
    assert command[-2:] == [str(output.resolve()), str(source.resolve())]
    assert "--headless" in command
    assert options == dict(check=True, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_word_backend_is_retained(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(fileparser, "sys", SimpleNamespace(platform=platform))
    calls = []

    def convert(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"pdf")

    monkeypatch.setitem(sys.modules, "docx2pdf", SimpleNamespace(convert=convert))
    source = tmp_path / "report.docx"
    source.write_bytes(b"docx")
    output = tmp_path / "output"
    result = FileParser().convert_docx_to_pdf(str(source), output)
    assert calls == [dict(input_path=str(source.resolve()), output_path=result, keep_active=False)]


def test_linux_docx_missing_converter_has_actionable_error(tmp_path, monkeypatch, linux_converter):
    monkeypatch.setattr(fileparser.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Install libreoffice-writer"):
        FileParser().convert_docx_to_pdf(str(tmp_path / "report.docx"), tmp_path / "out")


@pytest.mark.parametrize("failure", ["no_output", "empty_output", "exit_error", "timeout"])
def test_linux_docx_conversion_failures(tmp_path, monkeypatch, linux_converter, failure):
    output = tmp_path / "out"

    def run(command, **kwargs):
        if failure == "empty_output":
            (output / "report.pdf").touch()
        elif failure == "exit_error":
            raise subprocess.CalledProcessError(1, command)
        elif failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(fileparser.subprocess, "run", run)
    expected = {
        "no_output": RuntimeError,
        "empty_output": RuntimeError,
        "exit_error": subprocess.CalledProcessError,
        "timeout": subprocess.TimeoutExpired,
    }[failure]
    with pytest.raises(expected):
        FileParser().convert_docx_to_pdf(str(tmp_path / "report.docx"), output)


@pytest.mark.parametrize("failure_stage", [None, "convert", "parse"])
def test_docx_temp_files_are_cleaned_without_touching_sources(tmp_path, monkeypatch, failure_stage):
    source = tmp_path / "report.docx"
    source.write_bytes(b"docx")
    original_pdf = tmp_path / "report.pdf"
    original_pdf.write_bytes(b"original")
    parser = FileParser()
    output_dirs = []

    def convert(path, output_dir):
        output_dirs.append(Path(output_dir))
        pdf = Path(output_dir) / "report.pdf"
        pdf.write_bytes(b"temporary")
        if failure_stage == "convert":
            raise RuntimeError("conversion failed")
        return str(pdf)

    def parse(path):
        assert path.exists()
        if failure_stage == "parse":
            raise RuntimeError("parse failed")
        return [(0, "content")], {}

    monkeypatch.setattr(parser, "convert_docx_to_pdf", convert)
    monkeypatch.setattr(parser, "_parse_pdf_file", parse)
    if failure_stage:
        with pytest.raises(RuntimeError):
            parser._parse_docx_file(source)
    else:
        assert parser._parse_docx_file(source) == ([(0, "content")], {})
    assert not output_dirs[0].exists()
    assert original_pdf.read_bytes() == b"original"
    assert source.read_bytes() == b"docx"


def test_conversion_refuses_to_overwrite_existing_pdf(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        FileParser().convert_docx_to_pdf(str(tmp_path / "report.docx"), tmp_path)
    assert pdf.read_bytes() == b"original"


@pytest.mark.skipif(
    sys.platform != "linux" or not (shutil.which("libreoffice") or shutil.which("soffice")),
    reason="Requires Linux and LibreOffice Writer",
)
def test_real_linux_docx_conversion(tmp_path):
    source = tmp_path / "résumé with spaces.docx"
    with ZipFile(source, "w") as docx:
        docx.writestr("[Content_Types].xml", '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>''')
        docx.writestr("_rels/.rels", '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>''')
        docx.writestr("word/document.xml", '''<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body><w:p><w:r><w:t>This document verifies portable document conversion.</w:t></w:r></w:p></w:body>
</w:document>''')
    pages, metadata = FileParser().parse_file(source)
    assert "portable document conversion" in " ".join(text for _, text in pages)
    assert metadata["filename"] == source.name
    assert list(tmp_path.iterdir()) == [source]

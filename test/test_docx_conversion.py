from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import docx_conversion as conversion


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "documents with spaces" / "café report.DOCX"
    path.parent.mkdir()
    path.write_bytes(b"fake DOCX")
    return path


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_libreoffice_converts_with_private_profile_and_cleans_up(source, tmp_path, monkeypatch, platform):
    monkeypatch.setattr(conversion.sys, "platform", platform)
    monkeypatch.setattr(conversion, "find_libreoffice", lambda: "office executable")
    output = tmp_path / "converted.pdf"
    workdirs = []

    def run(command, **kwargs):
        assert command[0] == "office executable"
        assert command[-1] == str(source.resolve())
        assert "--headless" in command
        assert kwargs == {"capture_output": True, "timeout": 120, "check": True}
        workdir = Path(command[command.index("--outdir") + 1])
        assert command[1] == f"-env:UserInstallation={(workdir / 'profile').as_uri()}"
        assert workdir != source.parent
        workdirs.append(workdir)
        (workdir / f"{source.stem}.pdf").write_bytes(b"converted PDF")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(conversion.subprocess, "run", run)
    conversion.convert_docx_to_pdf(source, output)
    assert output.read_bytes() == b"converted PDF"
    assert source.read_bytes() == b"fake DOCX"
    assert all(not directory.exists() for directory in workdirs)


@pytest.mark.parametrize("failure", ["timeout", "exit", "missing", "empty"])
def test_libreoffice_failure_cleans_up_and_preserves_destination(source, tmp_path, monkeypatch, failure):
    monkeypatch.setattr(conversion, "find_libreoffice", lambda: "soffice")
    output = tmp_path / "converted.pdf"
    output.write_bytes(b"existing PDF")
    workdirs = []

    def run(command, **kwargs):
        workdir = Path(command[command.index("--outdir") + 1])
        workdirs.append(workdir)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 120)
        if failure == "exit":
            raise subprocess.CalledProcessError(1, command)
        if failure == "empty":
            (workdir / f"{source.stem}.pdf").touch()
        return subprocess.CompletedProcess(command, 0, b"", b"cannot load document")

    monkeypatch.setattr(conversion.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="LibreOffice"):
        conversion.convert_docx_to_pdf(source, output)
    assert output.read_bytes() == b"existing PDF"
    assert all(not directory.exists() for directory in workdirs)


def test_linux_without_libreoffice_has_actionable_error(source, tmp_path, monkeypatch):
    monkeypatch.setattr(conversion.sys, "platform", "linux")
    monkeypatch.setattr(conversion, "find_libreoffice", lambda: None)
    monkeypatch.setitem(sys.modules, "docx2pdf", None)
    with pytest.raises(RuntimeError, match="install LibreOffice Writer"):
        conversion.convert_docx_to_pdf(source, tmp_path / "out.pdf")


@pytest.mark.parametrize("platform", ["win32", "darwin"])
@pytest.mark.parametrize("fails", [False, True])
def test_word_backend_preserves_session_and_releases_com(source, tmp_path, monkeypatch, platform, fails):
    monkeypatch.setattr(conversion.sys, "platform", platform)
    monkeypatch.setattr(conversion, "find_libreoffice", lambda: None)
    events = []

    def convert(*args, **kwargs):
        events.append("convert")
        assert args == (str(source.resolve()), str((tmp_path / "out.pdf").resolve()))
        assert kwargs == {"keep_active": True}
        if fails:
            raise RuntimeError("Word failed")

    monkeypatch.setitem(sys.modules, "docx2pdf", SimpleNamespace(convert=convert))
    monkeypatch.setitem(sys.modules, "pythoncom", SimpleNamespace(
        CoInitialize=lambda: events.append("initialize"),
        CoUninitialize=lambda: events.append("uninitialize"),
    ))
    if fails:
        with pytest.raises(RuntimeError, match="Word failed"):
            conversion.convert_docx_to_pdf(source, tmp_path / "out.pdf")
    else:
        conversion.convert_docx_to_pdf(source, tmp_path / "out.pdf")
    assert events == (["initialize", "convert", "uninitialize"] if platform == "win32" else ["convert"])


def test_find_libreoffice_on_path(monkeypatch):
    monkeypatch.setattr(conversion.shutil, "which", lambda name: "/bin/soffice" if name == "soffice" else None)
    assert conversion.find_libreoffice() == "/bin/soffice"


def test_find_libreoffice_in_windows_program_files(tmp_path, monkeypatch):
    executable = tmp_path / "Program Files" / "LibreOffice" / "program" / "soffice.com"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(conversion.shutil, "which", lambda name: None)
    monkeypatch.setattr(conversion.sys, "platform", "win32")
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "Program Files"))
    assert conversion.find_libreoffice() == str(executable)


def test_non_docx_formats_do_not_require_word(tmp_path, monkeypatch):
    from fileparser import FileParser

    monkeypatch.setitem(sys.modules, "docx2pdf", None)
    path = tmp_path / "café.txt"
    path.write_text("Café text from Linux or Windows.\n", encoding="utf-8")
    pages, metadata = FileParser().parse_file(path)
    assert "Café" in pages[0][1]
    assert metadata["filename"] == "café.txt"


def test_libreoffice_real_docx(tmp_path):
    """Exercise the installed converter in CI; local machines may lack Office."""
    import os
    from zipfile import ZipFile
    from fileparser import FileParser

    if conversion.find_libreoffice() is None:
        if os.environ.get("REQUIRE_LIBREOFFICE") == "1":
            pytest.fail("CI requires an installed LibreOffice converter")
        pytest.skip("LibreOffice is not installed")
    source = tmp_path / "café report.docx"
    with ZipFile(source, "w") as archive:
        archive.writestr("[Content_Types].xml", '''<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
          <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
          <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
        </Types>''')
        archive.writestr("_rels/.rels", '''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
        </Relationships>''')
        archive.writestr("word/document.xml", '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
          <w:p><w:r><w:t>Alpha works with Beta.</w:t></w:r></w:p>
          <w:p><w:r><w:br w:type="page"/></w:r></w:p>
          <w:p><w:r><w:t>Gamma works with Delta.</w:t></w:r></w:p>
          <w:sectPr><w:pgMar w:top="1440" w:bottom="1440" w:left="1440" w:right="1440"/></w:sectPr>
        </w:body></w:document>''')
    pages, metadata = FileParser().parse_file(source)
    assert len(pages) == 2
    assert pages[0][0] == 0 and "Alpha" in pages[0][1]
    assert pages[1][0] == 1 and "Gamma" in pages[1][1]
    assert metadata["filename"] == source.name
    assert list(tmp_path.iterdir()) == [source]

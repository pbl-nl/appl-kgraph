"""DOCX-to-PDF conversion without requiring Microsoft Word on Linux."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def find_libreoffice() -> str | None:
    """Find a console executable on PATH or in a standard desktop installation."""
    for name in ("soffice.com", "soffice", "libreoffice"):
        executable = shutil.which(name)
        if executable:
            return executable
    candidates = []
    if sys.platform == "win32":
        for variable in ("ProgramFiles", "ProgramFiles(x86)"):
            if root := os.environ.get(variable):
                candidates.append(Path(root) / "LibreOffice" / "program" / "soffice.com")
    elif sys.platform == "darwin":
        candidates.append(Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"))
    return next((str(path) for path in candidates if path.is_file()), None)


def convert_docx_to_pdf(source: Path, destination: Path) -> None:
    """Convert using LibreOffice when available, otherwise Word on Windows/macOS.

    The caller owns the temporary destination directory. Each LibreOffice process
    gets its own profile so concurrent conversions cannot attach to a desktop session.
    """
    source, destination = source.resolve(), destination.resolve()
    executable = find_libreoffice()
    if executable:
        with tempfile.TemporaryDirectory(prefix="appl-kgraph-office-") as directory:
            workdir = Path(directory)
            command = [
                executable,
                f"-env:UserInstallation={(workdir / 'profile').as_uri()}",
                "--headless",
                "--convert-to", "pdf:writer_pdf_Export",
                "--outdir", str(workdir),
                str(source),
            ]
            try:
                result = subprocess.run(command, capture_output=True, timeout=120, check=True)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("LibreOffice DOCX conversion timed out after 120 seconds") from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"LibreOffice DOCX conversion failed (exit {exc.returncode})") from exc
            converted = workdir / f"{source.stem}.pdf"
            # LibreOffice can exit successfully without producing a PDF.
            if not converted.is_file() or converted.stat().st_size == 0:
                detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"LibreOffice did not produce a PDF for {source.name}: {detail}")
            shutil.move(str(converted), str(destination))
        return

    if sys.platform not in {"win32", "darwin"}:
        raise RuntimeError("DOCX conversion requires LibreOffice; install LibreOffice Writer and add soffice to PATH")

    # Import only for the Word backend: other formats and Linux need no docx2pdf.
    from docx2pdf import convert

    if sys.platform == "win32":
        import pythoncom

        # Gradio invokes synchronous callbacks in worker threads, which need COM.
        pythoncom.CoInitialize()
        try:
            convert(str(source), str(destination), keep_active=True)
        finally:
            pythoncom.CoUninitialize()
    else:
        convert(str(source), str(destination), keep_active=True)

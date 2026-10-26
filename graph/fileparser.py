import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import fitz
import logging
import mimetypes
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import TextLoader
from settings import VALID_EXTENSIONS
import utils as ut
import settings

LOGGER = logging.getLogger("appl_kgraph.fileparser")


class FileParser:
    """
    A parser that accepts .html, .docx, .md, .pdf, and .txt files
    and returns pages and metadata in a standardized format.
    """
    
    # Can we throw this one away? Yes?
    SUPPORTED_EXTENSIONS = {ext.lower() for ext in VALID_EXTENSIONS}

    def parse_file(self, filepath: Union[str, Path]) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """
        Parse a file and return (pages, metadata) tuple.
        
        Args:
            filepath: Path to the file to parse
            
        Returns:
            Tuple containing:
            - pages: List of (page_number, content) tuples (zero-indexed)
            - metadata: Dictionary with file metadata, filename, and last modified time
            
        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file doesn't exist
            ImportError: If required dependencies are missing
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        extension = filepath.suffix.lower()
        if extension not in settings.VALID_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {extension}. "
                           f"Supported extensions: {', '.join(settings.VALID_EXTENSIONS)}")
        
        # Get basic metadata
        stat = filepath.stat()
        metadata = {
            'doc_id': str(uuid.uuid4()),
            'filename': filepath.name,
            'filepath': str(filepath.resolve()),
            'file_size': stat.st_size,
            'last_modified': stat.st_mtime,
            'created': getattr(stat, "st_birthtime", stat.st_ctime),
            'extension': extension,
            'mime_type': mimetypes.guess_type(str(filepath))[0]
        }
        
        # Parse content based on file type
        try:
            if (extension == '.txt') or (extension == '.md'):
                pages, file_metadata = self._parse_text_file(filepath)
            elif extension == '.html':
                pages, file_metadata = self._parse_html_file(filepath)
            elif extension == '.pdf':
                pages, file_metadata = self._parse_pdf_file(filepath)
            elif extension == '.docx':
                pages, file_metadata = self._parse_docx_file(filepath)
        #if any error happens return none
        except Exception as e:
            LOGGER.warning("Error parsing file %s: %s", filepath, e)
            return [], {}

        # Merge file-specific metadata
        metadata.update(file_metadata)
        
        return pages, metadata
    
    def _parse_text_file(self, filepath: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """Parse text files (.txt)"""
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        raw_text = None
        for encoding in encodings_to_try:
            try:
                raw_text = filepath.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if raw_text is None:
            loader = TextLoader(file_path=filepath, autodetect_encoding=False, encoding="utf-8")
            text = loader.load()
            raw_text = text[0].page_content
        # txt files do not have multiple pages
        pages = [(0, raw_text)]
        # extract metadata
        metadata = {"language": ut.detect_language(raw_text)}

        return pages, metadata
    
    def _parse_html_file(self, filepath: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """Parse HTML files"""
        bs_kwargs = {"features": "lxml"}  # Use lxml parser if available
        try:
            loader = BSHTMLLoader(filepath, bs_kwargs=bs_kwargs)
        except Exception:
            # Fallback to default parser if lxml not available
            LOGGER.warning("lxml parser not available, falling back to default HTML parser")
            loader = BSHTMLLoader(filepath, open_encoding='utf-8')
        data = loader.load()
        raw_text = data[0].page_content.replace('\n', '')
        pages = [(0, raw_text)]
        metadata = {"language": ut.detect_language(raw_text)}

        return pages, metadata
    
    def _parse_pdf_file(self, filepath: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """
        Parse a PDF into [(page_no, text), ...] with 0-based page numbers.
        - Block-level filtering (skip headers/footers, page numbers, numbered headings).
        - Aggregates ONE string per page (paragraphs separated by blank line).
        - Detects language from the longest page.
        """
        doc = fitz.open(filepath)
        pages: List[Tuple[int, str]] = []
        longest_text: str = ""

        try:
            for i, page in enumerate(doc.pages()):
                rect = page.rect
                header_y = rect.height * 0.07   # top 7% likely header
                footer_y = rect.height * 0.93   # bottom 7% likely footer

                paragraphs: List[str] = []
                # PyMuPDF "blocks": (x0, y0, x1, y1, text, block_no, block_type, ...)
                for block in page.get_text("blocks"):
                    # Guard against odd tuples
                    if len(block) < 7:
                        # Best-effort fallback
                        txt = (block[4] if len(block) > 4 else "") or ""
                        btype = 0
                        y0 = block[1] if len(block) > 1 else 0.0
                        y1 = block[3] if len(block) > 3 else 0.0
                    else:
                        x0, y0, x1, y1, txt, _bno, btype = block[:7]

                    # Only text blocks
                    if btype != 0:
                        continue

                    block_text = (txt or "").strip()
                    if not block_text:
                        continue

                    # --- Heuristics: skip common headers/footers & page numbers ---
                    # Page number patterns
                    if re.match(r'^\s*\d+([.\-–—]|\s+)?\s*$', block_text):
                        # e.g., "12", "12.", "12 –"
                        continue
                    if re.match(r'^\s*\d+\s*/\s*\d+\s*$', block_text):
                        # e.g., "3/12"
                        continue
                    # Header with pipe variants: "Title | 12" or "12 | Title"
                    if re.match(r'^\s*(\d+)\s*\|\s*[\w\s]+$|^[\w\s]+\s*\|\s*(\d+)\s*$', block_text):
                        continue
                    # Numbered headings like "2.1 Something"
                    if re.match(r'^\s*\d+(\.\d+)*\s+[^\n]+$', block_text) and len(block_text) <= 50:
                        # Treat as a heading — often noise for RAG retrieval
                        continue
                    # Likely header/footer by vertical position and short length
                    if (y0 <= header_y or y1 >= footer_y) and len(block_text) <= 100:
                        continue

                    # --- Clean up block text: fix hyphenation, normalize whitespace ---
                    t = block_text.replace("\r", "\n")
                    # join words split by hyphen at line end
                    t = re.sub(r'(\w)-\n(\w)', r'\1\2', t)
                    # collapse multiple newlines, then spaces/tabs
                    t = re.sub(r'\n+', '\n', t)
                    t = re.sub(r'[ \t]+', ' ', t)
                    t = t.strip()
                    if t:
                        paragraphs.append(t)

                # ONE string per page (paragraphs separated by blank line)
                page_text = "\n\n".join(paragraphs).strip()
                pages.append((i, page_text))

                if len(page_text) > len(longest_text):
                    longest_text = page_text
        finally:
            doc.close()

        metadata = {"language": ut.detect_language(longest_text)}
        return pages, metadata
    
    def _parse_docx_file(self, filepath: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """Convert outside the source folder and clean up even if parsing fails."""
        with tempfile.TemporaryDirectory(prefix="appl-kgraph-docx-") as output_dir:
            path_to_pdf = self.convert_docx_to_pdf(str(filepath), output_dir)
            return self._parse_pdf_file(Path(path_to_pdf))
        
    def convert_docx_to_pdf(self, docx_path: str, output_dir: Union[str, Path]) -> str:
        """Use Word on Windows/macOS, or headless LibreOffice on Linux.

        The caller owns the output directory and its cleanup.
        """
        source = Path(docx_path).expanduser().resolve()
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        pdf_path = destination / f"{source.stem}.pdf"
        if pdf_path.exists():
            raise FileExistsError(f"Refusing to overwrite converted PDF: {pdf_path}")

        if sys.platform in {"win32", "darwin"}:
            from docx2pdf import convert

            convert(input_path=str(source), output_path=str(pdf_path), keep_active=False)
        else:
            executable = shutil.which("libreoffice") or shutil.which("soffice")
            if executable is None:
                raise RuntimeError(
                    "DOCX parsing on Linux requires LibreOffice Writer. "
                    "Install libreoffice-writer and ensure libreoffice or soffice is on PATH."
                )
            # A separate profile prevents concurrent conversions sharing an office instance.
            with tempfile.TemporaryDirectory(prefix="appl-kgraph-office-") as profile:
                subprocess.run(
                    [
                        executable,
                        f"-env:UserInstallation={Path(profile).resolve().as_uri()}",
                        "--headless", "--convert-to", "pdf",
                        "--outdir", str(destination), str(source),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

        # LibreOffice can exit successfully without producing a file.
        if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
            raise RuntimeError(f"DOCX conversion did not produce a PDF for {source.name}")
        return str(pdf_path)

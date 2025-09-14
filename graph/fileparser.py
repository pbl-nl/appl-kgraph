import os
import re
import uuid
import fitz
import mimetypes
from pathlib import Path
from docx2pdf import convert
from typing import List, Tuple, Dict, Any, Union
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import TextLoader
import utils as ut


class FileParser:
    """
    A parser that accepts .html, .docx, .md, .pdf, and .txt files
    and returns pages and metadata in a standardized format.
    """
    
    SUPPORTED_EXTENSIONS = {'.html', '.docx', '.md', '.pdf', '.txt'}
    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS - {'.docx'} ## THIS IS REMOVED DUE TO MACOS ERROR

    def __init__(self, root: Path = None):
        if root is not None:
            self.filepaths = list(root.rglob("*"))
            self.filepaths = [f for f in self.filepaths if f.suffix in self.SUPPORTED_EXTENSIONS]

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
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {extension}. "
                           f"Supported extensions: {', '.join(self.SUPPORTED_EXTENSIONS)}")
        
        # Get basic metadata
        stat = filepath.stat()
        metadata = {
            'doc_id': str(uuid.uuid4()),
            'filename': filepath.name,
            'filepath': str(filepath.absolute()),
            'file_size': stat.st_size,
            'last_modified': stat.st_mtime,
            'created': stat.st_ctime,
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
            print(f"Error parsing file {filepath}: {e}")
            return [], {}

        # Merge file-specific metadata
        metadata.update(file_metadata)
        
        return pages, metadata
    
    def _parse_text_file(self, filepath: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """Parse text files (.txt)"""
        loader = TextLoader(file_path=filepath, autodetect_encoding=True)
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
            print("lxml parser not available, falling back to default HTML parser")
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
        """Parse DOCX files"""
        path_to_pdf = self.convert_docx_to_pdf(filepath)
        pages, metadata = self._parse_pdf_file(path_to_pdf)
        os.remove(path_to_pdf)  # Clean up the temporary PDF file
        return pages, metadata
        
    def convert_docx_to_pdf(self, docx_path: str) -> str:
        """
        converts a Word file (.docx) to a pdf file and stores the pdf file in subfolder "conversions"

        Parameters
        ----------
        docx_path : str
            path of Word file to parse

        Returns
        -------
        str
            path of output pdf file
        """
        folder, file = os.path.split(docx_path)
        pdf_file = file + '.pdf'
        pdf_path = os.path.join(folder, pdf_file)
        convert(input_path=docx_path, output_path=pdf_path, keep_active=False)

        return pdf_path

    
# Example usage and utility functions
def parse_multiple_files(filepaths: List[Union[str, Path]]) -> Dict[str, Tuple[List[Tuple[int, str]], Dict[str, Any]]]:
    """
    Parse multiple files and return results in a dictionary.
    
    Args:
        filepaths: List of file paths to parse
        
    Returns:
        Dictionary mapping filepath to (pages, metadata) tuple
    """
    parser = FileParser()
    results = {}
    
    for filepath in filepaths:
        try:
            pages, metadata = parser.parse_file(filepath)
            results[str(filepath)] = (pages, metadata)
        except Exception as e:
            results[str(filepath)] = ([], {'error': str(e)})
    
    return results



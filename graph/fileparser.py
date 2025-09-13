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
        pages = [(1, raw_text)]
        # extract metadata
        metadata = {}
        metadata['Language'] = ut.detect_language(raw_text)

        return pages, metadata
    
    def _parse_html_file(self, filepath: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """Parse HTML files"""
        loader = BSHTMLLoader(filepath, open_encoding='utf-8')
        data = loader.load()
        raw_text = data[0].page_content.replace('\n', '')
        pages = [(1, raw_text)]
        metadata = {}
        metadata['Language'] = ut.detect_language(raw_text)

        return pages, metadata
    
    def _parse_pdf_file(self, filepath: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """Parse PDF files"""
        doc = fitz.open(filepath)
        pages = []
        page_with_max_text = -1
        max_page_text_length = -1

        # for each page in pdf file
        for i, page in enumerate(doc.pages()):
            first_block_of_page = True
            prv_block_text = ""
            prv_block_is_valid = True
            prv_block_is_paragraph = False
            # obtain the blocks
            blocks = page.get_text("blocks")

            # for each block
            for block in blocks:
                # only consider text blocks
                # if block["type"] == 0:
                if block[6] == 0:
                    block_is_valid = True
                    block_is_pagenr = False
                    block_is_paragraph = False
                    # block_tag = pdf_analyzer.get_block_tag(doc_tags, i, block_id)
                    # block_text = pdf_analyzer.get_block_text(doc_tags, i, block_id)
                    block_text = block[4]

                    # block text should not represent a page header or footer
                    pattern_pagenr = r'^\s*(\d+)([.\s]*)$|^\s*(\d+)([.\s]*)$'
                    if bool(re.match(pattern_pagenr, block_text)):
                        block_is_pagenr = True
                        block_is_valid = False

                    # block text should not represent a page header or footer containing a pipe character
                    # and some text
                    pattern_pagenr = r'^\s*(\d+)\s*\|\s*([\w\s]+)$|^\s*([\w\s]+)\s*\|\s*(\d+)$'
                    if bool(re.match(pattern_pagenr, block_text)):
                        block_is_pagenr = True
                        block_is_valid = False

                    # block text should not represent any form of paragraph title
                    pattern_paragraph = r'^\d+(\.\d+)*\s*.+$'
                    if bool(re.match(pattern_paragraph, block_text)):
                        if not block_is_pagenr:
                            block_is_paragraph = True

                    # if current block is content
                    if block_is_valid and (not block_is_paragraph):
                        # and the previous block was a paragraph
                        if prv_block_is_paragraph:
                            # extend the paragraph block text with a newline character and the current block text
                            block_text = prv_block_text + "\n" + block_text
                        # but if the previous block was a content block
                        else:
                            if prv_block_is_valid and block_is_valid:
                                # extend the content block text with a whitespace character and the current block text
                                block_text = prv_block_text + " " + block_text
                        # in both cases, set the previous block text to the current block text
                        prv_block_text = block_text
                    # else if current block text is not content
                    else:
                        # and the current block is not the very first block of the page
                        if not first_block_of_page:
                            # if previous block was content
                            if prv_block_is_valid and (not prv_block_is_paragraph):
                                # add text of previous block to pages together with page number
                                pages.append((i, prv_block_text))
                                # and empty the previous block text
                                prv_block_text = ""
                            # if previous block was not relevant
                            else:
                                # just set the set the previous block text to the current block text
                                prv_block_text = block_text

                    # set previous block validity indicators to current block validity indicators
                    prv_block_is_valid = block_is_valid
                    # prv_block_is_pagenr = block_is_pagenr
                    prv_block_is_paragraph = block_is_paragraph
                    prv_block_text = block_text

                    # set first_block_of_page to False
                    first_block_of_page = False

            # end of page:
            # if previous block was content
            if prv_block_is_valid and (not prv_block_is_paragraph):
                # add text of previous block to pages together with page number
                pages.append((i, prv_block_text))

            # In case the current page not added to pages, add an empty string to pages
            if (len(pages) - 1) != i:
                pages.append((i, ""))

            # store pagenr with maximum amount of characters for language detection of document
            page_text_length = len(pages[i][1])
            if page_text_length > max_page_text_length:
                page_with_max_text = i
                max_page_text_length = page_text_length

        metadata = {}
        metadata['Language'] = ut.detect_language(pages[page_with_max_text][1])

        return pages, metadata

    
    def _parse_docx_file(self, filepath: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """Parse DOCX files"""
        path_to_pdf = self.convert_docx_to_pdf(filepath)
        pages, metadata = self.parse_pymupdf(path_to_pdf)
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



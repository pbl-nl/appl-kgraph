"""
Document processing module for loading and chunking text files
"""
import os
import uuid
import hashlib
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config
import json

class DocumentProcessor:
    """Handles document loading, chunking, and metadata management"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        self.processed_files = self._load_processed_files()
    
    def _load_processed_files(self) -> Dict[str, str]:
        """Load metadata about previously processed files"""
        if self.config.METADATA_STORAGE_PATH.exists():
            with open(self.config.METADATA_STORAGE_PATH, 'r') as f:
                return json.load(f).get('processed_files', {})
        return {}
    
    def _save_processed_files(self):
        """Save metadata about processed files"""
        metadata = {
            'processed_files': self.processed_files,
            'last_updated': pd.Timestamp.now().isoformat()
        }
        with open(self.config.METADATA_STORAGE_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file content for change detection"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return hashlib.md5(content.encode()).hexdigest()
    
    def _has_file_changed(self, file_path: Path) -> bool:
        """Check if file has changed since last processing"""
        current_hash = self._get_file_hash(file_path)
        stored_hash = self.processed_files.get(str(file_path))
        return current_hash != stored_hash
    
    def load_documents(self, force_reload: bool = False) -> List[Document]:
        """
        Load all text documents from the input directory
        
        Args:
            force_reload: If True, reload all files regardless of changes
            
        Returns:
            List of Document objects
        """
        documents = []
        input_dir = self.config.DATA_INPUT_DIR
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory {input_dir} does not exist")
        
        # Get all .txt files
        txt_files = list(input_dir.glob("*.txt"))
        
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {input_dir}")
        
        for file_path in txt_files:
            # Check if file needs processing
            if not force_reload and not self._has_file_changed(file_path):
                print(f"Skipping {file_path.name} (no changes detected)")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.strip():  # Only process non-empty files
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "file_size": len(content),
                            "processed_at": pd.Timestamp.now().isoformat()
                        }
                    )
                    documents.append(doc)
                    
                    # Update processed files tracking
                    self.processed_files[str(file_path)] = self._get_file_hash(file_path)
                    print(f"Loaded document: {file_path.name}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        # Save updated metadata
        self._save_processed_files()
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks),
                    "chunk_size": len(chunk.page_content)
                })
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def documents_to_dataframe(self, documents: List[Document]) -> pd.DataFrame:
        """
        Convert document chunks to DataFrame for processing
        
        Args:
            documents: List of Document objects
            
        Returns:
            DataFrame with document data
        """
        rows = []
        for doc in documents:
            row = {
                "text": doc.page_content,
                "chunk_id": doc.metadata.get("chunk_id", str(uuid.uuid4())),
                **doc.metadata
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def get_new_documents(self) -> List[Document]:
        """
        Get only documents that have changed since last processing
        
        Returns:
            List of new/changed Document objects
        """
        return self.load_documents(force_reload=False)
    
    def process_documents(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Full document processing pipeline
        
        Args:
            force_reload: If True, reload all files regardless of changes
            
        Returns:
            DataFrame with processed document chunks
        """
        print("Starting document processing...")
        
        # Load documents
        documents = self.load_documents(force_reload=force_reload)
        
        if not documents:
            print("No new or changed documents to process")
            return pd.DataFrame()
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Convert to DataFrame
        df = self.documents_to_dataframe(chunks)
        
        print(f"Document processing complete. Generated {len(df)} chunks.")
        return df

def main():
    """Test the document processor"""
    processor = DocumentProcessor()
    df = processor.process_documents(force_reload=True)
    print(f"Processed {len(df)} chunks")
    print(df.head())

if __name__ == "__main__":
    main()
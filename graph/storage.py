
from __future__ import annotations
import os, sqlite3
from typing import Dict, Any, Iterable, List, Optional, Sequence
from contextlib import contextmanager
from openai import AzureOpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default

class AzureEmbedder:
    def __init__(self):
        if AzureOpenAI is None:
            raise RuntimeError("openai package not installed. pip install openai")
        endpoint = _get_env("AZURE_OPENAI_ENDPOINT")
        api_key = _get_env("AZURE_OPENAI_API_KEY")
        api_version = _get_env("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        if not endpoint or not api_key:
            raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
        self.model = _get_env("AZURE_OPENAI_EMB_DEPLOYMENT_NAME")
        if not self.model:
            raise RuntimeError("Set AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT to your embeddings deployment name")
        self.client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

    def embed_texts(self, texts: Iterable[str], batch_size: int = 64) -> List[List[float]]:
        out: List[List[float]] = []
        batch: List[str] = []
        for t in texts:
            batch.append(t)
            if len(batch) >= batch_size:
                resp = self.client.embeddings.create(input=batch, model=self.model)
                out.extend([d.embedding for d in resp.data])
                batch = []
        if batch:
            resp = self.client.embeddings.create(input=batch, model=self.model)
            out.extend([d.embedding for d in resp.data])
        return out

class SQLiteKV:
    def __init__(self, db_path: str = "store.sqlite"):
        self.db_path = db_path
        self._ensure_dir()

    def _ensure_dir(self):
        d = os.path.dirname(os.path.abspath(self.db_path))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    @contextmanager
    def conn(self):
        con = sqlite3.connect(self.db_path)
        try:
            yield con
        finally:
            con.close()

    def init(self):
        with self.conn() as con:
            cur = con.cursor()
            cur.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT,
                filepath TEXT,
                file_size INTEGER,
                last_modified REAL,
                created REAL,
                extension TEXT,
                mime_type TEXT,
                language TEXT,
                full_text TEXT,
                full_char_count INTEGER
            );''')
            cur.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_uuid TEXT PRIMARY KEY,
                doc_id TEXT,
                chunk_id INTEGER,
                filename TEXT,
                text TEXT,
                char_count INTEGER,
                start_page INTEGER,
                end_page INTEGER,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );''')
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_chunk ON chunks(doc_id, chunk_id);")
            con.commit()

    def upsert_document(self, metadata: Dict[str, Any], full_text: str):
        full_char_count = len(full_text or "")
        row = (
            metadata["doc_id"],
            metadata.get("filename"),
            metadata.get("filepath"),
            int(metadata.get("file_size") or 0),
            float(metadata.get("last_modified") or 0.0),
            float(metadata.get("created") or 0.0),
            metadata.get("extension"),
            metadata.get("mime_type"),
            metadata.get("Language") or metadata.get("language"),
            full_text,
            full_char_count
        )
        with self.conn() as con:
            cur = con.cursor()
            cur.execute('''
            INSERT INTO documents (doc_id, filename, filepath, file_size, last_modified, created, extension, mime_type, language, full_text, full_char_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(doc_id) DO UPDATE SET
              filename=excluded.filename,
              filepath=excluded.filepath,
              file_size=excluded.file_size,
              last_modified=excluded.last_modified,
              created=excluded.created,
              extension=excluded.extension,
              mime_type=excluded.mime_type,
              language=excluded.language,
              full_text=excluded.full_text,
              full_char_count=excluded.full_char_count;
            ''', row)
            con.commit()

    def upsert_chunks(self, chunks: Sequence[Dict[str, Any]]):
        rows = []
        for c in chunks:
            rows.append((
                c["chunk_uuid"],
                c.get("doc_id"),
                int(c.get("chunk_id", 0)),
                c.get("filename"),
                c.get("text"),
                int(c.get("char_count") or len(c.get("text") or "")),
                c.get("start_page"),
                c.get("end_page")
            ))
        with self.conn() as con:
            cur = con.cursor()
            cur.executemany('''
            INSERT INTO chunks (chunk_uuid, doc_id, chunk_id, filename, text, char_count, start_page, end_page)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(chunk_uuid) DO UPDATE SET
              doc_id=excluded.doc_id,
              chunk_id=excluded.chunk_id,
              filename=excluded.filename,
              text=excluded.text,
              char_count=excluded.char_count,
              start_page=excluded.start_page,
              end_page=excluded.end_page;
            ''', rows)
            con.commit()

class ChromaVectors:
    def __init__(self, collection: str = "chunks", chroma_dir: Optional[str] = None, embedder: Optional[AzureEmbedder] = None):
        if chromadb is None:
            raise RuntimeError("chromadb not installed. pip install chromadb")
        self.client = chromadb.PersistentClient(path=chroma_dir or _get_env("CHROMA_DIR", "./chroma_db"))
        self.col = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        self.embedder = embedder or AzureEmbedder()

    def add_chunks(self, chunks: Sequence[Dict[str, Any]]):
        ids = [c["chunk_uuid"] for c in chunks]
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        metadatas = [{
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
            "filename": c.get("filename"),
            "start_page": c.get("start_page"),
            "end_page": c.get("end_page"),
            "char_count": c.get("char_count"),
        } for c in chunks]
        self.col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

class Storage:
    def __init__(self, db_path: str = "store.sqlite", chroma_dir: Optional[str] = None, collection: str = "chunks"):
        self.kv = SQLiteKV(db_path=db_path)
        self._collection = collection
        self._chroma_dir = chroma_dir
        self._vectors = None

    def init(self):
        self.kv.init()

    def upsert_document(self, metadata: Dict[str, Any], full_text: str):
        self.kv.upsert_document(metadata, full_text)

    def upsert_chunks(self, chunks: Sequence[Dict[str, Any]]):
        self.kv.upsert_chunks(chunks)

    def add_chunks_to_vector(self, chunks: Sequence[Dict[str, Any]]):
        if self._vectors is None:
            self._vectors = ChromaVectors(collection=self._collection, chroma_dir=self._chroma_dir)
        self._vectors.add_chunks(chunks)

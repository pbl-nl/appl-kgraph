
from __future__ import annotations
import os
import sqlite3
from typing import Dict, Any, Iterable, List, Optional, Sequence, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import AzureOpenAI
import chromadb

load_dotenv()


# ---------------------------
# Helpers
# ---------------------------

def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default


def _ensure_dir_for(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _normalize_pair(a: str, b: str) -> Tuple[str, str]:
    """Return a canonical ordering for an undirected pair."""
    return (a, b) if a <= b else (b, a)


# ---------------------------
# Embeddings
# ---------------------------

class AzureEmbedder:
    """
    Thin wrapper around Azure OpenAI embeddings.
    Expects environment variables:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_API_VERSION (optional, default '2024-02-15-preview')
      - AZURE_OPENAI_EMB_DEPLOYMENT_NAME  (embedding deployment name)
    """
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
            raise RuntimeError("Set AZURE_OPENAI_EMB_DEPLOYMENT_NAME to your embeddings deployment name")
        self.client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

    def embed_texts(self, texts: Iterable[str], batch_size: int = 64) -> List[List[float]]:
        out: List[List[float]] = []
        batch: List[str] = []
        for t in texts:
            batch.append(t)
            if len(batch) >= batch_size:
                resp = self.client.embeddings.create(input=batch, model=self.model)  # type: ignore[attr-defined]
                out.extend([d.embedding for d in resp.data])
                batch = []
        if batch:
            resp = self.client.embeddings.create(input=batch, model=self.model)  # type: ignore[attr-defined]
            out.extend([d.embedding for d in resp.data])
        return out


# ---------------------------
# SQLite Schema DBs
# ---------------------------

class DocumentsDB:
    """
    Separate SQLite database for documents schema.
    Preserves table structure/metadata from the former 'documents' table.
    """
    def __init__(self, db_path: str = "documents.sqlite"):
        self.db_path = db_path
        _ensure_dir_for(self.db_path)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("PRAGMA foreign_keys = ON;")
            yield con
        finally:
            con.close()

    def init(self):
        with self.connect() as con:
            cur = con.cursor()
            # Same structure as before
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
            con.commit()

    def add_document(self, metadata: Dict[str, Any], full_text: str) -> None:
        """
        Insert-only. Will IGNORE if doc_id already exists.
        """
        doc_id = metadata.get("doc_id")
        if not doc_id:
            raise ValueError("metadata['doc_id'] is required")

        row = (
            doc_id,
            metadata.get("filename"),
            metadata.get("filepath"),
            metadata.get("file_size"),
            metadata.get("last_modified"),
            metadata.get("created"),
            metadata.get("extension"),
            metadata.get("mime_type"),
            metadata.get("language"),
            full_text,
            metadata.get("full_char_count", len(full_text) if full_text else 0),
        )
        with self.connect() as con:
            con.execute('''
                INSERT OR IGNORE INTO documents
                (doc_id, filename, filepath, file_size, last_modified, created, extension, mime_type, language, full_text, full_char_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            ''', row)
            con.commit()


class ChunksDB:
    """
    Separate SQLite database for chunks schema.
    Preserves table structure/metadata from the former 'chunks' table.
    Note: cross-database foreign keys cannot be enforced by SQLite,
    so we keep 'doc_id' but cannot reference documents(doc_id) here.
    """
    def __init__(self, db_path: str = "chunks.sqlite"):
        self.db_path = db_path
        _ensure_dir_for(self.db_path)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            # No FK across DBs
            yield con
        finally:
            con.close()

    def init(self):
        with self.connect() as con:
            cur = con.cursor()
            cur.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_uuid TEXT PRIMARY KEY,
                doc_id TEXT,
                chunk_id INTEGER,
                filename TEXT,
                text TEXT,
                char_count INTEGER,
                start_page INTEGER,
                end_page INTEGER
            );''')
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_chunk ON chunks(doc_id, chunk_id);")
            con.commit()

    def add_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """
        Insert-only. Will IGNORE duplicate chunk_uuid or duplicate (doc_id, chunk_id).
        Each chunk dict should include:
          - chunk_uuid (str, required)
          - doc_id (str, required)
          - chunk_id (int, required)
          - filename (str)
          - text (str)
          - char_count (int)
          - start_page (int)
          - end_page (int)
        """
        rows = []
        for c in chunks:
            if not c.get("chunk_uuid"):
                raise ValueError("chunk_uuid is required in chunk")
            if c.get("doc_id") is None:
                raise ValueError("doc_id is required in chunk")
            if c.get("chunk_id") is None:
                raise ValueError("chunk_id is required in chunk")
            rows.append((
                c["chunk_uuid"],
                c["doc_id"],
                int(c["chunk_id"]),
                c.get("filename"),
                c.get("text"),
                c.get("char_count"),
                c.get("start_page"),
                c.get("end_page"),
            ))
        with self.connect() as con:
            con.executemany('''
                INSERT OR IGNORE INTO chunks
                (chunk_uuid, doc_id, chunk_id, filename, text, char_count, start_page, end_page)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            ''', rows)
            con.commit()


class GraphDB:
    """
    Separate SQLite database for the knowledge graph schema (nodes & edges).
    - Nodes unique on (name, type)
    - Edges unique on undirected pair (source_name, target_name), stored canonically as (u_source_name, u_target_name)
    """
    def __init__(self, db_path: str = "graph.sqlite"):
        self.db_path = db_path
        _ensure_dir_for(self.db_path)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            yield con
        finally:
            con.close()

    def init(self):
        with self.connect() as con:
            cur = con.cursor()
            cur.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                source_id TEXT,
                filepath TEXT,
                UNIQUE(name, type)
            );''')

            # Store canonicalized pair to enforce undirected uniqueness
            cur.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                target_name TEXT NOT NULL,
                weight REAL,
                description TEXT,
                keywords TEXT,
                source_id TEXT,
                filepath TEXT,
                u_source_name TEXT NOT NULL,
                u_target_name TEXT NOT NULL,
                UNIQUE(u_source_name, u_target_name)
            );''')
            con.commit()

    def add_node(self, name: str, type: str, description: Optional[str] = None,
                 source_id: Optional[str] = None, filepath: Optional[str] = None) -> None:
        with self.connect() as con:
            con.execute('''
                INSERT OR IGNORE INTO nodes (name, type, description, source_id, filepath)
                VALUES (?, ?, ?, ?, ?);
            ''', (name, type, description, source_id, filepath))
            con.commit()

    def add_edge(self, source_name: str, target_name: str, weight: Optional[float] = None,
                 description: Optional[str] = None, keywords: Optional[str] = None,
                 source_id: Optional[str] = None, filepath: Optional[str] = None) -> None:
        u_src, u_tgt = _normalize_pair(source_name, target_name)
        with self.connect() as con:
            con.execute('''
                INSERT OR IGNORE INTO edges
                (source_name, target_name, weight, description, keywords, source_id, filepath, u_source_name, u_target_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            ''', (source_name, target_name, weight, description, keywords, source_id, filepath, u_src, u_tgt))
            con.commit()


# ---------------------------
# Chroma Vector DBs
# ---------------------------

class _ChromaBase:
    def __init__(self, collection: str, chroma_dir: str, embedder: Optional[AzureEmbedder] = None):
        if chromadb is None:
            raise RuntimeError("chromadb not installed. pip install chromadb")
        _ensure_dir_for(os.path.join(chroma_dir, ".sentinel"))
        self.client = chromadb.PersistentClient(path=chroma_dir)  # type: ignore
        self.col = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        self.embedder = embedder or AzureEmbedder()

    def _add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        embeddings = self.embedder.embed_texts(texts)
        self.col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


class ChunkVectors(_ChromaBase):
    """
    Vector DB for chunks. Mirrors previous metadata.
    metadatas:
      - doc_id, chunk_id, filename, start_page, end_page, char_count
    """
    def add_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        ids = [c["chunk_uuid"] for c in chunks]
        texts = [c.get("text", "") for c in chunks]
        metadatas = [{
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
            "filename": c.get("filename"),
            "start_page": c.get("start_page"),
            "end_page": c.get("end_page"),
            "char_count": c.get("char_count"),
        } for c in chunks]
        self._add(ids, texts, metadatas)


class EntityVectors(_ChromaBase):
    """
    Vector DB for entities.
    Uniqueness: (name, type) -> id "name::type"
    metadatas:
      - name, type, description, source_id, filepath
    """
    @staticmethod
    def _entity_id(name: str, type: str) -> str:
        return f"{name}::{type}"

    def add_entities(self, entities: Sequence[Dict[str, Any]]) -> None:
        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for e in entities:
            name = e["name"]
            etype = e["type"]
            desc = e.get("description", "") or ""
            ids.append(self._entity_id(name, etype))
            # Embed name + type + description
            texts.append(f"{name} [{etype}] {desc}")
            metas.append({
                "name": name,
                "type": etype,
                "description": desc,
                "source_id": e.get("source_id"),
                "filepath": e.get("filepath"),
            })
        self._add(ids, texts, metas)


class RelationVectors(_ChromaBase):
    """
    Vector DB for relations.
    Uniqueness: undirected (source_name, target_name) -> id "min::max"
    metadatas:
      - source_name, target_name, weight, description, keywords, source_id, filepath
    """
    @staticmethod
    def _edge_id(a: str, b: str) -> str:
        x, y = _normalize_pair(a, b)
        return f"{x}::{y}"

    def add_relations(self, relations: Sequence[Dict[str, Any]]) -> None:
        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for r in relations:
            src = r["source_name"]
            tgt = r["target_name"]
            desc = r.get("description", "") or ""
            kw = r.get("keywords", "") or ""
            ids.append(self._edge_id(src, tgt))
            # Embed src + tgt + description + keywords
            texts.append(f"{src} <-> {tgt} :: {desc} :: {kw}")
            metas.append({
                "source_name": src,
                "target_name": tgt,
                "weight": r.get("weight"),
                "description": desc,
                "keywords": kw,
                "source_id": r.get("source_id"),
                "filepath": r.get("filepath"),
            })
        self._add(ids, texts, metas)


# ---------------------------
# Unified Storage Facade
# ---------------------------

@dataclass
class StoragePaths:
    documents_db: str = "./storage/documents.sqlite"
    chunks_db: str = "./storage/chunks.sqlite"
    graph_db: str = "./storage/graph.sqlite"
    chroma_chunks: str = "./storage/chroma_chunks"
    chroma_entities: str = "./storage/chroma_entities"
    chroma_relations: str = "./storage/chroma_relations"


class Storage:
    """
    Storage facade that wires together SIX separate databases:
      1 - Documents schema (SQLite)
      2 - Chunks schema (SQLite)
      3 - Chunk vectors (Chroma)
      4 - Knowledge Graph schema: nodes & edges (SQLite)
      5 - Entity vectors (Chroma)
      6 - Relation vectors (Chroma)

    Only 'add' operations are implemented for now.
    """

    def __init__(self, paths: Optional[StoragePaths] = None, embedder: Optional[AzureEmbedder] = None):
        paths = paths or StoragePaths()

        # Schema DBs
        self.documents = DocumentsDB(paths.documents_db)
        self.chunks = ChunksDB(paths.chunks_db)
        self.graph = GraphDB(paths.graph_db)

        # Vector DBs (each in its own persistent directory => separate DBs)
        self.chunk_vectors = ChunkVectors(collection="chunks", chroma_dir=paths.chroma_chunks, embedder=embedder)
        self.entity_vectors = EntityVectors(collection="entities", chroma_dir=paths.chroma_entities, embedder=embedder)
        self.relation_vectors = RelationVectors(collection="relations", chroma_dir=paths.chroma_relations, embedder=embedder)

    def init(self):
        """Create tables/collections if they don't exist yet."""
        self.documents.init()
        self.chunks.init()
        self.graph.init()

    # ---------- Add-only APIs ----------

    # 1) Documents schema
    def add_document(self, metadata: Dict[str, Any], full_text: str) -> None:
        self.documents.add_document(metadata, full_text)

    # 2) Chunks schema
    def add_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        self.chunks.add_chunks(chunks)

    # 3) Chunk vectors
    def add_chunk_vectors(self, chunks: Sequence[Dict[str, Any]]) -> None:
        self.chunk_vectors.add_chunks(chunks)

    # 4) Knowledge Graph schema
    def add_kg_node(self, name: str, type: str, description: Optional[str] = None,
                    source_id: Optional[str] = None, filepath: Optional[str] = None) -> None:
        self.graph.add_node(name=name, type=type, description=description, source_id=source_id, filepath=filepath)

    def add_kg_edge(self, source_name: str, target_name: str, weight: Optional[float] = None,
                    description: Optional[str] = None, keywords: Optional[str] = None,
                    source_id: Optional[str] = None, filepath: Optional[str] = None) -> None:
        self.graph.add_edge(source_name=source_name, target_name=target_name, weight=weight,
                            description=description, keywords=keywords, source_id=source_id, filepath=filepath)

    # 5) Entity vectors
    def add_entity_vectors(self, entities: Sequence[Dict[str, Any]]) -> None:
        """
        Each entity dict must include: name, type; optional: description, source_id, filepath
        Uniqueness enforced via Chroma IDs "name::type".
        """
        if entities:
            self.entity_vectors.add_entities(entities)

    # 6) Relation vectors
    def add_relation_vectors(self, relations: Sequence[Dict[str, Any]]) -> None:
        """
        Each relation dict must include: source_name, target_name;
        optional: weight, description, keywords, source_id, filepath.
        Uniqueness enforced via normalized ID "min::max".
        """
        if relations:
            self.relation_vectors.add_relations(relations)

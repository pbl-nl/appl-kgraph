from __future__ import annotations
import os
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

import chromadb
import openai
from openai import AzureOpenAI


load_dotenv()


# --- Options & ID helpers ---
DELIM = "::"
ID_SAFE = re.compile(r"[^A-Za-z0-9_\-.]")  # deliberately NO colon to protect the DELIM

@dataclass
class StorageOptions:
    directed_relations: bool = True  # True: A->B ≠ B->A; False: undirected (canonical)
    id_delimiter: str = DELIM

def _normalize_token(s: str) -> str:
    # fold unicode to ASCII, drop unsafe chars, lowercase
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return ID_SAFE.sub("_", s).lower()

def build_entity_mention_id(name: str, typ: str, doc_id: str, chunk_id: str, mention_idx: int, *, delim: str = DELIM) -> str:
    parts = map(_normalize_token, (name, typ, doc_id, chunk_id, str(mention_idx)))
    return "ent" + delim + delim.join(parts)

def _edge_key(src: str, dst: str, *, directed: bool, delim: str = DELIM) -> str:
    # canonical edge key for grouping/graph-level identity
    a, b = (_normalize_token(src), _normalize_token(dst))
    return (a + delim + b) if directed else delim.join(sorted((a, b)))

def build_edge_mention_id(src: str, dst: str, doc_id: str, chunk_id: str, rel_idx: int, *, directed: bool, delim: str = DELIM) -> str:
    # keep direction in the mention id when directed; canonicalize when undirected
    if directed:
        parts = map(_normalize_token, (src, dst, doc_id, chunk_id, str(rel_idx)))
    else:
        # canonicalize endpoints for undirected mode
        s, t = sorted((_normalize_token(src), _normalize_token(dst)))
        parts = (s, t, _normalize_token(doc_id), _normalize_token(chunk_id), str(rel_idx))
    return "rel" + delim + delim.join(parts)

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
# ---------------------------
# Embeddings
# ---------------------------

class LLMEmbedder:
    """
    Thin wrapper around Azure OpenAI embeddings.
    Expects environment variables:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_API_VERSION (optional, default '2024-02-15-preview')
      - AZURE_OPENAI_EMB_DEPLOYMENT_NAME  (embedding deployment name)
    """
    def __init__(self):
        use_azure = os.getenv("USE_AZURE_OPENAI", "true").lower() == "true"
        if use_azure:
            if AzureOpenAI is None:  # type: ignore[truthy-bool]
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
        else:
            api_key = _get_env("OPENAI_API_KEY")
            model = _get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
            self.model = model
            self.client = openai.OpenAI(api_key=api_key)
    
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
        self.KEYS = [
            "doc_id",
            "filename",
            "filepath",
            "file_size",
            "last_modified",
            "created",
            "extension",
            "mime_type",
            "language",
            "full_text",
            "full_char_count",
        ]

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            # Safety + speed
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("PRAGMA temp_store=MEMORY;")
            con.execute("PRAGMA cache_size=-20000;")   # ~20MB cache
            con.execute("PRAGMA busy_timeout=5000;")   # wait for locks
            con.execute("PRAGMA foreign_keys = ON;")   # Enable foreign key constraints
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

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM documents WHERE doc_id = ?;", (doc_id,))
            row = cur.fetchone()
            if row:
                return dict(zip(self.KEYS, row))
            return None

    def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        with self.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM documents WHERE filename = ?;", (filename,))
            row = cur.fetchone()
            if row:
                return dict(zip(self.KEYS, row))
            return None
        
    def list_documents(self) -> List[Dict[str, Any]]:
        with self.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM documents;")
            rows = cur.fetchall()
            documents = []
            for row in rows:
                documents.append(dict(zip(self.KEYS, row)))
            return documents

    def delete_document(self, doc_id: str) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM documents WHERE doc_id = ?;", (doc_id,))
            con.commit()

    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        allowed_keys = set(self.KEYS) - {"doc_id"}
        set_clauses = []
        values = []
        for k, v in updates.items():
            if k in allowed_keys:
                set_clauses.append(f"{k} = ?")
                values.append(v)
        if not set_clauses:
            return
        values.append(doc_id)
        set_clause = ", ".join(set_clauses)
        with self.connect() as con:
            con.execute(f"UPDATE documents SET {set_clause} WHERE doc_id = ?;", values)
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
        self.KEYS = [
            "chunk_uuid",
            "doc_id",
            "chunk_id",
            "filename",
            "text",
            "char_count",
            "start_page",
            "end_page"
        ]

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            # Safety + speed
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("PRAGMA temp_store=MEMORY;")
            con.execute("PRAGMA cache_size=-20000;")   # ~20MB cache
            con.execute("PRAGMA busy_timeout=5000;")   # wait for locks
            con.execute("PRAGMA foreign_keys = ON;")   # Enable foreign key constraints
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

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        with self.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM chunks WHERE doc_id = ?;", (doc_id,))
            rows = cur.fetchall()
        return [dict(zip(self.KEYS, row)) for row in rows]

    def get_chunks_by_filename(self, filename: str) -> List[Dict[str, Any]]:
        with self.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM chunks WHERE filename = ?;", (filename,))
            rows = cur.fetchall()
        return [dict(zip(self.KEYS, row)) for row in rows]

    def get_chunk_by_uuid(self, chunk_uuid: str) -> List[Dict[str, Any]]:
        with self.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM chunks WHERE chunk_uuid = ?;", (chunk_uuid,))
            rows = cur.fetchall()
        return [dict(zip(self.KEYS, row)) for row in rows]

    def get_chunks_by_uuids(self, chunk_uuids: List[str]) -> List[Dict[str, Any]]:
        if not chunk_uuids:
            return []
        placeholders = ', '.join('?' for _ in chunk_uuids)
        with self.connect() as con:
            cur = con.cursor()
            cur.execute(f"SELECT * FROM chunks WHERE chunk_uuid IN ({placeholders});", chunk_uuids)
            rows = cur.fetchall()
        return [dict(zip(self.KEYS, row)) for row in rows]
    
    def delete_chunks_by_doc_id(self, doc_id: str) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM chunks WHERE doc_id = ?;", (doc_id,))
            con.commit()

    def delete_chunk_by_uuid(self, chunk_uuid: str) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM chunks WHERE chunk_uuid = ?;", (chunk_uuid,))
            con.commit()

    def delete_chunks_by_uuids(self, chunk_uuids: List[str]) -> None:
        if not chunk_uuids:
            return
        placeholders = ', '.join('?' for _ in chunk_uuids)
        with self.connect() as con:
            con.execute(f"DELETE FROM chunks WHERE chunk_uuid IN ({placeholders});", chunk_uuids)
            con.commit()

    def update_chunk(self, chunk_uuid: str, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        allowed_keys = set(self.KEYS) - {"chunk_uuid"}
        set_clauses = []
        values = []
        for k, v in updates.items():
            if k in allowed_keys:
                set_clauses.append(f"{k} = ?")
                values.append(v)
        if not set_clauses:
            return
        values.append(chunk_uuid)
        set_clause = ", ".join(set_clauses)
        with self.connect() as con:
            con.execute(f"UPDATE chunks SET {set_clause} WHERE chunk_uuid = ?;", values)
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
        self.KEYS_NODE = ["id", "name", "type", "description", "source_id", "filepath"]
        self.KEYS_EDGE = ["id", "source_name", "target_name", "weight", "description", "keywords", "source_id", "filepath", "u_source_name", "u_target_name"]

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            # Safety + speed
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("PRAGMA temp_store=MEMORY;")
            con.execute("PRAGMA cache_size=-20000;")   # ~20MB cache
            con.execute("PRAGMA busy_timeout=5000;")   # wait for locks
            con.execute("PRAGMA foreign_keys = ON;")   # Enable foreign key constraints
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

    # ---------------------------
    # NODE OPERATIONS
    # ---------------------------
    def add_node(self, name: str, type: str, description: Optional[str] = None,
                 source_id: Optional[str] = None, filepath: Optional[str] = None) -> None:
        with self.connect() as con:
            con.execute('''
                INSERT OR IGNORE INTO nodes (name, type, description, source_id, filepath)
                VALUES (?, ?, ?, ?, ?);
            ''', (name, type, description, source_id, filepath))
            con.commit()

    def add_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        with self.connect() as con:
            con.executemany('''
                INSERT OR IGNORE INTO nodes (name, type, description, source_id, filepath)
                VALUES (?, ?, ?, ?, ?);
            ''', [(n['name'], n['type'], n.get('description'), n.get('source_id'), n.get('filepath')) for n in nodes])
            con.commit()

    def get_node(self, name: str, type: str) -> Optional[Dict[str, Any]]:
        keys = [k for k in self.KEYS_NODE if k != "id"]
        with self.connect() as con:
            cur = con.cursor()
            cur.execute('''
                SELECT name, type, description, source_id, filepath FROM nodes WHERE name = ? AND type = ?;
            ''', (name, type))
            row = cur.fetchone()
            if row:
                return dict(zip(keys, row))
            return None

    def get_nodes(self, names: List[str], types: List[str]) -> List[Dict[str, Any]]:
        if not names or not types or len(names) != len(types):
            return []
        keys = [k for k in self.KEYS_NODE if k != "id"]
        pairs = list(zip(names, types))
        placeholders = ",".join(["(?, ?)"] * len(pairs))
        flat_params: List[Any] = []
        for n, t in pairs:
            flat_params.extend([n, t])
        with self.connect() as con:
            cur = con.cursor()
            cur.execute(f'''
                SELECT name, type, description, source_id, filepath
                FROM nodes
                WHERE (name, type) IN ({placeholders});
            ''', tuple(flat_params))
            rows = cur.fetchall()
            return [dict(zip(keys, row)) for row in rows] if rows else []

    def delete_node(self, name: str, type: str) -> None:
        with self.connect() as con:
            con.execute('''
                DELETE FROM nodes WHERE name = ? AND type = ?;
            ''', (name, type))
            con.commit()

    def delete_nodes(self, names: List[str], types: List[str]) -> None:
        if not names or not types or len(names) != len(types):
            return
        pairs = list(zip(names, types))
        with self.connect() as con:
            con.executemany('''
                DELETE FROM nodes WHERE name = ? AND type = ?;
            ''', pairs)
            con.commit()

    def update_node(self, name: str, type: str, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        allowed_keys = {"description", "source_id", "filepath"}
        set_clauses = []
        values = []
        for k, v in updates.items():
            if k in allowed_keys:
                set_clauses.append(f"{k} = ?")
                values.append(v)
        if not set_clauses:
            return
        values.append(name)
        values.append(type)
        set_clause = ", ".join(set_clauses)
        with self.connect() as con:
            con.execute(f"UPDATE nodes SET {set_clause} WHERE name = ? AND type = ?;", values)
            con.commit()

    def update_nodes(self, updates_list: List[Dict[str, Any]]) -> None:
        if not updates_list:
            return
        for update in updates_list:
            self.update_node(update["name"], update["type"], update)

    # ---------------------------
    # EDGE OPERATIONS
    # ---------------------------    
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

    def add_edges(self, edges: List[Dict[str, Any]]) -> None:
        edges = [{**e, **dict(zip(("u_source_name", "u_target_name"), _normalize_pair(e["source_name"], e["target_name"])))} for e in edges]
        with self.connect() as con:
            con.executemany('''
                INSERT OR IGNORE INTO edges
                (source_name, target_name, weight, description, keywords, source_id, filepath, u_source_name, u_target_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            ''', [(e["source_name"], e["target_name"], e.get("weight"), e.get("description"),
                    e.get("keywords"), e.get("source_id"), e.get("filepath"),
                    e["u_source_name"], e["u_target_name"]) for e in edges])
            con.commit()

    def get_edge(self, source_name: str, target_name: str) -> Optional[Dict[str, Any]]:
        u_src, u_tgt = _normalize_pair(source_name, target_name)
        keys = [k for k in self.KEYS_EDGE if k not in {"id", "u_source_name", "u_target_name"}]
        with self.connect() as con:
            cur = con.cursor()
            cur.execute('''
                SELECT source_name, target_name, weight, description, keywords, source_id, filepath
                FROM edges
                WHERE u_source_name = ? AND u_target_name = ?;
            ''', (u_src, u_tgt))
            row = cur.fetchone()
            if row:
                return dict(zip(keys, row))
            return None
        
    def get_edges(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        if not pairs:
            return []
        pairs = [_normalize_pair(a, b) for a, b in pairs]
        keys = [k for k in self.KEYS_EDGE if k not in {"id", "u_source_name", "u_target_name"}]
        placeholders = ",".join(["(?, ?)"] * len(pairs))
        flat_params: List[Any] = []
        for u_src, u_tgt in pairs:
            flat_params.extend([u_src, u_tgt])
        with self.connect() as con:
            cur = con.cursor()
            cur.execute(f'''
                SELECT source_name, target_name, weight, description, keywords, source_id, filepath
                FROM edges
                WHERE (u_source_name, u_target_name) IN ({placeholders});
            ''', tuple(flat_params))
            rows = cur.fetchall()
            return [dict(zip(keys, row)) for row in rows] if rows else []
        
    def delete_edge(self, source_name: str, target_name: str) -> None:
        u_src, u_tgt = _normalize_pair(source_name, target_name)
        with self.connect() as con:
            con.execute('''
                DELETE FROM edges WHERE u_source_name = ? AND u_target_name = ?;
            ''', (u_src, u_tgt))
            con.commit()

    def delete_edges(self, pairs: List[Tuple[str, str]]) -> None:
        if not pairs:
            return
        norm = [_normalize_pair(a, b) for a, b in pairs]
        with self.connect() as con:
            con.executemany('''
                DELETE FROM edges WHERE u_source_name = ? AND u_target_name = ?;
            ''', norm)
            con.commit()

    def update_edge(self, source_name: str, target_name: str, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        allowed_keys = {"weight", "description", "keywords", "source_id", "filepath"}
        filtered = {k: v for k, v in updates.items() if k in allowed_keys}
        if not filtered:
            return
        u_src, u_tgt = _normalize_pair(source_name, target_name)
        set_clause = ', '.join(f"{k} = ?" for k in filtered.keys())
        with self.connect() as con:
            con.execute(f'''
                UPDATE edges SET {set_clause} WHERE u_source_name = ? AND u_target_name = ?;
            ''', (*filtered.values(), u_src, u_tgt))
            con.commit()

    def update_edges(self, updates_list: List[Dict[str, Any]]) -> None:
        if not updates_list:
            return
        allowed_keys = {"weight", "description", "keywords", "source_id", "filepath"}
        with self.connect() as con:
            for u in updates_list:
                src = u["source_name"]
                tgt = u["target_name"]
                filtered = {k: v for k, v in u.items() if k in allowed_keys}
                if not filtered:
                    continue
                u_src, u_tgt = _normalize_pair(src, tgt)
                set_clause = ', '.join(f"{k} = ?" for k in filtered.keys())
                con.execute(f'''
                    UPDATE edges SET {set_clause} WHERE u_source_name = ? AND u_target_name = ?;
                ''', (*filtered.values(), u_src, u_tgt))
            con.commit()


# ---------------------------
# Chroma Vector DBs
# ---------------------------

class _ChromaBase:
    def __init__(self, collection: str, chroma_dir: str, embedder: Optional[LLMEmbedder] = None):
        if chromadb is None:
            raise RuntimeError("chromadb not installed. pip install chromadb")
        _ensure_dir_for(os.path.join(chroma_dir, ".sentinel"))
        self.client = chromadb.PersistentClient(path=chroma_dir)  # type: ignore
        self.col = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        self.embedder = embedder or LLMEmbedder()

    def _add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        embeddings = self.embedder.embed_texts(texts)
        self.col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def _get(self, ids: List[str]) -> List[Dict[str, Any]]:
        return self.col.get(ids=ids)

    def _delete(self, ids: List[str]) -> None:
        self.col.delete(ids=ids)

    def _upsert(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        embeddings = self.embedder.embed_texts(texts)
        self.col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def _query(self, texts: List[str], n_results: int, where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        embeddings = self.embedder.embed_texts(texts)
        return self.col.query(embeddings=embeddings, n_results=n_results, where=where, where_document=where_document)

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

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        return self._get(ids=ids)

    def delete(self, ids: List[str]) -> None:
        self._delete(ids=ids)

    def upsert(self, chunks: Sequence[Dict[str, Any]]) -> None:
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
        self._upsert(ids=ids, texts=texts, metadatas=metadatas)

    def query(self, text: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None, 
            where_document: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._query(texts=[text], n_results=n_results, where=where, where_document=where_document)

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

    def get_entities(self, names: List[str], types: List[str]) -> List[Dict[str, Any]]:
        ids = [self._entity_id(name, etype) for name, etype in zip(names, types)]
        return self._get(ids=ids)
    
    def delete_entities(self, names: List[str], types: List[str]) -> None:
        ids = [self._entity_id(name, etype) for name, etype in zip(names, types)]
        self._delete(ids=ids)

    def upsert_entities(self, entities: Sequence[Dict[str, Any]]) -> None:
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
        self._upsert(ids, texts, metas)

    def query(self, text: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._query(texts=[text], n_results=n_results, where=where, where_document=where_document)

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

    def get_relations(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        ids = [self._edge_id(src, tgt) for src, tgt in pairs]
        return self._get(ids=ids)

    def delete_relations(self, pairs: List[Tuple[str, str]]) -> None:
        ids = [self._edge_id(src, tgt) for src, tgt in pairs]
        self._delete(ids=ids)

    def upsert_relations(self, relations: Sequence[Dict[str, Any]]) -> None:
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
        self._upsert(ids, texts, metas)

    def query(self, text: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._query(texts=[text], n_results=n_results, where=where, where_document=where_document)

# storage.py — add after _ChromaBase and the other *Vectors classes
class MentionVectors(_ChromaBase):
    """
    Vector DB for individual mentions (entity or relation sightings in a chunk).
    We store one vector per mention with a stable mention_id.
    """
    def upsert(self, ids, texts, metas, embeddings=None):
        if embeddings is None:
            self._upsert(ids=ids, texts=texts, metadatas=metas)
        else:
            self.col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

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
    chroma_mentions: str = "./storage/chroma_mentions"


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
    def __init__(self, paths: "StoragePaths" = None, options: Optional[StorageOptions] = None):
        self.paths = paths or StoragePaths()
        self.options = options or StorageOptions()

        # Schema DBs
        self.documents = DocumentsDB(self.paths.documents_db)
        self.chunks = ChunksDB(self.paths.chunks_db)
        self.graph = GraphDB(self.paths.graph_db)

        # Vector DBs (each in its own persistent directory => separate DBs)
        embedder = LLMEmbedder() # default embedder; can be customized per *Vectors class if needed
        self.chunk_vectors    = ChunkVectors(collection="chunks",    chroma_dir=self.paths.chroma_chunks,    embedder=embedder)
        self.entity_vectors   = EntityVectors(collection="entities", chroma_dir=self.paths.chroma_entities,  embedder=embedder)
        self.relation_vectors = RelationVectors(collection="relations", chroma_dir=self.paths.chroma_relations, embedder=embedder)
        self.mention_vectors  = MentionVectors(collection="mentions", chroma_dir=self.paths.chroma_mentions, embedder=embedder)  # NEW

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

    # 3) Knowledge Graph schema
    def add_kg_node(self, name: str, type: str, description: Optional[str] = None,
                    source_id: Optional[str] = None, filepath: Optional[str] = None) -> None:
        self.graph.add_node(name=name, type=type, description=description, source_id=source_id, filepath=filepath)

    def add_kg_edge(self, source_name: str, target_name: str, weight: Optional[float] = None,
                    description: Optional[str] = None, keywords: Optional[str] = None,
                    source_id: Optional[str] = None, filepath: Optional[str] = None) -> None:
        self.graph.add_edge(source_name=source_name, target_name=target_name, weight=weight,
                            description=description, keywords=keywords, source_id=source_id, filepath=filepath)

    # 4) Chunk vectors
    def add_chunk_vectors(self, chunks: Sequence[Dict[str, Any]]) -> None:
        if chunks:
            self.chunk_vectors.add_chunks(chunks)

    # 5) Entity vectors
    # --- Vector: Entity mention ---
    def add_entity_mention_vector(
        self,
        *,
        name: str,
        typ: str,
        text: str,
        doc_id: str,
        chunk_id: str,
        mention_idx: int,
        page: Optional[int] = None,
        embedding: Optional[List[float]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        mention_id = build_entity_mention_id(name, typ, doc_id, chunk_id, mention_idx, delim=self.options.id_delimiter)
        meta: Dict[str, Any] = {
            "kind": "entity_mention",
            "entity_id": _normalize_token(name) + self.options.id_delimiter + _normalize_token(typ),
            "name": name,
            "type": typ,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "page": page,
        }
        if extra_meta:
            meta.update(extra_meta)

        self.mention_vectors.upsert(
            ids=[mention_id],
            texts=[text or ""],
            metas=[meta],
            embeddings=[embedding] if embedding is not None else None,
        )

    # --- Vector: Relation mention ---
    def add_edge_mention_vector(
        self,
        *,
        src: str,
        dst: str,
        text: str,
        doc_id: str,
        chunk_id: str,
        rel_idx: int,
        page: Optional[int] = None,
        embedding: Optional[List[float]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        directed = self.options.directed_relations
        mention_id = build_edge_mention_id(src, dst, doc_id, chunk_id, rel_idx, directed=directed, delim=self.options.id_delimiter)
        edge_id = _edge_key(src, dst, directed=directed, delim=self.options.id_delimiter)

        # normalize source/target for metadata if undirected
        if directed:
            src_meta, dst_meta = src, dst
        else:
            src_meta, dst_meta = sorted((_normalize_token(src), _normalize_token(dst)))

        meta: Dict[str, Any] = {
            "kind": "edge_mention",
            "edge_id": edge_id,
            "source": src_meta,
            "target": dst_meta,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "page": page,
        }
        if extra_meta:
            meta.update(extra_meta)

        self.mention_vectors.upsert(
            ids=[mention_id],
            texts=[text or ""],
            metas=[meta],
            embeddings=[embedding] if embedding is not None else None,
        )

    # ---------- Get-only APIs ----------

    # 1) Documents
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get_document(doc_id)
    
    def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        return self.documents.get_document_by_filename(filename)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        return self.documents.list_documents()

    # 2) Chunks
    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        return self.chunks.get_chunks_by_doc_id(doc_id)
    
    def get_chunks_by_filename(self, filename: str) -> List[Dict[str, Any]]:
        return self.chunks.get_chunks_by_filename(filename)

    def get_chunk_by_uuid(self, chunk_uuid: str) -> Optional[Dict[str, Any]]:
        return self.chunks.get_chunk_by_uuid(chunk_uuid)

    def get_chunks_by_uuids(self, chunk_uuids: List[str]) -> List[Dict[str, Any]]:
        return self.chunks.get_chunks_by_uuids(chunk_uuids)
    
    # 3) Graph
    def get_node(self, name: str, type: str):
        """Fetch a single node by (name, type)."""
        return self.graph.get_node(name, type)

    def get_nodes(self, pairs: List[Tuple[str, str]]):
        """Fetch multiple nodes; input as [(name, type), ...]."""
        if not pairs:
            return []
        names, types = zip(*pairs)
        return self.graph.get_nodes(list(names), list(types))

    def get_edge(self, source_name: str, target_name: str):
        """Fetch a single edge by (source, target)."""
        return self.graph.get_edge(source_name, target_name)

    def get_edges(self, pairs: List[Tuple[str, str]]):
        """Fetch multiple edges; input as [(source, target), ...]."""
        if not pairs:
            return []
        return self.graph.get_edges(pairs)

    # 4) Chunk Vectors
    def get_chunk_vector(self, chunk_uuid: str) -> Optional[Dict[str, Any]]:
        return self.chunk_vectors.get([chunk_uuid])

    # 5) Entity Vectors
    def get_entities(self, names: List[str], types: List[str]) -> List[Dict[str, Any]]:
        return self.entity_vectors.get_entities(names, types)

    # 6) Relation Vectors
    def get_relations(self, pairs: List[Tuple[str, str]]) -> Optional[Dict[str, Any]]:
        return self.relation_vectors.get_relations(pairs)

    # ---------- Delete-only APIs ----------

    # 1) Documents
    def delete_document(self, doc_id: str) -> None:
        self.documents.delete_document(doc_id)

    # 2) Chunks
    def delete_chunks_by_doc_id(self, doc_id: str) -> None:
        self.chunks.delete_chunks_by_doc_id(doc_id)

    def delete_chunk_by_uuid(self, chunk_uuid: str) -> None:
        self.chunks.delete_chunk_by_uuid(chunk_uuid)

    def delete_chunks_by_uuids(self, chunk_uuids: List[str]) -> None:
        self.chunks.delete_chunks_by_uuids(chunk_uuids)

    # 3) Graph
    def delete_node(self, name: str, type: str) -> None:
        self.graph.delete_node(name, type)

    def delete_nodes(self, names: List[str], types: List[str]) -> None:
        self.graph.delete_nodes(names, types)

    def delete_edge(self, source_name: str, target_name: str) -> None:
        self.graph.delete_edge(source_name, target_name)

    def delete_edges(self, pairs: List[Tuple[str, str]]) -> None:
        self.graph.delete_edges(pairs)

    # 4) Chunk Vectors
    def delete_chunk_vector(self, chunk_uuid: str) -> None:
        self.chunk_vectors.delete([chunk_uuid])

    # 5) Entity Vectors
    def delete_entity_vector(self, names: List[str], types: List[str]) -> None:
        self.entity_vectors.delete_entities(names, types)

    # 6) Relation Vectors
    def delete_relation_vector(self, pairs: List[Tuple[str, str]]) -> None:
        self.relation_vectors.delete_relations(pairs)

    # ---------- Update and Upsert APIs ----------

    # 1) Documents
    def upsert_document(self, doc_id: str, updates: Dict[str, Any]) -> None:
        self.documents.update_document(doc_id, updates)

    # 2) Chunks
    def upsert_chunk(self, chunk_uuid: str, updates: Dict[str, Any]) -> None:
        self.chunks.update_chunk(chunk_uuid, updates)

    # 3) Graph
    def upsert_node(self, name: str, type: str, updates: Dict[str, Any]) -> None:
        """
        Update a single node identified by (name, type).
        'updates' may include: description, source_id, filepath.
        """
        self.graph.update_node(name, type, updates)

    def upsert_nodes(self, updates_list: List[Dict[str, Any]]) -> None:
        """
        Batch update nodes. Each dict MUST include 'name' and 'type' keys,
        plus any updatable fields (description, source_id, filepath).
        """
        self.graph.update_nodes(updates_list)

    def upsert_edge(self, source_name: str, target_name: str, updates: Dict[str, Any]) -> None:
        """Update a single edge identified by (source, target)."""
        if not self.options.directed_relations:
            source_name, target_name = sorted((source_name, target_name))
        self.graph.update_edge(source_name, target_name, updates)

    def upsert_edges(self, updates_list: List[Dict[str, Any]]) -> None:
        """
        Batch update edges. Each dict MUST include 'source_name' and 'target_name',
        plus any updatable fields (weight, description, keywords, source_id, filepath).
        """
        self.graph.update_edges(updates_list)

    # 4) Chunk Vectors
    def upsert_chunk_vector(self, chunks: Sequence[Dict[str, Any]]) -> None:
        self.chunk_vectors.upsert(chunks)

    # 5) Entity Vectors
    def upsert_entity_vector(self, entities: List[Dict[str, Any]]) -> None:
        self.entity_vectors.upsert_entities(entities)

    # 6) Relation Vectors
    def upsert_relation_vector(self, relations: List[Dict[str, Any]]) -> None:
        self.relation_vectors.upsert_relations(relations)

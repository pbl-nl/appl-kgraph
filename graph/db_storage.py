
from __future__ import annotations
import os
import sqlite3
import json
import hashlib
from typing import Dict, Any, List, Optional, Sequence, Tuple
from contextlib import contextmanager
from dataclasses import replace
import chromadb  
from llm import Embedder
from settings import settings, StoragePaths as SettingsStoragePaths

# ---------------------------
# Helpers
# ---------------------------
def _ensure_dir_for(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _normalize_pair(a: str, b: str) -> Tuple[str, str]:
    """Return a canonical ordering for an undirected pair."""
    return (a, b) if a <= b else (b, a)

def _current_models_fingerprint(embedder) -> dict:
    """
    Build a stable fingerprint for both the embedding model and the chat model.
    Stored on each Chroma collection so we can verify at open time.
    """
    prov = settings.provider.provider  # "openai" | "azure"

    if prov == "openai":
        emb_model = settings.provider.openai_embeddings_model
        llm_model = settings.provider.openai_llm_model
        endpoint = settings.provider.openai_base_url or "https://api.openai.com/v1"
        api_version = ""  # not applicable
    else:
        emb_model = settings.provider.azure_embeddings_deployment
        llm_model = settings.provider.azure_llm_deployment
        endpoint = settings.provider.azure_endpoint
        api_version = settings.provider.azure_api_version

    fp = {
        "provider": prov,
        "embedding_model": emb_model,
        "embedding_dim": embedder.dimension,
        "llm_model": llm_model,
        "endpoint": endpoint,
        "api_version": api_version,
    }
    fp_json = json.dumps(fp, sort_keys=True)
    fp_sha = hashlib.sha256(fp_json.encode("utf-8")).hexdigest()
    fp["sha256"] = fp_sha
    fp["json"] = fp_json
    return fp

# ---------------------------


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
            "content_hash",
        ]

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
                full_char_count INTEGER,
                content_hash TEXT
            );''')
            cur.execute('''
            CREATE TABLE IF NOT EXISTS llm_cache (
                model      TEXT NOT NULL,
                prompt_sha TEXT NOT NULL,
                text_sha   TEXT NOT NULL,
                created    REAL NOT NULL,
                result     TEXT NOT NULL,
                PRIMARY KEY (model, prompt_sha, text_sha)
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
            metadata.get("content_hash"),
        )
        with self.connect() as con:
            con.execute('''
                INSERT OR IGNORE INTO documents
                (doc_id, filename, filepath, file_size, last_modified, created, extension, mime_type, language, full_text, full_char_count, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            cur.execute(
                "SELECT * FROM documents WHERE filename = ? ORDER BY created DESC LIMIT 1;",
                (filename,)
            )
            row = cur.fetchone()
            return dict(zip(self.KEYS, row)) if row else None
        
    def list_documents(self) -> List[Dict[str, Any]]:
        with self.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM documents;")
            rows = cur.fetchall()
            documents = []
            for row in rows:
                documents.append(dict(zip(self.KEYS, row)))
            return documents

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Alias for list_documents for backward compatibility."""
        return self.list_documents()

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

    def get_llm_cache(self, model: str, prompt_sha: str, text_sha: str, max_age_hours: int) -> Optional[str]:
        """Return cached raw LLM result or None if not found / expired / invalid JSON."""
        import time
        with self.connect() as con:
            row = con.execute(
                "SELECT result, created FROM llm_cache WHERE model=? AND prompt_sha=? AND text_sha=?;",
                (model, prompt_sha, text_sha)
            ).fetchone()
        if not row:
            return None
        result, created = row
        # TTL check
        age_h = (time.time() - float(created)) / 3600.0
        if age_h > float(max_age_hours):
            return None
        return result

    def put_llm_cache(self, model: str, prompt_sha: str, text_sha: str, result: str) -> None:
        """Insert or replace a cached raw LLM result (idempotent)."""
        import time
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO llm_cache (model, prompt_sha, text_sha, created, result) VALUES (?, ?, ?, ?, ?);",
                (model, prompt_sha, text_sha, time.time(), result)
            )
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
    - Nodes unique on name
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
                UNIQUE(name)
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

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch exactly one node by name.
        Returns a dict with: name, type, description, source_id, filepath — or None if missing.
        """
        keys = [k for k in self.KEYS_NODE if k != "id"]
        with self.connect() as con:
            cur = con.cursor()
            cur.execute('''
                SELECT name, type, description, source_id, filepath
                FROM nodes
                WHERE name = ?;
            ''', (name,))
            row = cur.fetchone()
        return dict(zip(keys, row)) if row else None


    def get_nodes(self, names: List[str]) -> List[Dict[str, Any]]:
        if not names:
            return []
        keys = [k for k in self.KEYS_NODE if k != "id"]
        placeholders = ",".join(["?"] * len(names))
        with self.connect() as con:
            cur = con.cursor()
            cur.execute(f'''
                SELECT name, type, description, source_id, filepath
                FROM nodes
                WHERE name IN ({placeholders});
            ''', tuple(names))
            rows = cur.fetchall()
            return [dict(zip(keys, row)) for row in rows] if rows else []
    

    def get_nodes_by_chunk_uuids(self, chunk_uuids: List[str]) -> List[Dict[str, Any]]:
        if not chunk_uuids:
            return []
        results = []
        with self.connect() as con:
            # Find all nodes that contain any of our chunk_uuids in their source_id
            for chunk_uuid in chunk_uuids:
                # Query nodes
                cur = con.cursor()
                cur.execute("""
                    SELECT name, type, description, source_id, filepath
                    FROM nodes
                    WHERE source_id LIKE ?
                """, (f"%{chunk_uuid}%",))
                rows = cur.fetchall()
                keys = [k for k in self.KEYS_NODE if k != "id"]
                results.extend([dict(zip(keys, row)) for row in rows] if rows else [])

        return results

    def delete_node(self, name: str) -> None:
        """
        Delete exactly one node identified by name.
        """
        with self.connect() as con:
            con.execute('DELETE FROM nodes WHERE name = ?;', (name,))
            con.commit()


    def delete_nodes(self, names: List[str]) -> None:
        """
        Bulk delete by name.
        """
        if not names:
            return
        with self.connect() as con:
            con.executemany('DELETE FROM nodes WHERE name = ?;', [(n,) for n in names])
            con.commit()


    def update_node(self, name: str, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        allowed_keys = {"type", "description", "source_id", "filepath"}
        set_clauses = []
        values = []
        for k, v in updates.items():
            if k in allowed_keys:
                set_clauses.append(f"{k} = ?")
                values.append(v)
        if not set_clauses:
            return
        values.append(name)
        set_clause = ", ".join(set_clauses)
        with self.connect() as con:
            con.execute(f"UPDATE nodes SET {set_clause} WHERE name = ?;", values)
            con.commit()

    def update_nodes(self, updates_list: List[Dict[str, Any]]) -> None:
        if not updates_list:
            return
        for update in updates_list:
            self.update_node(update["name"], update)

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
        
    def get_edges_by_chunk_uuids(self, chunk_uuids: List[str]) -> List[Dict[str, Any]]:
        if not chunk_uuids:
            return []
        results = []
        with self.connect() as con:
            # Find all edges that contain any of our chunk_uuids in their source_id
            for chunk_uuid in chunk_uuids:
                # Query edges
                cur = con.cursor()
                cur.execute("""
                    SELECT source_name, target_name, weight, description, keywords, source_id, filepath
                    FROM edges
                    WHERE source_id LIKE ?
                """, (f"%{chunk_uuid}%",))
                rows = cur.fetchall()
                keys = [k for k in self.KEYS_EDGE if k not in {"id", "u_source_name", "u_target_name"}]
                results.extend([dict(zip(keys, row)) for row in rows] if rows else [])

        return results
        
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
    def __init__(self, collection: str, chroma_dir: str, embedder: Optional[Embedder] = None):
        self._chroma_dir = chroma_dir
        
        _ensure_dir_for(os.path.join(chroma_dir, ".sentinel"))
        self.client = chromadb.PersistentClient(path=chroma_dir)  # type: ignore
        self.col = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"}  # keep existing setting
        )

        self.embedder = embedder or Embedder()

        # Verify or attach fingerprint
        self._verify_or_attach_fingerprint()

    def _verify_or_attach_fingerprint(self) -> None:
        fp = _current_models_fingerprint(self.embedder)
        meta = dict(self.col.metadata or {})

        stored_json = meta.get("fingerprint_json")
        stored_sha  = meta.get("fingerprint_sha")

        # First time (no fingerprint): try to attach it
        if not stored_json or not stored_sha:
            new_meta = {**meta, "fingerprint_json": fp["json"], "fingerprint_sha": fp["sha256"]}
            try:
                # chroma>=0.4 supports modify(); if not, recreate with metadata
                if hasattr(self.col, "modify"):
                    self.col.modify(metadata=new_meta)  # type: ignore[attr-defined]
                else:
                    # If modify() isn't available, recreate to persist metadata
                    name = self.col.name
                    self.client.delete_collection(name)
                    self.col = self.client.get_or_create_collection(name=name, metadata=new_meta)
            except Exception:
                # Non-fatal: if we cannot write metadata, we still proceed *this time*,
                # but note that subsequent runs may not detect mismatches.
                pass
            return

        # Compare with stored
        try:
            prev = json.loads(stored_json)
        except Exception:
            prev = {}

        mismatches = []
        for k in ("provider", "embedding_model", "embedding_dim", "llm_model"):
            if prev.get(k) != fp.get(k):
                mismatches.append((k, prev.get(k), fp.get(k)))

        if mismatches:
            # Suggest exact .env keys the user should set to match the stored DB
            suggestions = []
            if settings.provider.provider == "openai":
                for k, old, _ in mismatches:
                    if k == "embedding_model" and old:
                        suggestions.append(f"OPENAI_EMBEDDINGS_MODEL={old}")
                    if k == "llm_model" and old:
                        suggestions.append(f"OPENAI_LLM_MODEL={old}")
            else:
                for k, old, _ in mismatches:
                    if k == "embedding_model" and old:
                        suggestions.append(f"AZURE_OPENAI_EMB_DEPLOYMENT_NAME={old}")
                    if k == "llm_model" and old:
                        suggestions.append(f"AZURE_OPENAI_LLM_DEPLOYMENT_NAME={old}")

            lines = "\n".join(f"  - {k}: stored='{old}' vs now='{new}'" for k, old, new in mismatches)
            hint = "\n".join(f"* {s}" for s in suggestions) or "(adjust your settings to match the stored values)"

            raise RuntimeError(
                f"[Model mismatch] Chroma collection '{self.col.name}' at '{self._chroma_dir}' was built with different models.\n"
                f"{lines}\n\n"
                f"To use this existing database, set in your .env:\n{hint}\n\n"
                f"Or delete/rebuild the collection directory to regenerate vectors."
            )

    def _add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        metadatas = [{k: v if v is not None else "" for k, v in m.items()} for m in metadatas]
        embeddings = self.embedder.embed_texts(texts)
        self.col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def _get(self, ids: List[str]) -> List[Dict[str, Any]]:
        res = self.col.get(ids=ids, include=["documents", "metadatas"])
        return self.to_list(res)

    def _delete(self, ids: List[str]) -> None:
        self.col.delete(ids=ids)

    def _upsert(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        metadatas = [{k: v if v is not None else "" for k, v in m.items()} for m in metadatas]
        embeddings = self.embedder.embed_texts(texts)
        self.col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def _query(self, texts: List[str], n_results: int, where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        embeddings = self.embedder.embed_texts(texts)
        res = self.col.query(query_embeddings=embeddings, n_results=n_results, where=where,
                             where_document=where_document, include=["documents", "metadatas", "distances"])
        return self.to_list(res)

    @staticmethod
    def to_list(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not results:
            return []
        
        num_items = len(next(iter(results.values())))
        return [
            {key: values[i] if isinstance(values, list) else values for key, values in results.items()}
            for i in range(num_items)
        ]


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
    Uniqueness: name -> id "name"
    metadatas:
      - name, type, description, source_id, filepath
    """

    @staticmethod
    def _embed_text(name: str, etype: str, description: str) -> str:
        type_part = f"[{etype}]" if etype else ""
        parts = [name]
        if type_part:
            parts.append(type_part)
        if description:
            parts.append(description)
        return " ".join(parts).strip()

    def add_entities(self, entities: Sequence[Dict[str, Any]]) -> None:
        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for e in entities:
            name = e["name"]
            etype = e.get("type", "") or ""
            desc = e.get("description", "") or ""
            ids.append(name)
            texts.append(self._embed_text(name, etype, desc))
            metas.append({
                "name": name,
                "type": etype,
                "description": desc,
                "source_id": e.get("source_id"),
                "filepath": e.get("filepath"),
            })
        self._add(ids, texts, metas)

    def get_entities(self, names: List[str]) -> List[Dict[str, Any]]:
        return self._get(ids=names)

    def delete_entities(self, names: List[str]) -> None:
        self._delete(ids=names)

    def upsert_entities(self, entities: Sequence[Dict[str, Any]]) -> None:
        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for e in entities:
            name = e["name"]
            etype = e.get("type", "") or ""
            desc = e.get("description", "") or ""
            ids.append(name)
            texts.append(self._embed_text(name, etype, desc))
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

# ---------------------------
# Unified Storage Facade
# ---------------------------

# Re-export the unified StoragePaths dataclass from graph.settings for compatibility.
StoragePaths = SettingsStoragePaths


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

    def __init__(self, paths: Optional[StoragePaths] = None, embedder: Optional[Embedder] = None):
        if paths is None:
            paths = replace(settings.storage)

        # Schema DBs
        self.documents = DocumentsDB(paths.documents_db)
        self.chunks = ChunksDB(paths.chunks_db)
        self.graph = GraphDB(paths.graph_db)

        # Embedder (shared)
        self.embedder = embedder or Embedder()

        # Vector DBs (each in its own persistent directory => separate DBs) but same embedder.
        self.chunk_vectors = ChunkVectors(collection="chunks", chroma_dir=paths.chroma_chunks, embedder=self.embedder)
        self.entity_vectors = EntityVectors(collection="entities", chroma_dir=paths.chroma_entities, embedder=self.embedder)
        self.relation_vectors = RelationVectors(collection="relations", chroma_dir=paths.chroma_relations, embedder=self.embedder)

    def init(self):
        """Create tables/collections if they don't exist yet."""
        self.documents.init()
        self.chunks.init()
        self.graph.init()

    def get_llm_cache(self, model: str, prompt_sha: str, text_sha: str, max_age_hours: int) -> Optional[str]:
        return self.documents.get_llm_cache(model, prompt_sha, text_sha, max_age_hours)

    def put_llm_cache(self, model: str, prompt_sha: str, text_sha: str, result: str) -> None:
        self.documents.put_llm_cache(model, prompt_sha, text_sha, result)

    # ---------- Add-only APIs ----------

    # 1) Documents schema
    def add_document(self, metadata: Dict[str, Any], full_text: str) -> None:
        self.documents.add_document(metadata, full_text)

    # 2) Chunks schema
    def add_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        self.chunks.add_chunks(chunks)

    # 3) Knowledge Graph schema
    def add_kg_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        self.graph.add_nodes(nodes)

    def add_kg_edges(self, edges: List[Dict[str, Any]]) -> None:
        self.graph.add_edges(edges)

    # 4) Chunk vectors
    def add_chunk_vectors(self, chunks: Sequence[Dict[str, Any]]) -> None:
        if chunks:
            self.chunk_vectors.add_chunks(chunks)

    # 5) Entity vectors
    def add_entity_vectors(self, entities: Sequence[Dict[str, Any]]) -> None:
        """
        Each entity dict must include: name, type; optional: description, source_id, filepath
        Uniqueness enforced via entity name.
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

    # ---------- Get-only APIs ----------

    # 1) Documents
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get_document(doc_id)
    
    def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        return self.documents.get_document_by_filename(filename)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        return self.documents.list_documents()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Alias for list_documents for backward compatibility."""
        return self.documents.get_all_documents()

    # 2) Chunks
    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        return self.chunks.get_chunks_by_doc_id(doc_id)
    
    def get_chunks_by_filename(self, filename: str) -> List[Dict[str, Any]]:
        return self.chunks.get_chunks_by_filename(filename)

    def get_chunk_by_uuid(self, chunk_uuid: str) -> List[Dict[str, Any]]:
        return self.chunks.get_chunk_by_uuid(chunk_uuid)

    def get_chunks_by_uuids(self, chunk_uuids: List[str]) -> List[Dict[str, Any]]:
        return self.chunks.get_chunks_by_uuids(chunk_uuids)
    
    # 3) Graph
    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Pass-through: fetch one graph node by name.
        """
        return self.graph.get_node(name)

    def get_nodes(self, names: List[str]) -> List[Dict[str, Any]]:
        return self.graph.get_nodes(names)
    
    def get_nodes_by_chunk_uuids(self, chunk_uuids: List[str]) -> List[Dict[str, Any]]:
        return self.graph.get_nodes_by_chunk_uuids(chunk_uuids)

    def get_edge(self, source_name: str, target_name: str) -> Optional[Dict[str, Any]]:
        return self.graph.get_edge(source_name, target_name)

    def get_edges(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        return self.graph.get_edges(pairs)
    
    def get_edges_by_chunk_uuids(self, chunk_uuids: List[str]) -> List[Dict[str, Any]]:
        return self.graph.get_edges_by_chunk_uuids(chunk_uuids)

    # 4) Chunk Vectors
    def get_chunk_vector(self, chunk_uuid: str) -> Optional[Dict[str, Any]]:
        return self.chunk_vectors.get([chunk_uuid])

    # 5) Entity Vectors
    def get_entities(self, names: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve entity vectors by name.
        Returns a list of results (each with 'ids', 'documents', 'metadatas', 'distances' if available).
        """
        if not names:
            return []
        return self.entity_vectors.get_entities(names)


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
    def delete_node(self, name: str) -> None:
        """
        Pass-through: delete one graph node by name.
        """
        self.graph.delete_node(name)

    def delete_nodes(self, names: List[str]) -> None:
        """
        Pass-through: bulk delete graph nodes by name.
        """
        self.graph.delete_nodes(names)

    def delete_edge(self, source_name: str, target_name: str) -> None:
        self.graph.delete_edge(source_name, target_name)

    def delete_edges(self, pairs: List[Tuple[str, str]]) -> None:
        self.graph.delete_edges(pairs)

    # 4) Chunk Vectors
    def delete_chunk_vector(self, chunk_uuid: str) -> None:
        self.chunk_vectors.delete([chunk_uuid])

    # 5) Entity Vectors
    def delete_entity_vector(self, names: List[str]) -> None:
        """
        Delete entity vectors by name.
        """
        if not names:
            return
        self.entity_vectors.delete_entities(names)

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
    def upsert_node(self, name: str, updates: Dict[str, Any]) -> None:
        self.graph.update_node(name, updates)

    def upsert_nodes(self, updates_list: List[Dict[str, Any]]) -> None:
        """
        Split incoming rows into adds vs updates based on existing node names.
        """
        to_add: List[Dict[str, Any]] = []
        to_update: List[Dict[str, Any]] = []
        if not updates_list:
            return
        # Bulk fetch all names once, then test membership
        names = sorted({u["name"] for u in updates_list})
        existing = self.graph.get_nodes(names) if names else []
        existing_names = {row["name"] for row in existing}
        for upd in updates_list:
            name = upd.get("name")
            if not name:
                continue
            if name in existing_names:
                to_update.append(upd)
            else:
                to_add.append({
                    "name": name,
                    "type": (upd.get("type") or "unknown"),
                    "description": (upd.get("description") or ""),
                    "source_id": (upd.get("source_id") or ""),
                    "filepath": (upd.get("filepath") or ""),
                })
        if to_add:
            self.graph.add_nodes(to_add)
        if to_update:
            self.graph.update_nodes(to_update)

    def upsert_edge(self, source_name: str, target_name: str, updates: Dict[str, Any]) -> None:
        self.graph.update_edge(source_name, target_name, updates)

    def upsert_edges(self, updates_list: List[Dict[str, Any]]) -> None:
        to_add = []
        to_update = []
        for upd in updates_list:
            existing = self.graph.get_edge(upd["source_name"], upd["target_name"])
            if existing is None:
                to_add.append(upd)
            else:
                to_update.append(upd)
        if to_add:
            self.graph.add_edges(to_add)
        if to_update:
            self.graph.update_edges(to_update)

    # 4) Chunk Vectors
    def upsert_chunk_vector(self, chunks: Sequence[Dict[str, Any]]) -> None:
        self.chunk_vectors.upsert(chunks)

    # 5) Entity Vectors
    def upsert_entity_vector(self, entities: List[Dict[str, Any]]) -> None:
        self.entity_vectors.upsert_entities(entities)

    # 6) Relation Vectors
    def upsert_relation_vector(self, relations: List[Dict[str, Any]]) -> None:
        self.relation_vectors.upsert_relations(relations)

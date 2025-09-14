from __future__ import annotations
import mimetypes
import uuid
from pathlib import Path
from typing import Dict, Any, List, Sequence, Tuple, Optional
from storage import Storage, _normalize_pair
from fileparser import FileParser
from chunker import chunk_parsed_pages
from extractor import extract_from_chunks
from collections import defaultdict
from llm import llm_summarize_text
from collections import Counter
from settings import settings
import os
from dataclasses import dataclass
import hashlib

# Helpers -------------------------------------------

@dataclass(frozen=True)
class FileStats:
    size: int
    mtime: float

def normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    # Lower-case keys, unify language key
    meta = {str(k).lower(): v for k, v in (meta or {}).items()}
    # normalize to 'language'
    if "Language" in meta and "language" not in meta:
        meta["language"] = meta.pop("Language")
    return meta

def file_sha256(p: Path, chunk_size=1024*1024) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def should_skip_ingestion(storage: Storage, p: Path, content_hash: Optional[str] = None) -> bool:
    """
    Return True if file 'p' has already been ingested (same content_hash).
    If 'content_hash' is provided, reuse it to avoid hashing twice.
    """
    doc = storage.get_document_by_filename(p.name)
    if not doc:
        return False
    ch = content_hash or file_sha256(p)
    return doc.get("content_hash") == ch

# Core functions ------------------------------------

def parse_to_pages(filepath: Path):
    pages = None
    meta = {}
    try:
        fp = FileParser()
        res = fp.parse_file(str(filepath))
        if isinstance(res, tuple):
            pages = res[0]
            if len(res) > 1 and isinstance(res[1], dict):
                meta = res[1]
        else:
            pages = res
    except Exception:
        pages = None
    
    return pages, meta

def build_chunks(
    pages: Sequence[Tuple[int, str]],
    doc_id: str,
    filename: str,
) -> List[Dict[str, Any]]:
    """
    Normalize chunks for storage:
      - Ensures chunk_uuid, chunk_id, char_count, start_page, end_page
      - Attaches doc_id and filename to each chunk
      - Returns [] on chunking error (non-fatal)
    """
    try:
        raw = chunk_parsed_pages(pages)
    except Exception:
        return []

    norm: List[Dict[str, Any]] = []
    for i, c in enumerate(raw or []):
        text = (c.get("text") or "")
        start = int(c.get("start_page", 0))
        end = int(c.get("end_page", start))
        norm.append({
            "chunk_uuid": c.get("chunk_uuid") or str(uuid.uuid4()),
            "doc_id": doc_id,
            "chunk_id": int(c.get("chunk_id", i)),
            "filename": filename,
            "text": text,
            "char_count": int(c.get("char_count", len(text))),
            "start_page": start,
            "end_page": end,
        })
    return norm

# TODO: While grouping nodes and edges, pay attention to not exceed any DB field limits
# and also more importantly make sure that DELIMITER and some unnecessary punctuations at the start/end of merged fields are removed.
# e.g. if a node description is "desc1||desc2||", it should be "desc1||desc2" after merging.
# sometimes LLM generated descriptions or former versions may have trailing punctuation like .,;! etc.

def group_nodes(storage: Storage, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Consolidate entity nodes by the (name, type) key, merging duplicates across chunks/files,
    and fold in any already-stored rows **in bulk** (single DB round-trip).
    """
    delim = settings.ingestion.delimiter
    # 1) group incoming by (name, type)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for n in nodes:
        name = str(n["name"]).strip()
        etype = str(n["type"]).strip()
        grouped[(name, etype)].append({
            "description": n.get("description", "") or "",
            "source_id": n.get("source_id", "") or "",
            "filepath": n.get("filepath", "") or "",
        })
    # 2) bulk fetch existing rows for all names, then merge by (name, type)
    names = sorted({k[0] for k in grouped.keys()})
    if names:
        existing_rows = storage.get_nodes(names) or []
        for row in existing_rows:
            key = (row.get("name", "").strip(), row.get("type", "").strip())
            grouped[key].append({
                "description": row.get("description", "") or "",
                "source_id": row.get("source_id", "") or "",
                "filepath": row.get("filepath", "") or "",
            })
    # 3) collapse each (name, type) group into a single merged record
    merged: List[Dict[str, Any]] = []
    for (name, etype), attrs in grouped.items():
        def join(field: str) -> str:
            parts = []
            for a in attrs:
                v = (a.get(field, "") or "")
                if v:
                    parts.extend(p.strip() for p in v.split(delim) if p.strip())
            # stable dedupe
            uniq = []
            seen = set()
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            return delim.join(uniq)
        merged.append({
            "name": name,
            "type": etype,  # keep exact key's type (no "most_common" voting)
            "description": join("description"),
            "source_id": join("source_id"),
            "filepath": join("filepath"),
        })
    return merged


def group_edges(storage: Storage, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Consolidate relation edges by normalized undirected (source_name, target_name),
    merge duplicates across chunks/files, and blend in any already-stored rows via
    a single bulk fetch.

    Input edge schema (expected keys on each dict):
      - source_name (str), target_name (str), weight (number|None, optional),
        description (str, optional), keywords (str, optional), source_id (str, optional),
        filepath (str, optional)

    Steps:
      1) Normalize each (source, target) to a canonical undirected pair and accumulate attributes.
      2) Bulk-fetch any existing edges for ALL pairs (storage.get_edges) and inject their attrs
         into the same groups to preserve/merge prior state.
      3) For each pair, merge text fields using the configured delimiter, sum weights (ignoring None),
         and return a flat list of normalized edge dicts.

    Notes:
      - Keywords provided as comma-separated strings are normalized once into delimiter-joined sets.
      - Uses _normalize_pair() to ensure undirected uniqueness is consistent with the DB/vector IDs.
    """
    delim = settings.ingestion.delimiter

    # 1) group incoming by normalized undirected pair
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for e in edges:
        src, tgt = _normalize_pair(e["source_name"], e["target_name"])
        grouped[(src, tgt)].append({
            "description": e.get("description", "") or "",
            # normalize comma-separated input, keep unique tokens only
            "keywords": delim.join({k.strip() for k in (e.get("keywords", "") or "").split(",") if k.strip()}),
            "weight": float(e["weight"]) if e.get("weight") is not None else None,
            "source_id": e.get("source_id", "") or "",
            "filepath": e.get("filepath", "") or "",
        })

    # 2) bulk fetch ALL existing edges once
    pairs = list(grouped.keys())
    if pairs:
        existing_edges = storage.get_edges(pairs) or []
        for ex in existing_edges:
            u_src, u_tgt = _normalize_pair(ex["source_name"], ex["target_name"])
            grouped[(u_src, u_tgt)].append({
                "description": ex.get("description", "") or "",
                "keywords": ex.get("keywords", "") or "",
                "weight": ex.get("weight", 0),
                "source_id": ex.get("source_id", "") or "",
                "filepath": ex.get("filepath", "") or "",
            })

    # 3) collapse each pair's attrs into one merged record
    merged: List[Dict[str, Any]] = []
    for (src, tgt), attrs in grouped.items():
        desc = delim.join({
            s.strip()
            for a in attrs
            for s in (a.get("description", "") or "").split(delim)
            if s.strip()
        })
        kws = delim.join({
            k.strip()
            for a in attrs
            for k in (a.get("keywords", "") or "").split(delim)
            if k.strip()
        })
        w_sum = sum(a["weight"] for a in attrs if a.get("weight") is not None)
        srcs = delim.join({
            s.strip()
            for a in attrs
            for s in (a.get("source_id", "") or "").split(delim)
            if s.strip()
        })
        fps = delim.join({
            s.strip()
            for a in attrs
            for s in (a.get("filepath", "") or "").split(delim)
            if s.strip()
        })

        merged.append({
            "source_name": src,
            "target_name": tgt,
            "description": desc,
            "keywords": kws,
            "weight": w_sum,
            "source_id": srcs,
            "filepath": fps,
        })

    return merged



def merge_graph_data(storage: Storage, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]):
    """Merges new entities and relations with existing ones in the storage."""
    merged_entities = []
    entities = group_nodes(storage, entities)
    for ent in entities:
        if len(ent["description"].split("||")) > settings.ingestion.description_segment_limit:
            ent["description"] = llm_summarize_text(ent["description"])
        merged_entities.append(ent)

    
    merged_relations = []
    relations = group_edges(storage, relations)
    for rel in relations:
        if len(rel["description"].split("||")) > settings.ingestion.description_segment_limit:
            rel["description"] = llm_summarize_text(rel["description"])
        merged_relations.append(rel)

    return merged_entities, merged_relations


def fill_missing_nodes(storage: Storage, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure all names referenced by edges exist as nodes. Because relationships currently
    carry *names only*, we conservatively create a placeholder node per name with
    type='unknown' when no node for that (name, 'unknown') exists yet.
    """
    delim = settings.ingestion.delimiter
    by_name: Dict[str, Dict[str, set]] = defaultdict(lambda: {"source_id": set(), "filepath": set()})
    for e in edges:
        for nm in (e["source_name"], e["target_name"]):
            if e.get("source_id"):
                by_name[nm]["source_id"].add(e["source_id"])
            if e.get("filepath"):
                by_name[nm]["filepath"].add(e["filepath"])
    names = list(by_name.keys())
    existing = storage.get_nodes(names) if names else []
    existing_pairs = {(row["name"], row["type"]) for row in existing}
    to_add: List[Dict[str, Any]] = []
    for name, meta in by_name.items():
        pair = (name, "unknown")
        if pair not in existing_pairs:
            to_add.append({
                "name": name,
                "type": "unknown",
                "description": "",
                "source_id": delim.join(sorted(meta["source_id"])) if meta["source_id"] else "",
                "filepath": delim.join(sorted(meta["filepath"])) if meta["filepath"] else "",
            })
    if to_add:
        storage.upsert_nodes(to_add)
    return to_add

def ingest_paths(paths: List[Path]):
    """Ingests files at given paths into the storage system."""
    storage = Storage()
    storage.init()

    all_chunks: List[Dict[str, Any]] = []
    all_entities: List[Dict[str, Any]]  = []
    all_relations: List[Dict[str, Any]]  = []

    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        content_hash = file_sha256(p)
        if should_skip_ingestion(storage, p, content_hash):
            print(f"Skipping {p.name} (unchanged).")
            continue
        pages, file_meta = parse_to_pages(p)
        if not pages or not file_meta:
            print(f"Skipping {p} due to parsing error.")
            continue

        file_meta = normalize_metadata(file_meta)
        full_text = "\n".join([page[1] for page in pages])
        
        # Build storage row (doc_meta) – this is distinct from file_meta and is schema-aligned fields expected by DocumentsDB
        st = p.stat()
        doc_meta = {
            "doc_id": str(uuid.uuid4()),
            "filename": p.name,
            "filepath": str(p),                          # keep if useful for tracing
            "file_size": st.st_size,
            "last_modified": st.st_mtime,
            "created": st.st_mtime,                      # or time.time()
            "extension": p.suffix.lower(),
            "mime_type": ((file_meta or {}).get("mime_type") or mimetypes.guess_type(str(p))[0] or ""),
            "language": (file_meta or {}).get("language", "unknown"),
            "content_hash": content_hash,                
            "full_char_count": len(full_text),
        }
              
        storage.add_document(doc_meta, full_text)

        chunks = build_chunks(pages, doc_meta["doc_id"], doc_meta["filename"])
        storage.add_chunks(chunks)
        all_chunks.extend(chunks)

        # Extract entities and relations from chunks
        # res['entities'], res['relationships'], res['content_keywords']
        res = extract_from_chunks(chunks)
        nodes, edges = merge_graph_data(storage, res['entities'], res['relationships'])
        if nodes:
            storage.upsert_nodes(nodes)
        if edges:
            storage.upsert_edges(edges)
        # add missing nodes to storage with minimal info in case edges refer to non-existing nodes
        missing_nodes = fill_missing_nodes(storage, edges)
        if missing_nodes:
            nodes.extend(missing_nodes)

        all_entities.extend(nodes)
        all_relations.extend(edges)

    # Finally, add all chunks, entities, and relations to vector DB
    if all_chunks:
        storage.upsert_chunk_vector(all_chunks)
        storage.upsert_entity_vector(all_entities)
        storage.upsert_relation_vector(all_relations)

        #TODO: Add a node for filename/document with edges to all entities/relations extracted from it.

def main():
    root = Path('docs')
    paths = FileParser(root).filepaths
    if not paths:
        print("No files to ingest.")
        return
    ingest_paths(paths)

if __name__ == "__main__":
    main()

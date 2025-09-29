from __future__ import annotations
import mimetypes
import uuid
from pathlib import Path
from typing import Dict, Any, List, Sequence, Tuple, Optional
from collections import Counter, defaultdict

from storage import Storage, _normalize_pair
from fileparser import FileParser
from chunker import chunk_parsed_pages
from extractor import extract_from_chunks
from llm import llm_summarize_text
from settings import settings
import os
import hashlib

#--------------------------------------------------
# Helpers 
#--------------------------------------------------

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

def _resolve_type(votes: Counter, existing_type: str = "") -> str:
    existing = (existing_type or "").strip()
    if not votes:
        return existing or "unknown"

    normalized: Dict[str, int] = {}
    canonical: Dict[str, str] = {}
    for raw_type, count in votes.items():
        candidate = (raw_type or "").strip()
        if not candidate or candidate.lower() == "unknown":
            continue
        key = candidate.lower()
        normalized[key] = normalized.get(key, 0) + count
        canonical.setdefault(key, candidate)

    if normalized:
        max_count = max(normalized.values())
        contenders = sorted([k for k, c in normalized.items() if c == max_count])
        existing_key = existing.lower()
        for key in contenders:
            if existing and existing_key == key:
                return canonical[key]
        return canonical[contenders[0]]

    if existing and existing.lower() != "unknown":
        return existing

    # All votes were unknown or empty; default to unknown to avoid oscillation.
    return "unknown"


def dedupe_entities_for_vectors(entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return entities deduplicated by name, preferring typed entries over "unknown"."""

    dedup: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for ent in entities or []:
        name = str(ent.get("name", "") or "").strip()
        if not name:
            continue

        current = dedup.get(name)
        if current is None:
            dedup[name] = ent
            order.append(name)
            continue

        current_type = str(current.get("type", "") or "").strip().lower()
        new_type = str(ent.get("type", "") or "").strip().lower()

        if current_type == "unknown" and new_type and new_type != "unknown":
            dedup[name] = ent
        elif new_type == "unknown" and current_type and current_type != "unknown":
            continue
        else:
            dedup[name] = ent

    return [dedup[name] for name in order]


def group_nodes(storage: Storage, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Consolidate entity nodes by name (unique key) and determine the final node type by
    majority vote among all available hints (incoming + existing rows).
    """

    delim = settings.ingestion.delimiter
    grouped: Dict[str, Dict[str, Any]] = {}

    for n in nodes:
        name = str(n.get("name", "")).strip()
        if not name:
            continue
        etype = str(n.get("type", "")).strip()
        bucket = grouped.setdefault(name, {
            "attrs": [],
            "votes": Counter(),
            "existing": None,
        })
        bucket["attrs"].append({
            "description": n.get("description", "") or "",
            "source_id": n.get("source_id", "") or "",
            "filepath": n.get("filepath", "") or "",
        })
        vote_key = etype or "unknown"
        bucket["votes"][vote_key] += 1

    names = sorted(grouped.keys())
    if names:
        existing_rows = storage.get_nodes(names) or []
        for row in existing_rows:
            name = str(row.get("name", "") or "").strip()
            if not name:
                continue
            bucket = grouped.setdefault(name, {
                "attrs": [],
                "votes": Counter(),
                "existing": None,
            })
            bucket["attrs"].append({
                "description": row.get("description", "") or "",
                "source_id": row.get("source_id", "") or "",
                "filepath": row.get("filepath", "") or "",
            })
            existing_type = str(row.get("type", "") or "").strip()
            bucket["votes"][existing_type or "unknown"] += 1
            bucket["existing"] = row

    merged: List[Dict[str, Any]] = []
    for name, data in grouped.items():
        existing_type = ""
        if data.get("existing"):
            existing_type = str(data["existing"].get("type", "") or "").strip()

        def join(field: str) -> str:
            parts = []
            for a in data["attrs"]:
                v = (a.get(field, "") or "")
                if v:
                    parts.extend(p.strip() for p in v.split(delim) if p.strip())
            uniq: List[str] = []
            seen = set()
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            return delim.join(uniq)

        final_type = _resolve_type(data["votes"], existing_type)
        merged.append({
            "name": name,
            "type": final_type,
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


def ensure_edge_endpoints(storage: Storage, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure that nodes exist for every endpoint referenced in the provided edges.

    - Node identity is based on the node name only (string equality).
    - Note to self: Does not update node types anymore because we don't extract src/tgt types.
    - Newly created nodes are given type "unknown" and empty metadata fields.
    - Existing nodes are left unchanged.

    Returns
    -------
    affected : List[Dict[str, Any]]
        A list of nodes that were newly created (so that vector embeddings can be
        generated later). If no new nodes were needed, the list is empty.
    """
    # Collect all unique node names from edges (source + target)
    names = {
        (e.get("source_name") or "").strip()
        for e in edges
    } | {
        (e.get("target_name") or "").strip()
        for e in edges
    }
    names.discard("")  # drop empty names

    if not names:
        return []

    # Fetch any existing nodes in bulk
    existing_rows = storage.get_nodes(list(names)) or []
    existing_by_name = {row["name"]: row for row in existing_rows if row.get("name")}

    pending: List[Dict[str, Any]] = []   # nodes to insert into storage
    affected: List[Dict[str, Any]] = []  # nodes we created (return to caller)

    for nm in names:
        if nm not in existing_by_name:
            # Create a new placeholder node
            node = {
                "name": nm,
                "type": "unknown",
                "description": "",
                "source_id": "",
                "filepath": "",
            }
            pending.append(node)
            affected.append(node)

    # Bulk insert missing nodes
    if pending:
        storage.upsert_nodes(pending)

    return affected



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
            "created": st.st_mtime,                      
            "extension": p.suffix.lower(),
            "mime_type": ((file_meta or {}).get("mime_type") or mimetypes.guess_type(str(p))[0] or ""),
            "language": (file_meta or {}).get("language", "unknown"),
            "content_hash": content_hash,                
            "full_char_count": len(full_text),
        }

        storage.add_document(doc_meta, full_text)  # from storage.py

        chunks = build_chunks(pages, doc_meta["doc_id"], doc_meta["filename"])
        storage.add_chunks(chunks)  # from storage.py
        all_chunks.extend(chunks)

        # Extract entities and relations from chunks
        # res['entities'], res['relationships'], res['content_keywords']
        res = extract_from_chunks(chunks)  # from extractor.py
        
        # Consolidate/merge entities (by (name,type)) and upsert those first
        entities_in = res.get("entities", []) or []
        edges_in = res.get("relationships", []) or []

        # Ensure all edge endpoints exist as nodes, create placeholders if needed
        placeholders = ensure_edge_endpoints(storage, edges_in)
        if placeholders:
            all_entities.extend(placeholders)           # collect for vector DB later

        nodes, edges = merge_graph_data(storage, entities_in, edges_in)

        if nodes:
            storage.upsert_nodes(nodes)                 # write schema
            all_entities.extend(nodes)                  # collect for vector DB later

        # Group/merge edges and upsert
        if edges:
            storage.upsert_edges(edges)                 # write schema
            all_relations.extend(edges)                 # collect for vector DB later

    # Finally, add all chunks, entities, and relations to vector DB   
    if all_chunks:
        storage.upsert_chunk_vector(all_chunks) # from storage.py
        deduped_entities = dedupe_entities_for_vectors(all_entities)
        if all_chunks:
            print(f"[ingestion] sample chunk: {all_chunks[0]}")
        if deduped_entities:
            storage.upsert_entity_vector(deduped_entities)
        if all_relations:
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


from __future__ import annotations
import mimetypes
import uuid
from pathlib import Path
from typing import Dict, Any, List, Sequence, Tuple
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

def should_skip_ingestion(storage, p: Path) -> bool:
    doc = storage.get_document_by_filename(p.name)
    if not doc:
        return False
    return doc.get("content_hash") == file_sha256(p)

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
    grouped = defaultdict(list)
    for n in nodes:
        grouped[n["name"]].append({
            "type": n["type"],
            "description": n["description"],
            "source_id": n["source_id"],
            "filepath": n["filepath"],
        })
    # append existing nodes from storage
    for name in list(grouped.keys()):
        existing = storage.get_node(name=name)
        if existing:
            grouped[name].append({
                "type": existing.get("type", ""),
                "description": existing.get("description", ""),
                "source_id": existing.get("source_id", ""),
                "filepath": existing.get("filepath", ""),
                })

    for name in list(grouped.keys()):
        attrs = grouped[name]
        # Choose the most common type if multiple and try to avoid empty/None types if possible
        # which can happen when adding edges with inexisting nodes in graph
        types = [a["type"] for a in attrs]
        known_types = [t for t in types if t and str(t).strip()]
        most_common_type, _ = Counter(known_types or types).most_common(1)[0]
        t = most_common_type
        descriptions = settings.ingestion.delimiter.join({d.strip() for a in attrs if a.get("description") for d in a.get("description").split(settings.ingestion.delimiter)})
        source_ids = settings.ingestion.delimiter.join({s.strip() for a in attrs if a.get("source_id") for s in a.get("source_id").split(settings.ingestion.delimiter)})
        filepaths = settings.ingestion.delimiter.join({f.strip() for a in attrs if a.get("filepath") for f in a.get("filepath").split(settings.ingestion.delimiter)})
        grouped[name] = {
                "name": name,
                "type": t,
                "description": descriptions,
                "source_id": source_ids,
                "filepath": filepaths
            }
    grouped = list(grouped.values())
    return grouped


def group_edges(storage: Storage, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group edges by normalized (source_name, target_name), merge duplicates,
    append existing stored edge (if any), and return a flat list.

    Input edge schema (expected keys):
      - source_name, target_name, weight (optional/number), description, keywords,
        source_id, filepath
    """
    grouped = defaultdict(list)

    # 1) accumulate incoming edges by normalized pair
    for e in edges:
        a, b = e["source_name"], e["target_name"]
        src, tgt = _normalize_pair(a, b)
        grouped[(src, tgt)].append({
            "description": e.get("description", ""),
            "keywords": settings.ingestion.delimiter.join({k.strip() for k in e.get("keywords", "").split(',') if k.strip()}),
            "weight": float(e.get("weight", 0)),      # may be None or number
            "source_id": e.get("source_id", ""),
            "filepath": e.get("filepath", "")
        })

    # 2) append existing edges from storage (if present)
    for (src, tgt) in list(grouped.keys()):
        existing = storage.get_edge(source_name=src, target_name=tgt)
        if existing:
            grouped[(src, tgt)].append({
                "description": existing.get("description", ""),
                "keywords": existing.get("keywords", ""),
                "weight": existing.get("weight", None),
                "source_id": existing.get("source_id", ""),
                "filepath": existing.get("filepath", "")
            })

    # 3) collapse each pair's attrs into one merged record
    merged = []
    for (src, tgt), attrs in grouped.items():
        descriptions = settings.ingestion.delimiter.join({d.strip() for a in attrs if a.get("description") for d in a["description"].split(settings.ingestion.delimiter)})
        keywords     = settings.ingestion.delimiter.join({k.strip() for a in attrs if a.get("keywords") for k in a["keywords"].split(settings.ingestion.delimiter) if k.strip()})
        weights      = sum([a["weight"]  for a in attrs if a.get("weight") is not None])
        source_ids   = settings.ingestion.delimiter.join({a["source_id"]    for a in attrs if a.get("source_id")})
        filepaths    = settings.ingestion.delimiter.join({a["filepath"]     for a in attrs if a.get("filepath")})

        merged.append({
            "source_name": src,
            "target_name": tgt,
            "description": descriptions,
            "keywords": keywords,
            "weight": weights,
            "source_id": source_ids,
            "filepath": filepaths
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
    """Ensures all nodes referenced in edges exist in storage, adding minimal info if needed."""
    node_names = defaultdict(list)
    for edge in edges:
        for name in [edge["source_name"], edge["target_name"]]:
            node_names[name].append({"source_id": edge["source_id"], "filepath": edge["filepath"]})
    # Merge multiple source_id/filepath entries per node name
    node_names = {name: {"source_id": settings.ingestion.delimiter.join({info["source_id"] for info in infos}),
                        "filepath": settings.ingestion.delimiter.join({info["filepath"] for info in infos})}
                    for name, infos in node_names.items()}

    added_nodes = []
    for name, info in node_names.items():
        if not storage.get_node(name=name):
            storage.upsert_nodes([{"name": name, "type": "", "description": "", "source_id": info["source_id"], "filepath": info["filepath"]}])
            added_nodes.append({"name": name, "type": "", "description": "", "source_id": info["source_id"], "filepath": info["filepath"]})
    return added_nodes

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
        if should_skip_ingestion(storage, p):
            print(f"Skipping {p.name} (unchanged).")
            continue
        pages, file_meta = parse_to_pages(p)
        if not pages or not file_meta:
            print(f"Skipping {p} due to parsing error.")
            continue

        file_meta = normalize_metadata(file_meta)
        full_text = "\n".join([page[1] for page in pages])
        content_hash = file_sha256(p)

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
            "mime_type": "",                             # optional; leave empty if unused
            "language": (file_meta or {}).get("language", "unknown"),
            "content_hash": content_hash,                # ← your chosen idempotency key
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

from __future__ import annotations
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple
from graph.storage import Storage, _normalize_pair
from graph.fileparser import FileParser
from graph.chunker import chunk_parsed_pages
from graph.extractor import extract_from_chunks
from collections import defaultdict
from graph.llm import llm_summarize_text
from collections import Counter

DELIMITER = "||"
LIMIT_DESCRIPTION = 5 # Max number of '||' separated segments before summarization

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

def chunk_pages(pages: Tuple[int, str], meta_doc_id: str, filename: str):
    chunks = None
    
    try:
        chunks = chunk_parsed_pages(pages)
    except Exception:
        chunks = None
    
    norm = []
    for i, c in enumerate(chunks):
        norm.append({
            "chunk_id": int(c.get("chunk_id", i)),
            "chunk_uuid": c.get("chunk_uuid") or str(uuid.uuid4()),
            "text": c.get("text") or "",
            "char_count": int(c.get("char_count") or len(c.get("text") or "")),
            "start_page": c.get("start_page"),
            "end_page": c.get("end_page"),
            "doc_id": meta_doc_id,
            "filename": filename
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
        descriptions = DELIMITER.join({d.strip() for a in attrs if a.get("description") for d in a.get("description").split(DELIMITER)})
        source_ids = DELIMITER.join({s.strip() for a in attrs if a.get("source_id") for s in a.get("source_id").split(DELIMITER)})
        filepaths = DELIMITER.join({f.strip() for a in attrs if a.get("filepath") for f in a.get("filepath").split(DELIMITER)})
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
            "keywords": DELIMITER.join({k.strip() for k in e.get("keywords", "").split(',') if k.strip()}),
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
        descriptions = DELIMITER.join({d.strip() for a in attrs if a.get("description") for d in a["description"].split(DELIMITER)})
        keywords     = DELIMITER.join({k.strip() for a in attrs if a.get("keywords") for k in a["keywords"].split(DELIMITER) if k.strip()})
        weights      = sum([a["weight"]  for a in attrs if a.get("weight") is not None])
        source_ids   = DELIMITER.join({a["source_id"]    for a in attrs if a.get("source_id")})
        filepaths    = DELIMITER.join({a["filepath"]     for a in attrs if a.get("filepath")})

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
        if len(ent["description"].split("||")) > LIMIT_DESCRIPTION:
            ent["description"] = llm_summarize_text(ent["description"])
        merged_entities.append(ent)

    
    merged_relations = []
    relations = group_edges(storage, relations)
    for rel in relations:
        if len(rel["description"].split("||")) > LIMIT_DESCRIPTION:
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
    node_names = {name: {"source_id": DELIMITER.join({info["source_id"] for info in infos}),
                        "filepath": DELIMITER.join({info["filepath"] for info in infos})}
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
        pages, metadata = parse_to_pages(p)
        if not pages or not metadata:
            print(f"Skipping {p} due to parsing error.")
            continue
        full_text = "\n".join([page[1] for page in pages])
        chunks = chunk_pages(pages, meta_doc_id=metadata["doc_id"], filename=metadata["filename"])

        # Add document and chunks to storage
        if metadata.get("filepath") == storage.get_chunks_by_filename(metadata["filepath"]):
            # TODO: Add other checks like last modified time, file size, hash, etc.
            # which may require deletion/update of old chunks, entities, relations, vectors, etc.
            continue  # Skip if already ingested
        
        #TODO: Format all data specifically metadata. Lowercase keys, ensure required fields, etc.
        storage.add_document(metadata, full_text)
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

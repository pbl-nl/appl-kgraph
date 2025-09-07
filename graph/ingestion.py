from __future__ import annotations
import os
import argparse
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable
from storage import Storage, _normalize_token, DELIM
from fileparser import FileParser
from chunker import chunk_parsed_pages
from extractor import extract_from_chunks, LLMChat
from prompts import PROMPTS

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

# Max merged description length before summarizing (env overrideable, no CLI churn)
DEFAULT_MAX_ENTITY_DESC_CHARS = int(os.getenv("MAX_ENTITY_DESC_CHARS", "4000"))

def _dedup_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for it in items:
        it = (it or "").strip()
        if not it or it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out

def _merge_and_maybe_summarize_entity_descriptions(
    entities: list[dict[str, any]],
    *,
    language: str,
    max_chars: int = DEFAULT_MAX_ENTITY_DESC_CHARS,
    client: LLMChat | None = None,
) -> dict[tuple[str, str], str]:
    """
    Group by canonical (name,type) using storage._normalize_token + DELIM.
    Return {(name,type): merged_or_summarized_text}.
    """
    # group descriptions
    buckets: dict[str, dict[str, any]] = {}
    for e in entities:
        name = e.get("name") or ""
        etype = e.get("type") or ""
        key = _normalize_token(name) + DELIM + _normalize_token(etype)
        if key not in buckets:
            buckets[key] = {"name": name, "type": etype, "descs": []}
        desc = e.get("description") or ""
        if desc:
            buckets[key]["descs"].append(desc)

    if not buckets:
        return {}

    # reuse client if provided; otherwise build one lazily only if needed
    out: dict[tuple[str, str], str] = {}
    for b in buckets.values():
        name = b["name"]
        etype = b["type"]
        descs = _dedup_preserve_order(b["descs"])
        if not descs:
            continue
        merged = "\n".join(descs).strip()
        if len(merged) <= max_chars:
            out[(name, etype)] = merged
            continue

        # Need a summary → create client lazily to avoid overhead when not needed
        client = client or LLMChat()
        prompt = PROMPTS["summarize_entity_descriptions"].format(
            entity_name=f"{name} [{etype}]",
            description_list="\n".join(f"- {d}" for d in descs),
            language=language or PROMPTS["DEFAULT_LANGUAGE"],
        )
        summary = client.generate(prompt=prompt, system="You write concise, faithful summaries.")
        out[(name, etype)] = (summary or merged)[:max_chars]  # safety clamp
    return out


def ingest_paths(paths: List[Path], db_path: str, chroma_dir: str, collection: str):
    """
    Ingest a list of files:
      1) parse → pages
      2) chunk → chunk dicts via chunk_pages(...)
      3) store documents + chunks (SQLite)
      4) extract entities/relations from chunks
      5) add KG nodes/edges (SQLite)
      6) add vectors:
         - chunk vectors (one per chunk)
         - entity/relation mention vectors (one per sighting), grouped by chunk
    """
    from collections import defaultdict  # local import to keep module top clean

    storage = Storage()
    storage.init()

    all_chunks: List[Dict[str, Any]] = []
    all_entities: List[Dict[str, Any]] = []
    all_relations: List[Dict[str, Any]] = []

    for p in paths:
        if not p.exists() or not p.is_file():
            continue

        pages, metadata = parse_to_pages(p)
        if not pages or not metadata:
            print(f"Skipping {p} due to parsing error.")
            continue

        # Build full doc text from parsed pages
        full_text = "\n".join([txt for (_pg, txt) in pages])

        # Use your existing chunk_pages helper (adds chunk_uuid, doc_id, filename)
        chunks = chunk_pages(
            pages=pages,
            meta_doc_id=metadata["doc_id"],
            filename=metadata.get("filename", p.name)
        )
        if not chunks:
            print(f"Skipping {p} due to chunking error.")
            continue

        # Persist document + chunks in SQLite
        storage.add_document(metadata, full_text)
        storage.add_chunks(chunks)
        all_chunks.extend(chunks)

        # Extract entities & relations from the enriched chunks
        # (extractor attaches source_id = chunk_uuid, and filepath)
        res = extract_from_chunks(chunks)

        # Update canonical KG (deduped by schema)
        for ent in res["entities"]:
            storage.add_kg_node(**ent)
        for rel in res["relationships"]:
            storage.add_kg_edge(**rel)
        # Merge+summarize entity descriptions at the KG node level (no change to vectors/mentions)
        lang = metadata.get("language") or metadata.get("Language") or PROMPTS["DEFAULT_LANGUAGE"]
        merged = _merge_and_maybe_summarize_entity_descriptions(
            res["entities"],
            language=lang,
            max_chars=DEFAULT_MAX_ENTITY_DESC_CHARS,
        )
        if merged:
            # Batch upsert to minimize disk I/O
            updates = [{"name": n, "type": t, "description": d} for (n, t), d in merged.items()]
            storage.upsert_nodes(updates)


        # Collect for vector stage
        all_entities.extend(res["entities"])
        all_relations.extend(res["relationships"])

    # Vector stage
    if all_chunks:
        # 1) Add chunk vectors (id = chunk_uuid)
        storage.add_chunk_vectors(all_chunks)

        # 2) Mention-based vectors for entities & relations
        #    Group by chunk (source_id == chunk_uuid), enumerate per chunk
        chunk_uuid_to_doc: Dict[str, str] = {c["chunk_uuid"]: c.get("doc_id", "") for c in all_chunks}

        ents_by_chunk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in all_entities:
            cid = e.get("source_id")
            if cid:
                ents_by_chunk[cid].append(e)

        rels_by_chunk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in all_relations:
            cid = r.get("source_id")
            if cid:
                rels_by_chunk[cid].append(r)

        # Entities → one vector per mention (unique mention_id inside Storage)
        for chunk_uuid, ents in ents_by_chunk.items():
            doc_id = chunk_uuid_to_doc.get(chunk_uuid, "")
            for i, e in enumerate(ents):
                storage.add_entity_mention_vector(
                    name=e["name"],
                    typ=e["type"],
                    text=e.get("description") or "",
                    doc_id=doc_id,
                    chunk_id=chunk_uuid,
                    mention_idx=i,
                    page=None,
                    extra_meta={"filepath": e.get("filepath")}
                )

        # Relations → one vector per mention (direction preserved by Storage implementation)
        for chunk_uuid, rels in rels_by_chunk.items():
            doc_id = chunk_uuid_to_doc.get(chunk_uuid, "")
            for j, r in enumerate(rels):
                storage.add_edge_mention_vector(
                    src=r["source_name"],
                    dst=r["target_name"],
                    text=r.get("description") or "",
                    doc_id=doc_id,
                    chunk_id=chunk_uuid,
                    rel_idx=j,
                    page=None,
                    extra_meta={
                        "keywords": r.get("keywords"),
                        "weight": r.get("weight"),
                        "filepath": r.get("filepath"),
                    }
                )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./storage/file_chunk_store.sqlite")
    ap.add_argument("--chroma_dir", default="./storage/chroma_db")
    ap.add_argument("--collection", default="chunks")
    args = ap.parse_args()

    root = Path('docs')
    paths = FileParser(root).filepaths
    if not paths:
        print("No files to ingest.")
        return
    ingest_paths(paths, db_path=args.db, chroma_dir=args.chroma_dir, collection=args.collection)

if __name__ == "__main__":
    main()

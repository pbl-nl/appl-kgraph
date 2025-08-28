
from __future__ import annotations
import argparse
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple
from storage import Storage
from fileparser import FileParser
from chunker import chunk_parsed_pages
from extractor import extract_from_chunks


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


def ingest_paths(paths: List[Path], db_path: str, chroma_dir: str, collection: str):
    

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
        storage.add_document(metadata, full_text)
        storage.add_chunks(chunks)
        all_chunks.extend(chunks)

        # Extract entities and relations from chunks
        # res['entities'], res['relationships'], res['content_keywords']
        res = extract_from_chunks(chunks) 
        for ent in res['entities']:
            storage.add_kg_node(**ent)
        for rel in res['relationships']:
            storage.add_kg_edge(**rel)

        all_entities.extend(res['entities'])
        all_relations.extend(res['relationships'])

    # Finally, add all chunks, entities, and relations to vector DB
    if all_chunks:
        storage.add_chunk_vectors(all_chunks)
        storage.add_entity_vectors(all_entities)
        storage.add_relation_vectors(all_relations)

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

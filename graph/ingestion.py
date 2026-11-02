from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Sequence, Optional, Union
# local imports
from db_storage import Storage
from graph_pickle import save_graph_to_pickle, build_graph_from_storage
from settings import VALID_EXTENSIONS, settings
from logging_utils import configure_file_logger
from project_paths import (
    ProjectPaths,
    ensure_project_dirs,
    list_document_paths,
    resolve_project_paths,
)
import hashlib


LOGGER = configure_file_logger(
    "appl_kgraph.ingestion",
    log_file=Path("ingestion.log"),
    level=settings.logging.internal_log_level,
    enabled=settings.logging.internal_logging_enabled,
)


from production_ingestion import create_ingestion_pipeline, remove_document_from_storage, PENDING_CONTENT_HASH_PREFIX as PENDING_CONTENT_HASH_PREFIX
from ingestion_pipeline import ingest_documents, IngestionStageError
from document_parser import DocumentParseError
from discovery import is_temporary_document
from schemas import DocumentRef, IngestionSummary

#--------------------------------------------------
# Helpers 
#--------------------------------------------------

def file_sha256(p: Path, chunk_size=1024*1024) -> str:
    """
    Computes the SHA-256 hash of a file.

    Args:
        p (Path): Path to the file.
        chunk_size (int, optional): Size of chunks to read at a time. Defaults to 1MB.

    Returns:
        str: The hexadecimal SHA-256 hash of the file.
    """
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def should_skip_ingestion(storage: Storage, p: Path, content_hash: Optional[str] = None) -> bool:
    """
    Determines whether a file should be skipped during ingestion based on content hash.

    Args:
        storage (Storage): The storage instance to query.
        p (Path): Path to the file to check.
        content_hash (Optional[str], optional): Pre-computed content hash. If None, will be computed.

    Returns:
        bool: True if file has already been ingested with the same content, False otherwise.
    """
    doc = storage.get_document_by_filename(p.name)
    if not doc:
        return False
    ch = content_hash or file_sha256(p)
    if doc.get("content_hash") != ch:
        return False

    doc_id = doc.get("doc_id")
    if not doc_id:
        return False

    chunks = storage.get_chunks_by_doc_id(doc_id)
    if not chunks:
        return False

    # On environments where native vector mutations are disabled, chunk/entity/relation
    # vectors are not persisted in Chroma and SQL-backed retrieval is used instead.
    if getattr(storage, "_vector_mutations_disabled", False):
        return True

    for chunk in chunks:
        chunk_uuid = chunk.get("chunk_uuid")
        if not chunk_uuid:
            return False
        if not storage.get_chunk_vector(chunk_uuid):
            return False

    return True

# Core functions ------------------------------------

def _resolve_runtime_project_paths(
    *,
    paths: Sequence[Path],
    documents_root: Optional[Union[Path, str]] = None,
    project_paths: Optional[ProjectPaths] = None,
) -> Optional[ProjectPaths]:
    if project_paths is not None:
        return project_paths
    if documents_root is not None:
        return resolve_project_paths(documents_root)
    return None


def _configure_ingestion_logger(project_paths: Optional[ProjectPaths]) -> None:
    log_file = (
        project_paths.ingestion_log_file
        if project_paths is not None
        else Path("ingestion.log")
    )
    configure_file_logger(
        "appl_kgraph.ingestion",
        log_file=log_file,
        level=settings.logging.internal_log_level,
        enabled=settings.logging.internal_logging_enabled,
    )


def _write_extraction_diagnostics(
    project_paths: Optional[ProjectPaths],
    filename: str,
    validation_results: List[Dict[str, Any]],
) -> None:
    if (
        project_paths is None
        or not validation_results
        or not settings.logging.internal_logging_enabled
    ):
        return
    ensure_project_dirs(project_paths)
    target = project_paths.extraction_diagnostics_dir / f"{Path(filename).name}.validation.json"
    target.write_text(
        json.dumps(validation_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_retrieval_graph_snapshot(
    storage: Storage,
    project_paths: Optional[ProjectPaths],
) -> Optional[Path]:
    if project_paths is None:
        return None
    ensure_project_dirs(project_paths)
    graph = build_graph_from_storage(storage, logger=LOGGER)
    return save_graph_to_pickle(graph, project_paths.retrieval_graph_pickle_file)

def ingest_paths(
    paths: List[Path], *, documents_root=None, project_paths=None, storage_paths=None,
    prune_missing_documents=True, progress_callback=None,
) -> Dict[str, Any]:
    """Collection lifecycle around the production typed ingestion pipeline."""
    paths = [Path(path) for path in paths]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Duplicate document paths")
    # Current database identity is filename-based. Refuse collisions before pruning.
    if len({path.name.casefold() for path in paths}) != len(paths):
        raise ValueError("Duplicate document filenames are not supported in one collection")

    def report(message):
        LOGGER.info(message)
        if progress_callback is not None and settings.logging.verbosity_enabled:
            progress_callback(message)

    active = _resolve_runtime_project_paths(paths=paths, documents_root=documents_root, project_paths=project_paths)
    if active is not None:
        ensure_project_dirs(active)
    _configure_ingestion_logger(active)
    report("Initializing project storage")
    storage = Storage(paths=active.storage if active else storage_paths)
    processed = []
    skipped = removed = 0
    try:
        storage.init()
        if active is not None:
            active.retrieval_graph_pickle_file.unlink(missing_ok=True)
        pipeline = create_ingestion_pipeline(
            storage, active, remove_existing=remove_document_from_storage,
            diagnostics=lambda ref, result: _write_extraction_diagnostics(active, ref.path.name, result.diagnostics),
        )
        report(f"Queued {len(paths)} files for ingestion")
        if prune_missing_documents:
            keep = {path.name for path in paths}
            for doc in storage.get_all_documents():
                if doc.get("filename") and doc["filename"] not in keep:
                    report(f"Removing stale document {doc['filename']}")
                    remove_document_from_storage(storage, doc["filename"])
                    removed += 1
        for index, path in enumerate(paths, 1):
            prefix = f"{index}/{len(paths)} {path.name}"
            if not path.is_file() or is_temporary_document(path) or path.suffix.lower() not in VALID_EXTENSIONS:
                skipped += 1
                report(f"{prefix} - skipped (missing, temporary or unsupported)")
                continue
            content_hash = file_sha256(path)
            if should_skip_ingestion(storage, path, content_hash):
                skipped += 1
                report(f"{prefix} - skipped (unchanged)")
                continue
            ref = DocumentRef(path=path, root=active.documents_root if active else None,
                              metadata={"content_hash": content_hash})
            try:
                summary = ingest_documents([ref], pipeline,
                    progress=lambda stage, ref: report(f"{prefix} - {stage}"))
            except IngestionStageError as exc:
                if exc.stage == "parse" and isinstance(exc.__cause__, DocumentParseError):
                    skipped += 1
                    report(f"{prefix} - parsing failed: {exc.__cause__}")
                    continue
                raise
            processed.extend(summary.documents)
            report(f"{prefix} - completed")
        snapshot = _write_retrieval_graph_snapshot(storage, active)
        totals = IngestionSummary(documents=tuple(processed))
        report("Completed ingestion")
        return {
            "documents_root": str(active.documents_root) if active else None,
            "project_root": str(active.project_root) if active else None,
            "retrieval_graph_pickle": str(snapshot) if snapshot else None,
            "processed_files": totals.processed_files, "skipped_files": skipped, "removed_files": removed,
            "chunk_count": totals.chunk_count, "entity_count": totals.entity_count, "relation_count": totals.relation_count,
        }
    finally:
        storage.close()


def main():
    folder_input = input(
        "Source folder path of documents (leave blank for ./docs): "
    ).strip()
    root = Path(folder_input) if folder_input else Path("docs")
    paths = list_document_paths(root)
    if not paths:
        print(
            f"No supported files found in {root.resolve()} "
            f"with extensions: {', '.join(VALID_EXTENSIONS)}."
        )
        return
    ingest_paths(paths, documents_root=root)

if __name__ == "__main__":
    main()

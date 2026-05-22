from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

from settings import VALID_EXTENSIONS, StoragePaths, settings


@dataclass(frozen=True)
class ProjectPaths:
    documents_root: Path
    project_root: Path
    storage_root: Path
    knowledge_graph_dir: Path
    graph_pickle_file: Path
    retrieval_graph_pickle_file: Path
    logs_dir: Path
    qa_logs_dir: Path
    audits_dir: Path
    extraction_audits_dir: Path
    storage: StoragePaths
    ingestion_log_file: Path
    pathrag_log_file: Path
    lightrag_log_file: Path


def resolve_project_paths(documents_root: Union[Path, str]) -> ProjectPaths:
    root = Path(documents_root).expanduser().resolve()
    project_root = root / settings.project.artifacts_dirname
    storage_root = project_root / settings.project.storage_dirname
    knowledge_graph_dir = project_root / "knowledge_graph"
    logs_dir = project_root / settings.project.logs_dirname
    qa_logs_dir = logs_dir / settings.project.qa_logs_dirname
    audits_dir = project_root / settings.project.audits_dirname
    extraction_audits_dir = audits_dir / settings.project.extraction_audits_dirname

    storage = StoragePaths(
        documents_db=str(storage_root / "documents.sqlite"),
        chunks_db=str(storage_root / "chunks.sqlite"),
        graph_db=str(storage_root / "graph.sqlite"),
        chroma_chunks=str(storage_root / "chroma_chunks"),
        chroma_entities=str(storage_root / "chroma_entities"),
        chroma_relations=str(storage_root / "chroma_relations"),
    )

    return ProjectPaths(
        documents_root=root,
        project_root=project_root,
        storage_root=storage_root,
        knowledge_graph_dir=knowledge_graph_dir,
        graph_pickle_file=knowledge_graph_dir / "kg.pkl",
        retrieval_graph_pickle_file=knowledge_graph_dir / "kg_retrieval.pkl",
        logs_dir=logs_dir,
        qa_logs_dir=qa_logs_dir,
        audits_dir=audits_dir,
        extraction_audits_dir=extraction_audits_dir,
        storage=storage,
        ingestion_log_file=logs_dir / "ingestion.log",
        pathrag_log_file=logs_dir / "pathrag.log",
        lightrag_log_file=logs_dir / "lightrag.log",
    )


def ensure_project_dirs(project_paths: ProjectPaths) -> None:
    for path in (
        project_paths.project_root,
        project_paths.storage_root,
        project_paths.knowledge_graph_dir,
        project_paths.logs_dir,
        project_paths.qa_logs_dir,
        project_paths.audits_dir,
        project_paths.extraction_audits_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def list_document_paths(
    documents_root: Union[Path, str],
    *,
    valid_extensions: Optional[Iterable[str]] = None,
) -> List[Path]:
    root = Path(documents_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    allowed = {ext.lower() for ext in (valid_extensions or VALID_EXTENSIONS)}
    project_root = root / settings.project.artifacts_dirname
    paths: List[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if project_root in path.parents:
            continue
        if path.suffix.lower() not in allowed:
            continue
        paths.append(path)

    return sorted(paths, key=lambda item: str(item).lower())

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from db_storage import DocumentsDB
from graph_pickle import load_graph_from_pickle
from project_paths import resolve_project_paths
from settings import settings


_DOCS_ROOT = Path(__file__).resolve().parents[1] / settings.project.documents_root_dirname


def discover_existing_graph_choices() -> List[Tuple[str, str]]:
    if not _DOCS_ROOT.exists() or not _DOCS_ROOT.is_dir():
        return []

    choices: List[Tuple[str, str]] = []
    pattern = f"*/{settings.project.artifacts_dirname}/knowledge_graph/kg.pkl"
    for kg_pickle in sorted(_DOCS_ROOT.glob(pattern), key=lambda p: str(p).lower()):
        docs_folder = kg_pickle.parents[2]
        rel_folder = docs_folder.relative_to(_DOCS_ROOT)
        label = rel_folder.parts[0] if rel_folder.parts else docs_folder.name
        choices.append((label, str(docs_folder)))
    return choices


def stored_document_names_from_database(folder_path: str) -> List[str]:
    if not folder_path:
        return []

    project_paths = resolve_project_paths(folder_path)
    documents_db_path = Path(project_paths.storage.documents_db)
    if not documents_db_path.exists():
        return []

    names: set[str] = set()
    try:
        rows = DocumentsDB(str(documents_db_path)).list_documents()
        for row in rows:
            filename = str(row.get("filename", "") or "").strip()
            if filename:
                names.add(filename)
    except Exception:
        return []

    return sorted(names, key=str.lower)


def stored_document_names(folder_path: str) -> List[str]:
    names: set[str] = set(stored_document_names_from_database(folder_path))

    if not names:
        project_paths = resolve_project_paths(folder_path)
        if project_paths.graph_pickle_file.exists():
            graph = load_graph_from_pickle(project_paths.graph_pickle_file)
            for _, data in graph.nodes(data=True):
                raw_filepaths = str(data.get("filepath", "") or "")
                for raw_path in raw_filepaths.split("||"):
                    token = raw_path.strip()
                    if token:
                        names.add(Path(token).name)

    return sorted(names, key=str.lower)

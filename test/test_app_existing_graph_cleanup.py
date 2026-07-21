import os
import sys
from pathlib import Path

import networkx as nx


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from db_storage import DocumentsDB
from existing_graphs import stored_document_names, stored_document_names_from_database
from graph_pickle import save_graph_to_pickle
from project_paths import ensure_project_dirs, resolve_project_paths


def test_database_names_do_not_fall_back_to_pickle(tmp_path):
    docs_folder = tmp_path / "docs" / "project-one"
    docs_folder.mkdir(parents=True)

    project_paths = resolve_project_paths(docs_folder)
    ensure_project_dirs(project_paths)

    documents_db = DocumentsDB(project_paths.storage.documents_db)
    documents_db.init()

    graph = nx.Graph()
    graph.add_node(
        "n1",
        label="Node 1",
        type="entity",
        filepath=str(docs_folder / "alpha.txt"),
    )
    saved_path = save_graph_to_pickle(graph, project_paths.graph_pickle_file)
    assert saved_path == project_paths.graph_pickle_file

    assert stored_document_names(str(docs_folder)) == ["alpha.txt"]
    assert stored_document_names_from_database(str(docs_folder)) == []

import os
import sys
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from graph.project_paths import list_document_paths, resolve_project_paths


def test_resolve_project_paths_nests_artifacts_under_documents_root(tmp_path):
    documents_root = tmp_path / "docs"
    documents_root.mkdir()

    project_paths = resolve_project_paths(documents_root)

    assert project_paths.documents_root == documents_root.resolve()
    assert project_paths.project_root == documents_root.resolve() / ".appl-kgraph"
    assert Path(project_paths.storage.documents_db).parent == project_paths.storage_root
    assert project_paths.qa_logs_dir.parent == project_paths.logs_dir
    assert project_paths.graph_pickle_file == project_paths.knowledge_graph_dir / "kg.pkl"
    assert project_paths.retrieval_graph_pickle_file == project_paths.knowledge_graph_dir / "kg_retrieval.pkl"


def test_list_document_paths_excludes_project_artifacts(tmp_path):
    documents_root = tmp_path / "docs"
    documents_root.mkdir()
    (documents_root / "a.txt").write_text("alpha", encoding="utf-8")
    (documents_root / "b.md").write_text("beta", encoding="utf-8")

    project_root = documents_root / ".appl-kgraph"
    project_root.mkdir()
    (project_root / "ignored.txt").write_text("ignore me", encoding="utf-8")

    paths = list_document_paths(documents_root)

    assert [path.name for path in paths] == ["a.txt", "b.md"]

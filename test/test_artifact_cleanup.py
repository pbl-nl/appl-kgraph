import os
import sys
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from artifact_cleanup import cleanup_graph_artifacts_best_effort
from project_paths import ensure_project_dirs, resolve_project_paths


def test_cleanup_empties_storage_and_preserves_other_artifacts(tmp_path):
    docs_folder = tmp_path / "docs" / "project-one"
    docs_folder.mkdir(parents=True)

    project_paths = resolve_project_paths(docs_folder)
    ensure_project_dirs(project_paths)

    # Seed files across artifact subfolders.
    (project_paths.knowledge_graph_dir / "kg.pkl").write_text("graph", encoding="utf-8")
    (project_paths.storage_root / "documents.sqlite").write_text("db", encoding="utf-8")
    (project_paths.storage_root / "chroma_chunks").mkdir(parents=True, exist_ok=True)
    (project_paths.storage_root / "chroma_chunks" / "chunk.bin").write_text("chunk", encoding="utf-8")

    (project_paths.logs_dir / "ingestion.log").write_text("log", encoding="utf-8")
    (project_paths.diagnostics_dir / "diag.txt").write_text("diag", encoding="utf-8")

    error = cleanup_graph_artifacts_best_effort(project_paths)

    assert error is None
    assert not project_paths.knowledge_graph_dir.exists()
    assert project_paths.storage_root.exists()
    assert project_paths.storage_root.is_dir()
    assert list(project_paths.storage_root.iterdir()) == []

    # Non-storage artifact folders remain untouched.
    assert (project_paths.logs_dir / "ingestion.log").exists()
    assert (project_paths.diagnostics_dir / "diag.txt").exists()

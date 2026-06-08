import json
import os
import sys
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from project_paths import resolve_project_paths
from query_logging import write_audit_log, write_query_log


def test_write_audit_log_records_reproducibility_payload(tmp_path):
    documents_root = tmp_path / "docs"
    documents_root.mkdir()
    project_paths = resolve_project_paths(documents_root)

    target = write_audit_log(
        project_paths=project_paths,
        retriever_name="pathrag",
        payload={
            "question": "What changed?",
            "answer": "The audit writer was renamed.",
            "conversation_history": [("hello", "hi")],
            "model": {"name": "test-model"},
        },
    )

    assert target is not None
    assert target.parent == project_paths.audit_logs_dir
    body = json.loads(target.read_text(encoding="utf-8"))
    assert body["retriever"] == "pathrag"
    assert body["question"] == "What changed?"
    assert body["answer"] == "The audit writer was renamed."
    assert body["conversation_history"] == [["hello", "hi"]]
    assert body["model"] == {"name": "test-model"}


def test_write_query_log_is_deprecated_alias(tmp_path):
    documents_root = tmp_path / "docs"
    documents_root.mkdir()
    project_paths = resolve_project_paths(documents_root)

    target = write_query_log(
        project_paths=project_paths,
        retriever_name="lightrag",
        payload={"question": "Alias?"},
    )

    assert target is not None
    assert target.parent == project_paths.audit_logs_dir

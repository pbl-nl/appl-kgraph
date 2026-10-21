import json
import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from project_paths import resolve_project_paths
import query_logging
from query_logging import build_audit_settings_snapshot, write_audit_log


def test_write_audit_log_records_reproducibility_payload(tmp_path, llm_provider):
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
    assert body["settings"]["provider"]["provider"] == llm_provider
    assert "openai_api_key" not in json.dumps(body["settings"]).lower()


def test_audit_log_respects_audit_gate(tmp_path, monkeypatch):
    documents_root = tmp_path / "docs"
    documents_root.mkdir()
    project_paths = resolve_project_paths(documents_root)

    monkeypatch.setattr(
        query_logging,
        "settings",
        SimpleNamespace(logging=SimpleNamespace(audit_enabled=False)),
    )

    target = query_logging.write_audit_log(
        project_paths=project_paths,
        retriever_name="lightrag",
        payload={"question": "Disabled?"},
    )

    assert target is None
    assert not any(project_paths.audit_logs_dir.glob("*.json"))


def test_audit_settings_snapshot_excludes_secret_provider_fields(llm_provider):
    snapshot = build_audit_settings_snapshot()

    assert "api_key" not in json.dumps(snapshot).lower()
    assert snapshot["provider"] == {
        "provider": llm_provider,
        "llm_model": f"test-{llm_provider}-chat",
        "embeddings_model": f"test-{llm_provider}-embedding",
        "azure_api_version": "2024-02-15-preview" if llm_provider == "azure" else None,
    }
    assert "test-openai-key" not in json.dumps(snapshot)
    assert "test-azure-key" not in json.dumps(snapshot)

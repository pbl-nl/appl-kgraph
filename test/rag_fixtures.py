from dataclasses import replace

import pytest

from test.production_fixtures import (
    configure_ingestion,
    DeterministicEmbedder,
    DeterministicChat,
)
import ingestion
from db_storage import Storage
from project_paths import resolve_project_paths


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    configure_ingestion(monkeypatch)
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("Alpha works with Beta. Evidence from a.")
    b.write_text("Alpha works with Beta. Evidence from b.")
    ingestion.ingest_paths([a, b], documents_root=tmp_path)
    return resolve_project_paths(tmp_path)


def configure_rag(module, monkeypatch):
    monkeypatch.setattr(
        module, "Storage", lambda paths=None: Storage(paths, DeterministicEmbedder())
    )
    monkeypatch.setattr(module, "RetrieveChat", DeterministicChat)
    monkeypatch.setattr(
        module,
        "settings",
        replace(
            module.settings,
            retrieval=replace(module.settings.retrieval, enable_rerank=False),
        ),
    )
    DeterministicChat.calls = []


def observe_stages(module, monkeypatch):
    calls = []
    for name in ("retrieve_candidates", "build_context", "rerank_context"):
        original = getattr(module.stages, name)

        def wrapped(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module.stages, name, wrapped)
    original_answer = module.generate_answer_async

    async def answer(*args, **kwargs):
        calls.append("generate_answer")
        return await original_answer(*args, **kwargs)

    monkeypatch.setattr(module, "generate_answer_async", answer)
    original_audit = module.record_chat_audit

    def audit(*args, **kwargs):
        calls.append("audit")
        return original_audit(*args, **kwargs)

    monkeypatch.setattr(module, "record_chat_audit", audit)
    return calls

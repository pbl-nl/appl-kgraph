import os
import sys
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")
sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from discovery import discover_documents
from schemas import DocumentRef


def test_discover_documents_returns_supported_refs_in_stable_order(tmp_path):
    documents_root = tmp_path / "documents"
    nested = documents_root / "nested"
    artifacts = documents_root / ".appl-kgraph"
    nested.mkdir(parents=True)
    artifacts.mkdir()
    (documents_root / "b.md").write_text("beta", encoding="utf-8")
    (nested / "A.TXT").write_text("alpha", encoding="utf-8")
    (nested / "ignored.csv").write_text("ignored", encoding="utf-8")
    (artifacts / "hidden.pdf").write_text("hidden", encoding="utf-8")

    documents = discover_documents(documents_root)

    assert documents == [
        DocumentRef(path=(documents_root / "b.md").resolve(), root=documents_root.resolve()),
        DocumentRef(path=(nested / "A.TXT").resolve(), root=documents_root.resolve()),
    ]

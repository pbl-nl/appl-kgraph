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


def test_discover_documents_returns_empty_for_missing_or_non_directory_roots(tmp_path):
    missing = tmp_path / "missing"
    file_root = tmp_path / "file.txt"
    file_root.write_text("not a directory", encoding="utf-8")

    assert discover_documents(missing) == []
    assert discover_documents(file_root) == []


def test_discover_documents_honors_case_insensitive_custom_extensions(tmp_path):
    (tmp_path / "data.CsV").write_text("data", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")

    custom = discover_documents(tmp_path, valid_extensions=[".CSV"])

    assert [document.path.name for document in custom] == ["data.CsV"]
    assert discover_documents(tmp_path, valid_extensions=[]) == []


def test_discover_documents_keeps_duplicate_filenames_in_nested_paths(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "report.txt").write_text("first", encoding="utf-8")
    (second_dir / "report.txt").write_text("second", encoding="utf-8")

    documents = discover_documents(tmp_path)

    assert [document.path.relative_to(tmp_path.resolve()) for document in documents] == [
        Path("first/report.txt"),
        Path("second/report.txt"),
    ]


def test_discover_documents_excludes_only_known_temporary_file_patterns(tmp_path):
    filenames = [
        "~$draft.docx",
        "~$legacy.doc",
        "word-recovery.tmp",
        "word-recovery.temp",
        "report.docx",
        "word.md",
        "~$notes.txt",
    ]
    for filename in filenames:
        (tmp_path / filename).write_text("content", encoding="utf-8")

    documents = discover_documents(
        tmp_path,
        valid_extensions=[".docx", ".doc", ".tmp", ".temp", ".md", ".txt"],
    )

    assert [document.path.name for document in documents] == [
        "report.docx",
        "word.md",
        "~$notes.txt",
    ]

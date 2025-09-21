import os
import sys
import types
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.modules.setdefault("fitz", types.ModuleType("fitz"))

langdetect_stub = types.ModuleType("langdetect")


class _DummyLangDetectException(Exception):
    pass


langdetect_stub.detect = lambda text: "en"
langdetect_stub.LangDetectException = _DummyLangDetectException
sys.modules.setdefault("langdetect", langdetect_stub)

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from ingestion import _resolve_type, dedupe_entities_for_vectors


def test_resolve_type_prefers_non_unknown():
    votes = Counter({"unknown": 1, "Person": 1})
    result = _resolve_type(votes, "unknown")
    assert result == "Person"


def test_resolve_type_uses_majority_ignoring_unknown():
    votes = Counter({"Person": 2, "Organization": 1, "unknown": 5})
    result = _resolve_type(votes, "Organization")
    assert result == "Person"


def test_resolve_type_prefers_existing_on_tie():
    votes = Counter({"Person": 1, "Organization": 1})
    result = _resolve_type(votes, "Organization")
    assert result == "Organization"


def test_dedupe_entities_prefers_typed_over_unknown():
    entities = [
        {"name": "Alice", "type": "unknown", "description": "from edge"},
        {"name": "Alice", "type": "Person", "description": "from extraction"},
        {"name": "Bob", "type": "Company"},
        {"name": "Bob", "type": "unknown"},
    ]

    deduped = dedupe_entities_for_vectors(entities)

    assert len(deduped) == 2
    names_to_types = {e["name"]: e.get("type") for e in deduped}
    assert names_to_types["Alice"] == "Person"
    assert names_to_types["Bob"] == "Company"

from pathlib import Path


def test_lightrag_uses_top_k_chunk_per_entity_setting():
    source = (
        Path(__file__).resolve().parent.parent / "graph" / "lightrag.py"
    ).read_text(encoding="utf-8")

    assert "top_k_chunk_per_entity=settings.retrieval.top_k_chunk_per_entity" in source
    assert "top_k_chunk_per_entity=3" not in source

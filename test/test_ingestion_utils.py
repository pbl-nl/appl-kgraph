from dataclasses import replace
import pytest

from test.production_fixtures import configure_ingestion
import ingestion
from schemas import Relation, ExtractionResult
from graph_normalization import GraphView, normalize_graph, remove_graph_sources


def test_placeholder_provenance_and_shared_source_removal():
    extraction = ExtractionResult(
        relations=[
            Relation(
                "Alpha",
                "Beta",
                "Related",
                source_ids=("one", "two"),
                filepath="a.txt||b.txt",
            )
        ]
    )
    added = normalize_graph(extraction, GraphView())
    assert all(node.source_ids == ("one", "two") for node in added.entities)
    removed = remove_graph_sources(
        GraphView(added.entities, added.relations), ["one"], filepath="a.txt"
    )
    assert all(
        node.source_ids == ("two",) and node.filepath == "b.txt"
        for node in removed.entities
    )
    assert removed.deleted_entity_names == ()
    final = remove_graph_sources(
        GraphView(removed.entities, removed.relations), ["two"], filepath="b.txt"
    )
    assert set(final.deleted_entity_names) == {"Alpha", "Beta"}
    assert final.deleted_relation_pairs == (("Alpha", "Beta"),)


@pytest.mark.storage_integration
def test_canonical_snapshots_survive_disabled_logging_and_progress(
    tmp_path, monkeypatch
):
    configure_ingestion(monkeypatch)
    logging = replace(
        ingestion.settings.logging,
        internal_logging_enabled=False,
        verbosity_enabled=False,
    )
    monkeypatch.setattr(
        ingestion, "settings", replace(ingestion.settings, logging=logging)
    )
    document = tmp_path / "a.txt"
    document.write_text("Alpha.")
    messages = []
    result = ingestion.ingest_paths(
        [document], documents_root=tmp_path, progress_callback=messages.append
    )
    assert messages == []
    from project_paths import resolve_project_paths

    paths = resolve_project_paths(tmp_path)
    assert (paths.raw_documents_dir / "a.txt.raw.json").exists()
    assert (paths.raw_documents_dir / "a.txt.enriched.json").exists()
    assert not list(paths.extraction_diagnostics_dir.glob("*.json"))
    assert result["processed_files"] == 1

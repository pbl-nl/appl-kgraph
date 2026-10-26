import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import networkx as nx
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

import lightrag
import pathrag
from app import _doc_names_from_node_data
from existing_graphs import stored_document_names
from graph_pickle import save_graph_to_pickle
from ingestion import remove_document_from_storage
from path_utils import stored_path_name
from project_paths import resolve_project_paths


@pytest.mark.parametrize("value, expected", [
    (r"C:\Users\Alice\documents\report.txt", "report.txt"),
    (r"\\server\share\report.txt", "report.txt"),
    ("C:/Users/Alice/documents/report.txt", "report.txt"),
    ("/home/alice/documents/report.txt", "report.txt"),
    (r"documents\report.txt", "report.txt"),
    ("documents/report.txt", "report.txt"),
    ("/home/alice/résumé with spaces.txt", "résumé with spaces.txt"),
    (r"/home/alice/literal\name.txt", r"literal\name.txt"),
    ("", ""),
])
def test_stored_path_name_ignores_host_platform(value, expected):
    assert stored_path_name(value) == expected


@pytest.mark.parametrize("module", [lightrag, pathrag])
def test_retrieval_filters_accept_paths_from_either_os(module):
    raw = r"C:\docs\alpha.txt||/home/user/docs/beta.txt"
    assert module._filepath_matches_allowed(raw, {"alpha.txt"})
    assert module._filepath_matches_allowed(raw, {"beta.txt"})
    assert not module._filepath_matches_allowed(raw, {"other.txt"})
    chunk = dict(filename="", document_id=r"C:\docs\alpha.txt")
    if module is pathrag:
        chunk = SimpleNamespace(**chunk)
    assert module._chunk_matches_allowed(chunk, {"alpha.txt"})
    assert not module._chunk_matches_allowed(chunk, {"other.txt"})


def test_moved_graph_displays_windows_and_linux_document_names(tmp_path):
    data = {"filepath": r"C:\docs\alpha.txt||/home/user/docs/beta.txt"}
    graph = nx.Graph()
    graph.add_node("shared", **data)
    paths = resolve_project_paths(tmp_path)
    save_graph_to_pickle(graph, paths.graph_pickle_file)
    assert stored_document_names(str(tmp_path)) == ["alpha.txt", "beta.txt"]
    assert _doc_names_from_node_data(data) == {"alpha.txt", "beta.txt"}


def test_removing_document_prunes_its_old_windows_path():
    storage = Mock()
    storage.get_document_by_filename.return_value = {
        "doc_id": "removed", "filepath": "/new/location/alpha.txt",
    }
    storage.get_chunks_by_doc_id.side_effect = lambda doc_id: [
        {"chunk_uuid": "old" if doc_id == "removed" else "keep"},
    ]
    metadata = {
        "source_id": "old||keep",
        "filepath": r"C:\old\alpha.txt||/new/location/beta.txt",
    }
    storage.get_nodes_by_chunk_uuids.return_value = [{"name": "shared", **metadata}]
    storage.get_edges_by_chunk_uuids.return_value = [
        {"source_name": "shared", "target_name": "other", **metadata},
    ]
    storage.get_all_documents.return_value = [{"doc_id": "remaining"}]
    storage.graphdb.list_nodes.return_value = []
    storage.graphdb.list_edges.return_value = []

    remove_document_from_storage(storage, "alpha.txt")

    for update in (storage.graphdb.update_nodes, storage.graphdb.update_edges):
        rows = update.call_args.args[0]
        assert rows[0]["source_id"] == "keep"
        assert rows[0]["filepath"] == "/new/location/beta.txt"
    storage.delete_document.assert_called_once_with("removed")

import importlib
from types import SimpleNamespace

import networkx as nx
import pytest

from path_utils import source_filename


@pytest.mark.parametrize("path", [
    "/home/user/document folder/café.txt",
    r"C:\Users\user\document folder\café.txt",
    "C:/Users/user/document folder/café.txt",
    r"\\server\share\document folder\café.txt",
    "café.txt",
])
def test_metadata_filename_accepts_both_path_styles(path):
    assert source_filename(path) == "café.txt"


@pytest.mark.parametrize("module_name", ["pathrag", "lightrag"])
@pytest.mark.parametrize("path", [r"C:\docs\Report.TXT", "/home/user/docs/Report.TXT", r"\\server\share\Report.TXT"])
def test_retrieval_filters_accept_paths_from_other_platforms(module_name, path):
    module = importlib.import_module(module_name)
    assert module._filepath_matches_allowed(f"/docs/other.txt||{path}", {"report.txt"})
    assert not module._filepath_matches_allowed(path, {"other.txt"})
    for field in ("filename", "document_id"):
        chunk = {"filename": "", "document_id": "", field: path}
        if module_name == "pathrag":
            chunk = SimpleNamespace(**chunk)
        assert module._chunk_matches_allowed(chunk, {"report.txt"})
        assert not module._chunk_matches_allowed(chunk, {"other.txt"})


def test_graph_document_selection_accepts_windows_and_linux_paths(tmp_path):
    from app import _doc_names_from_node_data, _graph_filtered_to_documents
    from existing_graphs import stored_document_names
    from graph_pickle import save_graph_to_pickle
    from project_paths import resolve_project_paths

    graph = nx.Graph()
    graph.add_node("windows", filepath=r"C:\docs\report.txt")
    graph.add_node("linux", filepath="/home/user/docs/other.txt")
    assert _doc_names_from_node_data({"filepath": r"C:\docs\report.txt||/docs/other.txt"}) == {"report.txt", "other.txt"}
    assert set(_graph_filtered_to_documents(graph, ["report.txt"])) == {"windows"}
    save_graph_to_pickle(graph, resolve_project_paths(tmp_path).graph_pickle_file)
    assert stored_document_names(str(tmp_path)) == ["other.txt", "report.txt"]


def test_graph_rendering_supports_unicode_without_writing_files(tmp_path, monkeypatch):
    from app import render_graph_iframe

    monkeypatch.chdir(tmp_path)
    graph = nx.Graph()
    graph.add_node("Café", description="Résumé", filepath=r"C:\docs\café.txt")
    result = render_graph_iframe(graph)
    assert result.startswith("<iframe")
    assert "</iframe>" in result
    assert list(tmp_path.iterdir()) == []

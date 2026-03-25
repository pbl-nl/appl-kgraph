import os
import sys
from pathlib import Path

import networkx as nx


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

from graph.graph_pickle import (
    load_graph_from_pickle,
    load_or_build_graph_snapshot,
    save_graph_to_pickle,
)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, node_rows, edge_rows):
        self._node_rows = node_rows
        self._edge_rows = edge_rows

    def execute(self, sql):
        if "FROM nodes" in sql:
            return _FakeCursor(self._node_rows)
        if "FROM edges" in sql:
            return _FakeCursor(self._edge_rows)
        raise AssertionError(f"Unexpected query: {sql}")


class _FakeGraphDB:
    def __init__(self, node_rows, edge_rows):
        self._node_rows = node_rows
        self._edge_rows = edge_rows

    def connect(self):
        connection = _FakeConnection(self._node_rows, self._edge_rows)

        class _Context:
            def __enter__(self_inner):
                return connection

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Context()


class _FakeStorage:
    def __init__(self, node_rows, edge_rows):
        self.graphdb = _FakeGraphDB(node_rows, edge_rows)


def test_graph_pickle_round_trip(tmp_path):
    graph = nx.Graph()
    graph.add_node("A", type="person")
    graph.add_edge("A", "B", weight=2.0)

    target = tmp_path / "graph.pkl"
    save_graph_to_pickle(graph, target)
    loaded = load_graph_from_pickle(target)

    assert isinstance(loaded, nx.Graph)
    assert sorted(loaded.nodes()) == ["A", "B"]
    assert loaded["A"]["B"]["weight"] == 2.0


def test_load_or_build_graph_snapshot_rebuilds_and_saves_when_missing(tmp_path):
    storage = _FakeStorage(
        node_rows=[
            ("Node A", "person", "Alpha", "chunk-1||chunk-2", "doc-a.txt"),
            ("Node B", "organization", "Beta", "", "doc-b.txt"),
        ],
        edge_rows=[
            ("Node A", "Node B", 1.5, "works with", "partnership", "chunk-1", "doc-a.txt"),
        ],
    )

    target = tmp_path / "kg_retrieval.pkl"
    graph = load_or_build_graph_snapshot(storage, snapshot_path=target)

    assert target.exists()
    assert sorted(graph.nodes()) == ["Node A", "Node B"]
    assert graph.nodes["Node A"]["chunk_uuids"] == ["chunk-1", "chunk-2"]
    assert graph["Node A"]["Node B"]["keywords"] == "partnership"


def test_load_or_build_graph_snapshot_prefers_existing_pickle(tmp_path):
    graph = nx.Graph()
    graph.add_node("Saved Node", type="event")
    target = tmp_path / "kg_retrieval.pkl"
    save_graph_to_pickle(graph, target)

    class _BrokenStorage:
        class _GraphDB:
            def connect(self):
                raise AssertionError("Storage should not be consulted when snapshot exists")

        graphdb = _GraphDB()

    loaded = load_or_build_graph_snapshot(_BrokenStorage(), snapshot_path=target)

    assert sorted(loaded.nodes()) == ["Saved Node"]

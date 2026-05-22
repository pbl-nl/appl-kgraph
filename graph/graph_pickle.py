from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx

from db_storage import Storage


def load_graph_from_pickle(path: Path) -> nx.Graph:
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.Graph):
        raise ValueError(f"Pickle at {path} does not contain a NetworkX graph.")
    return graph


def save_graph_to_pickle(graph: nx.Graph, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph, handle)
    return path


def build_graph_from_storage(
    storage: Storage,
    *,
    logger: Optional[logging.Logger] = None,
) -> nx.Graph:
    graph = nx.Graph()

    with storage.graphdb.connect() as con:
        node_rows = con.execute(
            "SELECT name, type, description, source_id, filepath FROM nodes;"
        ).fetchall()
        edge_rows = con.execute(
            "SELECT source_name, target_name, weight, description, keywords, "
            "source_id, filepath FROM edges;"
        ).fetchall()

    for name, type_, description, source_id, filepath in node_rows:
        node_id = (name or "").strip()
        if not node_id:
            continue

        chunk_uuids = []
        if source_id:
            chunk_uuids = [value.strip() for value in source_id.split("||") if value.strip()]

        graph.add_node(
            node_id,
            type=(type_ or "unknown").strip() or "unknown",
            description=(description or "").strip(),
            source_id=(source_id or "").strip(),
            filepath=(filepath or "").strip(),
            chunk_uuids=chunk_uuids,
        )

    for source, target, weight, description, keywords, source_id, filepath in edge_rows:
        src_id = (source or "").strip()
        tgt_id = (target or "").strip()
        if not src_id or not tgt_id:
            continue
        if src_id not in graph or tgt_id not in graph:
            if logger is not None:
                logger.debug("Skipping edge with missing endpoints: %s -> %s", src_id, tgt_id)
            continue

        chunk_uuids = []
        if source_id:
            chunk_uuids = [value.strip() for value in source_id.split("||") if value.strip()]

        graph.add_edge(
            src_id,
            tgt_id,
            weight=float(weight) if weight is not None else 1.0,
            description=(description or "").strip(),
            keywords=(keywords or "").strip(),
            source_id=(source_id or "").strip(),
            filepath=(filepath or "").strip(),
            chunk_uuids=chunk_uuids,
        )

    if logger is not None:
        logger.debug(
            "Loaded graph snapshot with %d nodes and %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
    return graph


def load_or_build_graph_snapshot(
    storage: Storage,
    *,
    snapshot_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> nx.Graph:
    if snapshot_path is not None and snapshot_path.exists():
        try:
            graph = load_graph_from_pickle(snapshot_path)
            if logger is not None:
                logger.debug("Loaded graph snapshot from pickle: %s", snapshot_path)
            return graph
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Failed to load graph snapshot pickle at %s; rebuilding from SQLite (%s)",
                    snapshot_path,
                    exc,
                )

    graph = build_graph_from_storage(storage, logger=logger)
    if snapshot_path is not None:
        save_graph_to_pickle(graph, snapshot_path)
        if logger is not None:
            logger.debug("Saved graph snapshot pickle to %s", snapshot_path)
    return graph

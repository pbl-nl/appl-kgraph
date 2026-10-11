"""Single-file PathRAG retriever integrated with the ingestion storage backend."""
from __future__ import annotations

import asyncio
import logging
import json
from dataclasses import dataclass, asdict, replace
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict
from pathlib import Path

import networkx as nx
import tiktoken

from settings import settings, RetrievalSettings
from db_storage import Storage
from db_storage import StoragePaths
from llm import Chat
from prompts import PROMPTS
from logging_utils import configure_file_logger
from graph_pickle import load_or_build_graph_snapshot
from project_paths import ProjectPaths
from answer_generation import AsyncTextAnswerGenerator, generate_answer_async, record_chat_audit
from query_analysis import PassThroughQueryAnalyzer, analyze_query
import retrieval as stages
from schemas import AnswerResult, retrieval_candidates_from_legacy
from context_serialization import context_from_candidates


LOGGER = logging.getLogger("PathRAG")


def _normalise_document_filter(allowed_document_names: Optional[Iterable[str]]) -> Optional[set[str]]:
    if not allowed_document_names:
        return None
    allowed = {
        str(name).strip().lower()
        for name in allowed_document_names
        if str(name).strip()
    }
    return allowed or None


def _filepath_matches_allowed(raw_filepaths: Any, allowed_documents: Optional[set[str]]) -> bool:
    if not allowed_documents:
        return True
    raw = str(raw_filepaths or "")
    if not raw:
        return False
    for token in raw.split("||"):
        token = token.strip()
        if not token:
            continue
        name = Path(token).name.strip().lower()
        if name and name in allowed_documents:
            return True
    return False


def _chunk_matches_allowed(chunk: "ChunkMatch", allowed_documents: Optional[set[str]]) -> bool:
    if not allowed_documents:
        return True
    filename = (chunk.filename or "").strip()
    document_id = (chunk.document_id or "").strip()
    candidates = {
        filename.lower(),
        Path(filename).name.lower(),
        document_id.lower(),
        Path(document_id).name.lower(),
    }
    candidates.discard("")
    return bool(candidates & allowed_documents)

# ---------------------------------------------------------------------------
# Verbosity helper
# ---------------------------------------------------------------------------

def render_full_context(result: AnswerResult) -> str:
    lines = []

    # 1) Entities
    lines.append("\n== ENTITY MATCHES ==")
    if result.context.entities:
        for i, m in enumerate(result.context.entities, 1):
            lines.append(
                f"{i}. {m.name} [{m.type or 'unknown'}] "
                f"(score={m.metadata.get('score', 0.0):.3f})\n   {m.description or '(no description)'}"
            )
    else:
        lines.append("(none)")

    # 2) Relations
    lines.append("\n== RELATION MATCHES ==")
    if result.context.relations:
        for i, r in enumerate(result.context.relations, 1):
            kw = f" | keywords: {r.keywords}" if r.keywords else ""
            lines.append(
                f"{i}. {r.source_name} ↔ {r.target_name} (score={r.metadata.get('score', 0.0):.3f})"
                f"\n   {r.description or '(no description)'}{kw}"
            )
    else:
        lines.append("(none)")

    # 3) Chunks
    lines.append("\n== CHUNK MATCHES ==")
    if result.context.chunks:
        for i, c in enumerate(result.context.chunks, 1):
            head = f"{c.filename or c.doc_id or '(unknown doc)'}"
            lines.append(
                f"{i}. {head} (score={c.metadata.get('score', 0.0):.3f})"
                f"\n   chunk_uuid={c.chunk_uuid} doc_id={c.doc_id}"
            )
            # If you want full chunk text printed, uncomment below:
            # lines.append("   --- chunk text ---")
            # lines.append(c.text)
    else:
        lines.append("(none)")

    # 4) Context windows (graph + chunks)
    lines.append("\n== CONTEXT WINDOWS ==")
    if result.context.context_windows:
        for w in result.context.context_windows:
            lines.append(f"[{w.label}] score={w.score:.3f}")
            lines.append(w.text)
            lines.append("")  # blank line
    else:
        lines.append("(none)")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Logging and token helpers
# ---------------------------------------------------------------------------

def set_logger(log_file: Path) -> None:
    configure_file_logger(
        "PathRAG",
        log_file=log_file,
        level=settings.logging.internal_log_level,
        enabled=settings.logging.internal_logging_enabled,
    )


@lru_cache(maxsize=4)
def _encoder(model: str) -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(model)


def count_tokens(text: str, model: str) -> int:
    if not text:
        return 0
    return len(_encoder(model).encode(text))


def truncate_by_tokens(text: str, max_tokens: int, model: str) -> str:
    if max_tokens <= 0:
        return ""
    encoding = _encoder(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def join_non_empty(parts: Iterable[str], delimiter: str = "\n") -> str:
    return delimiter.join(part for part in parts if part)


#----------------------------------------------------------------------------
# Contex window helpers
#----------------------------------------------------------------------------

def _find_paths_up_to_k(graph: nx.Graph, sources: List[str], *, max_depth: int) -> Dict[Tuple[str, str], List[List[str]]]:
    """Collect all simple paths up to max_depth between distinct seed pairs."""
    results: Dict[Tuple[str, str], List[List[str]]] = defaultdict(list)
    if max_depth <= 0 or len(sources) < 2:
        return results
    srcs = [s for s in sources if s in graph]
    for i, s in enumerate(srcs):
        for t in srcs[i+1:]:
            # enumerate all simple paths up to max_depth edges (max_depth nodes-1)
            try:
                # networkx returns generator; we cap by cutoff
                for p in nx.all_simple_paths(graph, source=s, target=t, cutoff=max_depth):
                    # Keep only 1..max_depth edges (i.e., len(path)-1 in [1..max_depth])
                    if 2 <= len(p) <= (max_depth + 1):
                        results[(s, t)].append(p)
            except nx.NetworkXNoPath:
                continue
    return results

def _bfs_weighted_paths(paths_by_pair: Dict[Tuple[str, str], List[List[str]]],
                        *,
                        threshold: float,
                        alpha: float) -> List[Tuple[List[str], float]]:
    """
    Replicates the weighting logic from the inspiration:
    - Build a follow_dict from enumerated paths.
    - Propagate weights with thresholding and alpha decay up to length 3.
    - Score each path by average edge weight across its edges.
    """
    combined: List[Tuple[List[str], float]] = []
    for (source, target), paths in paths_by_pair.items():
        if not paths:
            continue

        follow_dict: Dict[str, set] = {}
        for p in paths:
            for i in range(len(p) - 1):
                cur, nxt = p[i], p[i+1]
                follow_dict.setdefault(cur, set()).add(nxt)

        edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)

        # fan-out from source to depth<=3 using follow_dict (not the whole graph)
        if source not in follow_dict:
            # still score the enumerated paths with zero weights
            combined.extend((p, 0.0) for p in paths)
            continue

        # depth 1
        for n1 in follow_dict[source]:
            edge_weights[(source, n1)] += 1.0 / max(1, len(follow_dict[source]))

        # depth 2
        for n1 in follow_dict.get(source, []):
            if edge_weights[(source, n1)] > threshold:
                children = follow_dict.get(n1, [])
                for n2 in children:
                    w = edge_weights[(source, n1)] * alpha / max(1, len(children))
                    edge_weights[(n1, n2)] += w

        # depth 3
        for n1 in follow_dict.get(source, []):
            if edge_weights[(source, n1)] > threshold:
                for n2 in follow_dict.get(n1, []):
                    if edge_weights[(n1, n2)] > threshold:
                        children = follow_dict.get(n2, [])
                        for n3 in children:
                            w = edge_weights[(n1, n2)] * alpha / max(1, len(children))
                            edge_weights[(n2, n3)] += w

        # score each enumerated path by avg edge weight
        for p in paths:
            if len(p) < 2:
                combined.append((p, 0.0))
                continue
            total = 0.0
            cnt = 0
            for i in range(len(p) - 1):
                e = (p[i], p[i+1])
                total += edge_weights.get(e, 0.0)
                cnt += 1
            combined.append((p, (total / cnt) if cnt else 0.0))

    # sort descending by score
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined

def _format_path_context(graph: nx.Graph, path: List[str]) -> str:
    """
    Build a concise narrative for a 1/2/3-hop path using node/edge attrs from the in-memory graph.
    Uses node attrs: type, description; edge attrs: keywords, description.
    """
    segs: List[str] = []
    def node_desc(n: str) -> str:
        data = graph.nodes.get(n, {})
        typ = (data.get("type") or "unknown")
        desc = (data.get("description") or "").strip()
        return f"{n} ({typ})" + (f": {desc}" if desc else "")

    def edge_desc(a: str, b: str) -> str:
        data = graph.get_edge_data(a, b, default={})
        kw = (data.get("keywords") or "").strip()
        dsc = (data.get("description") or "").strip()
        parts = []
        if kw: parts.append(f"keywords={kw}")
        if dsc: parts.append(f"{dsc}")
        return " | ".join(parts)

    # Example: A --[edge info]--> B --[...]--> C ...
    segs.append(node_desc(path[0]))
    for i in range(len(path)-1):
        a, b = path[i], path[i+1]
        ed = edge_desc(a, b)
        arrow = " → "
        if ed:
            segs.append(f"{arrow}[{ed}]{arrow}{node_desc(b)}")
        else:
            segs.append(f"{arrow}{node_desc(b)}")
    return "".join(segs)

def build_path_windows(
    adapter: "StorageAdapter",
    seeds: Sequence["EntityMatch"],
    *,
    use_top_entities: int,
    max_depth: int,
    threshold: float,
    alpha: float,
    max_windows: int,
    max_tokens: int,
    model: str,
    label_prefix: str = "",
) -> List["ContextWindow"]:
    """
    PathRAG-style: enumerate simple paths up to `max_depth` between top-K seed entities,
    score via threshold+alpha propagation, and emit top windows as context.
    """
    if not seeds:
        return []

    graph = adapter.graph  # in-memory NetworkX
    # pick top-K seeds that actually exist in the graph
    seed_names = [s.name for s in seeds if s.name in graph][:use_top_entities]
    if len(seed_names) < 2:
        return []

    paths_by_pair = _find_paths_up_to_k(graph, seed_names, max_depth=max_depth)
    scored = _bfs_weighted_paths(paths_by_pair, threshold=threshold, alpha=alpha)
    if not scored:
        return []

    windows: List[ContextWindow] = []
    taken_pairs = set()  # optional: avoid spamming same pair
    for path, score in scored:
        if len(windows) >= max_windows:
            break
        pair = (path[0], path[-1])
        if pair in taken_pairs:
            continue
        text = _format_path_context(graph, path)
        text = truncate_by_tokens(text, max_tokens, model)
        label = f"{label_prefix}path::{path[0]}→{path[-1]} (len={len(path)-1})"
        windows.append(ContextWindow(label=label, text=text, score=score, source_refs=tuple(dict.fromkeys(
            source for name in path for source in graph.nodes[name].get("chunk_uuids", ())
        ))))
        taken_pairs.add(pair)
    return windows

def render_history(history: Optional[List[Tuple[str, str]]], *, max_turns: int = 4, max_chars_per_turn: int = 800) -> str:
    """
    history: list of (role, text) pairs; roles "user" or "assistant".
    Returns a compact plaintext block for the template.
    """
    if not history or max_turns <= 0:
        return ""
    # keep only the last N turns
    tail = history[-max_turns:]
    lines = []
    for role, text in tail:
        if not text:
            continue
        s = text.strip()
        if len(s) > max_chars_per_turn:
            s = s[:max_chars_per_turn] + "…"
        lines.append(f"{role}: {s}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

from schemas import EntityCandidate as EntityMatch, RelationCandidate as RelationMatch
from schemas import ChunkCandidate as ChunkMatch, ContextWindow


@dataclass(frozen=True)
class GraphSnapshot:
    """A cached view of the knowledge graph stored in SQLite."""

    graph: nx.Graph


# ---------------------------------------------------------------------------
# Storage adapter
# ---------------------------------------------------------------------------


class StorageAdapter:
    """High level helper around the ingestion ``Storage`` facade."""

    def __init__(
        self,
        paths: Optional[StoragePaths] = None,
        *,
        graph_pickle_path: Optional[Path] = None,
    ):
        self._storage = Storage(paths=paths)
        # Ensure tables exist before we start querying them.
        self._storage.init()
        self._graph_snapshot: Optional[GraphSnapshot] = None
        self._graph_pickle_path = graph_pickle_path

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    @property
    def graph(self) -> nx.Graph:
        if self._graph_snapshot is None:
            self._graph_snapshot = self._load_graph()
        return self._graph_snapshot.graph

    def refresh_graph(self) -> None:
        """Reload the graph from SQLite."""

        self._graph_snapshot = self._load_graph()

    def _load_graph(self) -> GraphSnapshot:
        graph = load_or_build_graph_snapshot(
            self._storage,
            snapshot_path=self._graph_pickle_path,
            logger=LOGGER,
        )
        return GraphSnapshot(graph=graph)

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        node_name = name.strip()
        if not node_name:
            return None
        graph = self.graph
        if node_name not in graph:
            return None
        data = dict(graph.nodes[node_name])
        data["name"] = node_name
        return data

    def get_neighbors(self, name: str) -> List[str]:
        graph = self.graph
        if name not in graph:
            return []
        return list(graph.neighbors(name))

    # ------------------------------------------------------------------
    # Vector queries
    # ------------------------------------------------------------------
    def query_entities(self, text: str, limit: int = 5) -> List[EntityMatch]:
        """Query entity vector index and return EntityMatch objects."""
        results = self._storage.search_entities(text=text, n_results=limit) or []
        matches: List[EntityMatch] = []
        if not results:
            return matches

        for metadata in results:
            if not isinstance(metadata, dict):
                continue
            matches.append(
                EntityMatch(
                    name=metadata.get("name", ""),
                    type=metadata.get("type"),
                    description=metadata.get("description", ""),
                    score=float(metadata.get("score", 0.0) or 0.0),
                    metadata=dict(metadata),
                )
            )
        return matches

    def query_relations(self, text: str, limit: int = 5) -> List[RelationMatch]:
        """Query relation vector index and return RelationMatch objects."""
        results = self._storage.search_relations(text=text, n_results=limit) or []
        matches: List[RelationMatch] = []
        if not results:
            return matches

        for metadata in results:
            if not isinstance(metadata, dict):
                continue
            matches.append(
                RelationMatch(
                    source_name=metadata.get("source_name", ""),
                    target_name=metadata.get("target_name", ""),
                    description=metadata.get("description", ""),
                    keywords=metadata.get("keywords", ""),
                    score=float(metadata.get("score", 0.0) or 0.0),
                    metadata=dict(metadata),
                )
            )
        return matches


    def query_chunks(self, text: str, limit: int = 5) -> List[ChunkMatch]:
        results = self._storage.search_chunks(text=text, n_results=limit) or []
        matches: List[ChunkMatch] = []
        for metadata in results:
            if not isinstance(metadata, dict):
                continue
            matches.append(
                ChunkMatch(
                    chunk_uuid=str(metadata.get("chunk_uuid", "")),
                    document_id=str(metadata.get("doc_id", "")),
                    filename=str(metadata.get("filename", "")),
                    text=str(metadata.get("text", "")),
                    score=float(metadata.get("score", 0.0) or 0.0),
                    metadata=dict(metadata),
                )
            )
        return matches

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def describe_subgraph(self, seeds: Sequence[str], depth: int = 2) -> List[Tuple[str, str]]:
        graph = self.graph
        descriptions: List[Tuple[str, str]] = []
        for seed in seeds:
            if seed not in graph:
                continue
            visited = {seed}
            frontier = [(seed, 0)]
            nodes_in_scope = {seed}
            while frontier:
                current, dist = frontier.pop(0)
                if dist >= depth:
                    continue
                for neighbor in graph.neighbors(current):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    nodes_in_scope.add(neighbor)
                    frontier.append((neighbor, dist + 1))

            subgraph = graph.subgraph(nodes_in_scope)
            descriptions.append((seed, self._format_subgraph(seed, subgraph)))
        return descriptions

    @staticmethod
    def _format_subgraph(seed: str, subgraph: nx.Graph) -> str:
        node_lines = []
        for node, data in subgraph.nodes(data=True):
            prefix = "Seed" if node == seed else "Node"
            node_lines.append(
                f"{prefix} {node} ({data.get('type', 'unknown')}): {data.get('description', '')}"
            )

        edge_lines = []
        for src, tgt, data in subgraph.edges(data=True):
            edge_lines.append(
                f"Relation {src} ↔ {tgt}: {data.get('description', '')} | Keywords: {data.get('keywords', '')}"
            )
        return "\n".join(node_lines + edge_lines)

    def get_documents(self, doc_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        documents: Dict[str, Dict[str, Any]] = {}
        for doc_id in doc_ids:
            if not doc_id or doc_id in documents:
                continue
            record = self._storage.get_document(doc_id)
            if record:
                documents[doc_id] = record
        return documents

    def close(self) -> None:
        close_method = getattr(self._storage, "close", None)
        if callable(close_method):
            close_method()
        self._graph_snapshot = None


def _distance_to_similarity(value: Any) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            try:
                return max(0.0, 1.0 - float(item))
            except (TypeError, ValueError):
                continue
        return 0.0
    try:
        return max(0.0, 1.0 - float(value))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


# ---------------------------------------------------------------------------
# LLM bridge
# ---------------------------------------------------------------------------


@dataclass
class RetrieveChat:
    """Lightweight async wrapper around :class:`llm.Chat`."""

    system_prompt: Optional[str] = None

    async def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Run :meth:`llm.Chat.generate` in a worker thread."""

        chat = Chat.singleton()
        return await asyncio.to_thread(
            chat.generate,
            prompt,
            system=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ---------------------------------------------------------------------------
# Context window helpers
# ---------------------------------------------------------------------------


def build_graph_windows(
    adapter: StorageAdapter,
    seeds: Sequence[EntityMatch],
    *,
    depth: int,
    max_windows: int,
    max_tokens: int,
    model: str,
    label_prefix: str = "",
) -> List[ContextWindow]:
    descriptions = adapter.describe_subgraph([seed.name for seed in seeds], depth=depth)
    windows: List[ContextWindow] = []
    score_by_seed = {seed.name: seed.score for seed in seeds}
    for seed, text in descriptions:
        trimmed = truncate_by_tokens(text, max_tokens, model)
        windows.append(
            ContextWindow(label=f"{label_prefix}graph::{seed}", text=trimmed, score=score_by_seed.get(seed, 1.0),
                          source_refs=tuple(dict.fromkeys(source for _, row in nx.ego_graph(adapter.graph, seed, radius=depth).nodes(data=True)
                                                         for source in row.get("chunk_uuids", ()))))
        )
        if len(windows) >= max_windows:
            break
    return windows


def build_chunk_windows(
    chunks: Sequence[ChunkMatch],
    *,
    max_windows: int,
    max_tokens: int,
    model: str,
    label_prefix: str = "",
) -> List[ContextWindow]:
    windows: List[ContextWindow] = []
    for chunk in chunks[:max_windows]:
        snippet = truncate_by_tokens(chunk.text, max_tokens, model)
        label = f"{label_prefix}chunk::{chunk.filename or chunk.document_id}"
        windows.append(ContextWindow(label=label, text=snippet, score=chunk.score, source_refs=(chunk.chunk_uuid,)))
    return windows


def build_context_for_prompt(context_windows, entity_matches, relation_matches, chunk_matches) -> str:
    payload = {
        "context_windows": [{"label": item.label, "text": item.text, "score": item.score} for item in context_windows],
        "entity_matches": [{"name": item.name, "type": item.type, "description": item.description,
                            "score": item.metadata.get("score", 0.0)} for item in entity_matches],
        "relation_matches": [{"source_name": item.source_name, "target_name": item.target_name,
                              "description": item.description, "keywords": item.keywords,
                              "score": item.metadata.get("score", 0.0)} for item in relation_matches],
        "chunk_matches": [{"chunk_uuid": item.chunk_uuid, "document_id": item.doc_id,
                           "filename": item.filename, "text": item.text,
                           "score": item.metadata.get("score", 0.0)} for item in chunk_matches],
    }
    return json.dumps(payload, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def _retrieval_model_metadata() -> Dict[str, Any]:
    provider = settings.provider.provider
    if provider == "azure":
        return {
            "provider": provider,
            "model_name": settings.provider.azure_llm_deployment,
            "model_version": settings.provider.azure_api_version,
            "api_version": settings.provider.azure_api_version,
            "endpoint": settings.provider.azure_endpoint,
        }
    return {
        "provider": provider,
        "model_name": settings.provider.openai_llm_model,
        "model_version": settings.provider.openai_llm_model,
        "api_version": "",
        "endpoint": settings.provider.openai_base_url or "https://api.openai.com/v1",
    }


# ---------------------------------------------------------------------------
# PathRAG entry point
# ---------------------------------------------------------------------------


class PathCandidateRetriever:
    def __init__(self, storage):
        self.storage = storage

    def retrieve(self, plan):
        config = RetrievalSettings(**plan.settings)
        entity_matches = self.storage.query_entities(
            plan.query,
            limit=config.entity_top_k,
        )
        relation_matches = self.storage.query_relations(
            plan.query,
            limit=config.relation_top_k,
        )
        chunk_matches = self.storage.query_chunks(
            plan.query,
            limit=config.chunk_top_k,
        )

        allowed_documents = _normalise_document_filter(plan.metadata.get("allowed_document_names"))
        if allowed_documents:
            entity_matches = [
                match
                for match in entity_matches
                if _filepath_matches_allowed(
                    (self.storage.get_node(match.name) or {}).get("filepath", ""),
                    allowed_documents,
                )
            ]
            relation_matches = [
                match
                for match in relation_matches
                if _filepath_matches_allowed(
                    (self.storage.graph.get_edge_data(match.source_name, match.target_name) or {}).get("filepath", ""),
                    allowed_documents,
                )
            ]
            chunk_matches = [
                match
                for match in chunk_matches
                if _chunk_matches_allowed(match, allowed_documents)
            ]

        candidates = retrieval_candidates_from_legacy(plan, {
            "entity_matches": entity_matches, "relation_matches": relation_matches, "chunk_matches": chunk_matches})
        graph = self.storage.graph.copy()
        if allowed_documents:
            graph.remove_edges_from([(source, target) for source, target, row in graph.edges(data=True)
                if not _filepath_matches_allowed(row.get("filepath"), allowed_documents)])
            graph.remove_nodes_from([name for name, row in graph.nodes(data=True)
                if not _filepath_matches_allowed(row.get("filepath"), allowed_documents)])
        for entity in candidates.entities:
            row = graph.nodes.get(entity.name, {})
            entity.metadata.update(filepath=row.get("filepath"), source_ids=tuple(row.get("chunk_uuids", ())))
        for relation in candidates.relations:
            row = graph.get_edge_data(relation.source_name, relation.target_name) or {}
            relation.metadata.update(filepath=row.get("filepath"), source_ids=tuple(row.get("chunk_uuids", ())))
        return replace(candidates, metadata={"graph_data": nx.node_link_data(graph, edges="edges")})


class PathGraphView:
    """In-memory graph adapter; context construction cannot query storage."""
    def __init__(self, graph):
        self.graph = graph

    describe_subgraph = StorageAdapter.describe_subgraph
    _format_subgraph = staticmethod(StorageAdapter._format_subgraph)


class PathContextBuilder:
    def build(self, plan, candidates):
        config = RetrievalSettings(**plan.settings)
        adapter = PathGraphView(nx.node_link_graph(candidates.metadata["graph_data"], edges="edges"))
        entity_matches = candidates.entities
        chunk_matches = candidates.chunks
        def build_global_windows(
            adapter: "StorageAdapter",
            seeds: Sequence["EntityMatch"],
        ) -> List["ContextWindow"]:
            """
            Build high-level (global) context windows using path-finding between entities.
            - Uses top-K entity seeds.
            - Enumerates paths up to depth (hops).
            - Weights and ranks paths (PathRAG-style).
            - Returns ContextWindows labeled 'global::path::<src>→<tgt>'.
            """
            if not config.hybrid_use_paths_for_global or not seeds:
                return []

            # weighted paths between top entities
            path_windows = build_path_windows(
                adapter,
                seeds,
                use_top_entities=config.path_use_top_entities,
                max_depth=config.path_max_depth,
                threshold=config.path_threshold,
                alpha=config.path_alpha,
                max_windows=config.path_max_windows,
                max_tokens=config.path_window_tokens,
                model=config.tiktoken_model,
                label_prefix="global::",
            )
            return path_windows

        def build_local_windows(
            adapter: "StorageAdapter",
            seeds: Sequence["EntityMatch"],
            chunks: Sequence["ChunkMatch"],
        ) -> List["ContextWindow"]:
            """
            Build low-level (local) context windows.
            - Includes raw chunk windows (from vector search).
            - Optionally includes graph neighborhoods (subgraph).
            - Returns ContextWindows labeled 'local::chunk' or 'local::graph'.
            """
            # initialise empty list
            local_windows: List[ContextWindow] = []

            if config.use_local_chunks and chunks:
                local_windows += build_chunk_windows(
                    chunks,
                    max_windows=config.chunk_windows,
                    max_tokens=config.chunk_window_tokens,
                    model=config.tiktoken_model,
                    label_prefix="local::",
                )

            if config.use_local_graph and seeds:
                local_windows += build_graph_windows(
                    adapter,
                    seeds,
                    depth=config.graph_depth,
                    max_windows=config.graph_windows,
                    max_tokens=config.graph_window_tokens,
                    model=config.tiktoken_model,
                    label_prefix="local::",
                )

            return local_windows[:config.local_max_windows]

        # Assemble global context windows
        global_windows = build_global_windows(adapter, entity_matches)

        # Assemble local context windows
        local_windows = build_local_windows(
            adapter,
            entity_matches,
            chunk_matches,
        )

        # Merge windows
        context_windows = global_windows + local_windows

        windows = context_windows
        return context_from_candidates(candidates, windows, metadata={
            "global_window_count": len(global_windows), "local_window_count": len(local_windows)})


async def generate_path_answer(query, context, history, *, chat):
    config = RetrievalSettings(**context.query_plan.settings)
    prompt = PROMPTS["pathrag_response"].format(
        context_data=build_context_for_prompt(context.context_windows, context.entities, context.relations, context.chunks),
        response_type=config.response_type,
        history=render_history(history, max_turns=config.history_turns), user_prompt=query)
    return await chat.generate(prompt, max_tokens=config.llm_max_tokens,
                                     temperature=config.llm_temperature)


def make_answer_generator(chat=None):
    """Create an answer adapter that needs supplied context and a model, without storage."""
    chat = chat if chat is not None else RetrieveChat()
    return AsyncTextAnswerGenerator(
        lambda query, context, history: generate_path_answer(query, context, history, chat=chat),
        model=_retrieval_model_metadata(),
    )


class PathRAG:
    """ PathRAG retriever that integrates storage and LLM to answer questions.
    Example usage:
        rag = PathRAG(storage_paths=StoragePaths(...), system_prompt="You are a helpful assistant.")
        result = rag.retrieve("What is the capital of France?")
    """

    def __init__(
        self,
        *,
        storage_paths: Optional[StoragePaths] = None,
        project_paths: Optional[ProjectPaths] = None,
        system_prompt: Optional[str] = None,
        log_file: Optional[str] = None,
        query_analyzer=None, candidate_retriever=None, context_builder=None,
        answer_generator=None, reranker=None, audit_recorder=None,
    ) -> None:
        self._query_analyzer = query_analyzer
        self._candidate_retriever = candidate_retriever
        self._context_builder = context_builder
        self._answer_generator = answer_generator
        self._reranker = reranker
        self._audit_recorder = audit_recorder
        self._project_paths = project_paths
        effective_storage_paths = (
            storage_paths
            if storage_paths is not None
            else (project_paths.storage if project_paths is not None else None)
        )
        effective_log_file = (
            Path(log_file)
            if log_file is not None
            else (
                project_paths.pathrag_log_file
                if project_paths is not None
                else Path("PathRAG.log")
            )
        )
        set_logger(effective_log_file)
        LOGGER.info("Initialising PathRAG retriever")
        self._storage = StorageAdapter(
            paths=effective_storage_paths,
            graph_pickle_path=(
                project_paths.retrieval_graph_pickle_file
                if project_paths is not None
                else None
            ),
        )
        self._chat = RetrieveChat(system_prompt=system_prompt)

    def close(self) -> None:
        close_method = getattr(self._storage, "close", None)
        if callable(close_method):
            close_method()

    # ------------------------------------------------------------------
    # Retrieval entry points
    # ------------------------------------------------------------------
    async def aretrieve(self, question, conversation_history=None, allowed_document_names=None) -> AnswerResult:
        plan = analyze_query(question, conversation_history or [], self._query_analyzer or PassThroughQueryAnalyzer(settings=asdict(settings.retrieval)))
        plan = replace(plan, metadata={**plan.metadata, "allowed_document_names": list(allowed_document_names) if allowed_document_names is not None else None})
        candidates = stages.retrieve_candidates(plan, self._candidate_retriever or PathCandidateRetriever(self._storage))
        context = stages.build_context(plan, candidates, self._context_builder or PathContextBuilder())
        context = stages.rerank_context(context, self._reranker)
        result = await generate_answer_async(question, context, plan.history,
            self._answer_generator or make_answer_generator(self._chat))
        record_chat_audit(result, self._project_paths, self._audit_recorder)
        return result

    def retrieve(
        self,
        question: str,
        conversation_history: Optional[List[Tuple[str, str]]] = None,
        allowed_document_names: Optional[Iterable[str]] = None,
    ) -> AnswerResult:
        """Synchronous helper that creates an event loop if needed."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aretrieve(question, conversation_history, allowed_document_names))
        else:  # pragma: no cover - usage depends on embedding application
            raise RuntimeError(
                "retrieve() cannot be used when an event loop is already running. "
                "Use 'await aretrieve(...)' instead."
            )

__all__ = [
    "PathRAG",
    "StorageAdapter",
    "StoragePaths",
    "EntityMatch",
    "RelationMatch",
    "ChunkMatch",
    "ContextWindow",
    "AnswerResult",
]

"""Single-file PathRAG retriever integrated with the ingestion storage backend."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict

import networkx as nx
import tiktoken

from settings import settings
from storage import Storage
from storage import StoragePaths
from llm import Chat
from prompts import PROMPTS


LOGGER = logging.getLogger("PathRAG")

# ---------------------------------------------------------------------------
# Verbosity helper
# ---------------------------------------------------------------------------

def render_full_context(result: RetrievalResult) -> str:
    lines = []

    # 1) Entities
    lines.append("\n== ENTITY MATCHES ==")
    if result.entity_matches:
        for i, m in enumerate(result.entity_matches, 1):
            lines.append(
                f"{i}. {m.name} [{m.type or 'unknown'}] "
                f"(score={m.score:.3f})\n   {m.description or '(no description)'}"
            )
    else:
        lines.append("(none)")

    # 2) Relations
    lines.append("\n== RELATION MATCHES ==")
    if result.relation_matches:
        for i, r in enumerate(result.relation_matches, 1):
            kw = f" | keywords: {r.keywords}" if r.keywords else ""
            lines.append(
                f"{i}. {r.source_name} ↔ {r.target_name} (score={r.score:.3f})"
                f"\n   {r.description or '(no description)'}{kw}"
            )
    else:
        lines.append("(none)")

    # 3) Chunks
    lines.append("\n== CHUNK MATCHES ==")
    if result.chunk_matches:
        for i, c in enumerate(result.chunk_matches, 1):
            head = f"{c.filename or c.document_id or '(unknown doc)'}"
            lines.append(
                f"{i}. {head} (score={c.score:.3f})"
                f"\n   chunk_uuid={c.chunk_uuid} doc_id={c.document_id}"
            )
            # If you want full chunk text printed, uncomment below:
            # lines.append("   --- chunk text ---")
            # lines.append(c.text)
    else:
        lines.append("(none)")

    # 4) Context windows (graph + chunks)
    lines.append("\n== CONTEXT WINDOWS ==")
    if result.context_windows:
        for w in result.context_windows:
            lines.append(f"[{w.label}] score={w.score:.3f}")
            lines.append(w.text)
            lines.append("")  # blank line
    else:
        lines.append("(none)")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Logging and token helpers
# ---------------------------------------------------------------------------

def set_logger(log_file: str) -> None:
    """Configure the package wide logger."""

    LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if not LOGGER.handlers:
        LOGGER.addHandler(handler)


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
        windows.append(ContextWindow(label=label, text=text, score=score))
        taken_pairs.add(pair)
    return windows

def render_history(history: Optional[List[Tuple[str, str]]], *, max_turns: int = 4, max_chars_per_turn: int = 800) -> str:
    """
    history: list of (role, text) pairs; roles "user" or "assistant".
    Returns a compact plaintext block for the template.
    """
    if not history:
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

@dataclass(frozen=True)
class EntityMatch:
    """Represents a node match returned from the entity vector index."""

    name: str
    type: Optional[str]
    description: str
    score: float


@dataclass(frozen=True)
class RelationMatch:
    """Represents an edge match returned from the relation vector index."""

    source_name: str
    target_name: str
    description: str
    keywords: str
    score: float


@dataclass(frozen=True)
class ChunkMatch:
    """Represents a chunk retrieved from the chunk vector index."""

    chunk_uuid: str
    document_id: str
    filename: str
    text: str
    score: float


@dataclass(frozen=True)
class ContextWindow:
    """A context window summarising graph and chunk evidence."""

    label: str
    text: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Structured output returned by :class:`PathRAG`."""

    answer: str
    context_windows: List[ContextWindow]
    entity_matches: List[EntityMatch]
    relation_matches: List[RelationMatch]
    chunk_matches: List[ChunkMatch]

@dataclass(frozen=True)
class GraphSnapshot:
    """A cached view of the knowledge graph stored in SQLite."""

    graph: nx.Graph


# ---------------------------------------------------------------------------
# Storage adapter
# ---------------------------------------------------------------------------


class StorageAdapter:
    """High level helper around the ingestion ``Storage`` facade."""

    def __init__(self, paths: Optional[StoragePaths] = None):
        self._storage = Storage(paths=paths)
        # Ensure tables exist before we start querying them.
        self._storage.init()
        self._graph_snapshot: Optional[GraphSnapshot] = None

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
        graph = nx.Graph()

        with self._storage.graph.connect() as con:
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
            graph.add_node(
                node_id,
                type=(type_ or "unknown").strip() or "unknown",
                description=(description or "").strip(),
                source_id=(source_id or "").strip(),
                filepath=(filepath or "").strip(),
            )

        for row in edge_rows:
            source, target, weight, description, keywords, source_id, filepath = row
            src_id = (source or "").strip()
            tgt_id = (target or "").strip()
            if not src_id or not tgt_id:
                continue
            if src_id not in graph or tgt_id not in graph:
                LOGGER.debug("Skipping edge with missing endpoints: %s -> %s", src_id, tgt_id)
                continue
            graph.add_edge(
                src_id,
                tgt_id,
                weight=float(weight) if weight is not None else 1.0,
                description=(description or "").strip(),
                keywords=(keywords or "").strip(),
                source_id=(source_id or "").strip(),
                filepath=(filepath or "").strip(),
            )

        LOGGER.debug(
            "Loaded graph snapshot with %d nodes and %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
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
    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        return [value]

    def query_entities(self, text: str, limit: int = 5) -> List[EntityMatch]:
        """Query entity vector index and return EntityMatch objects."""
        results = self._storage.entity_vectors.query(text=text, n_results=limit) or []
        matches: List[EntityMatch] = []
        if not results:
            return matches

        # Chroma returns a single dict with lists
        ids = results[0].get("ids", [])
        metadatas = results[0].get("metadatas", [])
        distances = results[0].get("distances", [])

        for i, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            entity_id = ids[i] if i < len(ids) else ""
            distance = distances[i] if i < len(distances) else None
            matches.append(
                EntityMatch(
                    name=metadata.get("name", entity_id),
                    type=metadata.get("type"),
                    description=metadata.get("description", ""),
                    score=_distance_to_similarity(distance),
                )
            )
        return matches

    def query_relations(self, text: str, limit: int = 5) -> List[RelationMatch]:
        """Query relation vector index and return RelationMatch objects."""
        results = self._storage.relation_vectors.query(text=text, n_results=limit) or []
        matches: List[RelationMatch] = []
        if not results:
            return matches

        ids = results[0].get("ids", [])
        metadatas = results[0].get("metadatas", [])
        distances = results[0].get("distances", [])

        for i, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            distance = distances[i] if i < len(distances) else None
            matches.append(
                RelationMatch(
                    source_name=metadata.get("source_name", ""),
                    target_name=metadata.get("target_name", ""),
                    description=metadata.get("description", ""),
                    keywords=metadata.get("keywords", ""),
                    score=_distance_to_similarity(distance),
                )
            )
        return matches


    def query_chunks(self, text: str, limit: int = 5) -> List[ChunkMatch]:
        results = self._storage.chunk_vectors.query(text=text, n_results=limit) or []
        matches: List[ChunkMatch] = []
        for result in results:
            metadatas = self._as_list(result.get("metadatas"))
            ids = self._as_list(result.get("ids"))
            distances = self._as_list(result.get("distances"))
            documents = self._as_list(result.get("documents"))
            max_len = max(
                (
                    len(seq)
                    for seq in (metadatas, ids, distances, documents)
                    if seq
                ),
                default=0,
            )
            for index in range(max_len):
                metadata = metadatas[index] if index < len(metadatas) else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                chunk_id = ids[index] if index < len(ids) else ""
                distance = distances[index] if index < len(distances) else None
                document = documents[index] if index < len(documents) else ""
                if not isinstance(document, str):
                    document = str(document or "")
                matches.append(
                    ChunkMatch(
                        chunk_uuid=str(chunk_id),
                        document_id=str(metadata.get("doc_id", "")),
                        filename=str(metadata.get("filename", "")),
                        text=document,
                        score=_distance_to_similarity(distance),
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
            ContextWindow(label=f"{label_prefix}graph::{seed}", text=trimmed, score=score_by_seed.get(seed, 1.0))
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
        windows.append(ContextWindow(label=label, text=snippet, score=chunk.score))
    return windows


def merge_windows(graph_windows: Sequence[ContextWindow], chunk_windows: Sequence[ContextWindow]) -> List[ContextWindow]:
    return list(graph_windows) + list(chunk_windows)


def format_windows_for_prompt(windows: Iterable[ContextWindow]) -> str:
    blocks = [f"[{window.label}]\n{window.text}" for window in windows]
    return join_non_empty(blocks, delimiter="\n\n")


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# !! lightweight placeholder for testing, replace with imported PROMPT when finished.
RAG_PROMPT = (
    "You are a helpful assistant. Use the supplied context to answer the question."
    "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer in Markdown."
)


# ---------------------------------------------------------------------------
# PathRAG entry point
# ---------------------------------------------------------------------------


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
        system_prompt: Optional[str] = None,
        log_file: str = "PathRAG.log",
    ) -> None:
        set_logger(log_file)
        LOGGER.info("Initialising PathRAG retriever")
        self._storage = StorageAdapter(paths=storage_paths)
        self._chat = RetrieveChat(system_prompt=system_prompt)

    # ------------------------------------------------------------------
    # Retrieval entry points
    # ------------------------------------------------------------------
    async def aretrieve(self, question: str) -> RetrievalResult:
        LOGGER.info("Running retrieval for query: %s", question)

        entity_matches = self._storage.query_entities(question, limit=settings.retrieval.entity_top_k)
        relation_matches = self._storage.query_relations(question, limit=settings.retrieval.relation_top_k)
        chunk_matches = self._storage.query_chunks(question, limit=settings.retrieval.chunk_top_k)

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
            if not settings.retrieval.hybrid_use_paths_for_global or not seeds:
                return []

            # weighted paths between top entities
            path_windows = build_path_windows(
                adapter,
                seeds,
                use_top_entities=settings.retrieval.path_use_top_entities,
                max_depth=settings.retrieval.path_max_depth,
                threshold=settings.retrieval.path_threshold,
                alpha=settings.retrieval.path_alpha,
                max_windows=settings.retrieval.path_max_windows,
                max_tokens=settings.retrieval.path_window_tokens,
                model=settings.retrieval.tiktoken_model,
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

            if settings.retrieval.use_local_chunks and chunks:
                local_windows += build_chunk_windows(
                    chunks,
                    max_windows=settings.retrieval.chunk_windows,          # FIXED
                    max_tokens=settings.retrieval.chunk_window_tokens,     # FIXED
                    model=settings.retrieval.tiktoken_model,
                    label_prefix="local::",
                )
                
            if settings.retrieval.use_local_graph and seeds:
                local_windows += build_graph_windows(
                    adapter,
                    seeds,
                    depth=settings.retrieval.graph_depth,
                    max_windows=settings.retrieval.graph_windows,
                    max_tokens=settings.retrieval.graph_window_tokens,
                    model=settings.retrieval.tiktoken_model,
                    label_prefix="local::",
                )

            return local_windows[:settings.retrieval.local_max_windows]

        # Assemble global context windows
        global_windows = build_global_windows(self._storage, entity_matches)

        # Assemble local context windows
        local_windows = build_local_windows(self._storage, entity_matches, chunk_matches)

        # Merge and format for prompt
        context_windows = global_windows + local_windows
        context_block = format_windows_for_prompt(context_windows)

        prompt = RAG_PROMPT.format(context=context_block, question=question)
        answer = await self._chat.generate(
            prompt,
            max_tokens=settings.retrieval.llm_max_tokens,
            temperature=settings.retrieval.llm_temperature,
        )
        return RetrievalResult(
            answer=answer,
            context_windows=context_windows,
            entity_matches=entity_matches,
            relation_matches=relation_matches,
            chunk_matches=chunk_matches,
        )

    def retrieve(self, question: str) -> RetrievalResult:
        """Synchronous helper that creates an event loop if needed."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aretrieve(question))
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
    "RetrievalResult",
]


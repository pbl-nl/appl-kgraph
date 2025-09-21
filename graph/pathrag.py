"""Single-file PathRAG retriever integrated with the ingestion storage backend."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import tiktoken

try:  # pragma: no cover - the ingestion package provides these modules
    from storage import Storage as IngestionStorage
    from storage import StoragePaths
except ImportError as exc:  # pragma: no cover - surface a helpful error message
    raise ImportError(
        "The ingestion storage module could not be imported. "
        "Ensure that the PathRAG ingestion package is installed and available "
        "on PYTHONPATH before initialising the retriever."
    ) from exc

try:  # pragma: no cover - consumers are expected to provide this module
    from llm import Chat as IngestionChat
except ImportError as exc:  # pragma: no cover - surface a useful error message
    raise ImportError(
        "The ingestion 'llm' module could not be imported. Ensure the PathRAG "
        "ingestion package (or equivalent) is available on PYTHONPATH before "
        "initialising the retriever."
    ) from exc


LOGGER = logging.getLogger("PathRAG")


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


@dataclass
class RetrieverConfig:
    """Tunable knobs that control how evidence is assembled."""

    entity_top_k: int = 5
    relation_top_k: int = 5
    chunk_top_k: int = 6
    graph_depth: int = 2
    graph_windows: int = 3
    chunk_windows: int = 3
    graph_window_tokens: int = 512
    chunk_window_tokens: int = 512
    tiktoken_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.2


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
        self._storage = IngestionStorage(paths=paths)
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
    def query_entities(self, text: str, limit: int = 5) -> List[EntityMatch]:
        results = self._storage.entity_vectors.query(text=text, n_results=limit) or []
        matches: List[EntityMatch] = []
        for result in results:
            metadata = result.get("metadatas") or {}
            matches.append(
                EntityMatch(
                    name=metadata.get("name", result.get("ids", "")),
                    type=metadata.get("type"),
                    description=metadata.get("description", ""),
                    score=_distance_to_similarity(result.get("distances")),
                )
            )
        return matches

    def query_relations(self, text: str, limit: int = 5) -> List[RelationMatch]:
        results = self._storage.relation_vectors.query(text=text, n_results=limit) or []
        matches: List[RelationMatch] = []
        for result in results:
            metadata = result.get("metadatas") or {}
            matches.append(
                RelationMatch(
                    source_name=metadata.get("source_name", ""),
                    target_name=metadata.get("target_name", ""),
                    description=metadata.get("description", ""),
                    keywords=metadata.get("keywords", ""),
                    score=_distance_to_similarity(result.get("distances")),
                )
            )
        return matches

    def query_chunks(self, text: str, limit: int = 5) -> List[ChunkMatch]:
        results = self._storage.chunk_vectors.query(text=text, n_results=limit) or []
        matches: List[ChunkMatch] = []
        for result in results:
            metadata = result.get("metadatas") or {}
            matches.append(
                ChunkMatch(
                    chunk_uuid=result.get("ids", ""),
                    document_id=str(metadata.get("doc_id", "")),
                    filename=str(metadata.get("filename", "")),
                    text=result.get("documents", ""),
                    score=_distance_to_similarity(result.get("distances")),
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
    try:
        return max(0.0, 1.0 - float(value))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


# ---------------------------------------------------------------------------
# LLM bridge
# ---------------------------------------------------------------------------


@dataclass
class AsyncChat:
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

        chat = IngestionChat.singleton()
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
) -> List[ContextWindow]:
    descriptions = adapter.describe_subgraph([seed.name for seed in seeds], depth=depth)
    windows: List[ContextWindow] = []
    score_by_seed = {seed.name: seed.score for seed in seeds}
    for seed, text in descriptions:
        trimmed = truncate_by_tokens(text, max_tokens, model)
        windows.append(
            ContextWindow(label=f"graph::{seed}", text=trimmed, score=score_by_seed.get(seed, 1.0))
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
) -> List[ContextWindow]:
    windows: List[ContextWindow] = []
    for chunk in chunks[:max_windows]:
        snippet = truncate_by_tokens(chunk.text, max_tokens, model)
        label = f"chunk::{chunk.filename or chunk.document_id}" if chunk.filename else "chunk"
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


RAG_PROMPT = (
    "You are a helpful assistant. Use the supplied context to answer the question."
    "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer in Markdown."
)


# ---------------------------------------------------------------------------
# PathRAG entry point
# ---------------------------------------------------------------------------


class PathRAG:
    """Minimal retriever that relies on the ingestion storage backend."""

    def __init__(
        self,
        *,
        storage_paths: Optional[StoragePaths] = None,
        system_prompt: Optional[str] = None,
        log_file: str = "PathRAG.log",
        config: Optional[RetrieverConfig] = None,
    ) -> None:
        set_logger(log_file)
        LOGGER.info("Initialising PathRAG retriever")
        self._config = config or RetrieverConfig()
        self._storage = StorageAdapter(paths=storage_paths)
        self._chat = AsyncChat(system_prompt=system_prompt)

    # ------------------------------------------------------------------
    # Retrieval entry points
    # ------------------------------------------------------------------
    async def aretrieve(self, question: str) -> RetrievalResult:
        LOGGER.info("Running retrieval for query: %s", question)
        cfg = self._config

        entity_matches = self._storage.query_entities(question, limit=cfg.entity_top_k)
        relation_matches = self._storage.query_relations(question, limit=cfg.relation_top_k)
        chunk_matches = self._storage.query_chunks(question, limit=cfg.chunk_top_k)

        graph_windows = build_graph_windows(
            self._storage,
            entity_matches,
            depth=cfg.graph_depth,
            max_windows=cfg.graph_windows,
            max_tokens=cfg.graph_window_tokens,
            model=cfg.tiktoken_model,
        )
        chunk_windows = build_chunk_windows(
            chunk_matches,
            max_windows=cfg.chunk_windows,
            max_tokens=cfg.chunk_window_tokens,
            model=cfg.tiktoken_model,
        )
        context_windows = merge_windows(graph_windows, chunk_windows)
        context_block = format_windows_for_prompt(context_windows)

        prompt = RAG_PROMPT.format(context=context_block, question=question)
        answer = await self._chat.generate(
            prompt,
            max_tokens=cfg.llm_max_tokens,
            temperature=cfg.llm_temperature,
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
    "RetrieverConfig",
    "StorageAdapter",
    "StoragePaths",
    "EntityMatch",
    "RelationMatch",
    "ChunkMatch",
    "ContextWindow",
    "RetrievalResult",
]
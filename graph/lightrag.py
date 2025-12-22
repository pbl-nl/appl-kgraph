"""LightRAG retriever integrated with the ingestion storage backend."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import tiktoken
from flashrank import Ranker, RerankRequest

from settings import settings
from db_storage import Storage
from db_storage import StoragePaths
from llm import Chat
from prompts import PROMPTS


LOGGER = logging.getLogger("LightRAG")

# ---------------------------------------------------------------------------
# Verbosity helper
# ---------------------------------------------------------------------------

def render_full_context(result: RetrievalResult) -> str:
    """Render full context for debugging purposes."""
    lines = []

    # 1) Entities
    lines.append("\n== ENTITY CONTEXT ==")
    if result.entities_context:
        for i, e in enumerate(result.entities_context, 1):
            lines.append(
                f"{i}. {e.get('entity', 'N/A')} [{e.get('type', 'unknown')}]\n"
                f"   {e.get('description', '(no description)')}\n"
                f"   File: {e.get('file_path', 'N/A')}"
            )
    else:
        lines.append("(none)")

    # 2) Relations
    lines.append("\n== RELATION CONTEXT ==")
    if result.relations_context:
        for i, r in enumerate(result.relations_context, 1):
            lines.append(
                f"{i}. {r.get('entity1', 'N/A')} ↔ {r.get('entity2', 'N/A')}\n"
                f"   {r.get('description', '(no description)')}\n"
                f"   File: {r.get('file_path', 'N/A')}"
            )
    else:
        lines.append("(none)")

    # 3) Chunks
    lines.append("\n== ALL CHUNKS ==")
    if result.all_chunks:
        for i, c in enumerate(result.all_chunks, 1):
            source = c.get('source_type', 'unknown')
            lines.append(
                f"{i}. [{source}] {c.get('id', 'N/A')}\n"
                f"   Order: {c.get('order', 'N/A')}, Relation: {c.get('relation', 'N/A')}\n"
                f"   Text: {c.get('text', '')[:100]}..."
            )
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


# def truncate_by_tokens(list_dict: List[Dict[str, Any]], max_tokens: int, model: str) -> List[Dict[str, Any]]:
#     if max_tokens <= 0:
#         return []
#     tokens = 0
#     for i, item in enumerate(list_dict):
#         item_tokens = count_tokens(str(item), model)
#         tokens += item_tokens
#         if tokens > max_tokens:
#             return list_dict[:i]
#     return list_dict


def join_non_empty(parts: Iterable[str], delimiter: str = "\n") -> str:
    return delimiter.join(part for part in parts if part)


def get_conversation_turns(
    history: Optional[List[Tuple[str, str]]],
    max_turns: int = 4
) -> str:
    """
    Format conversation history for the prompt.

    Args:
        history: List of (role, text) tuples
        max_turns: Maximum number of conversation turns to include

    Returns:
        Formatted conversation history string
    """
    if not history:
        return ""

    # Keep only the last N turns
    tail = history[-max_turns:]
    lines = []
    for role, text in tail:
        if not text:
            continue
        s = text.strip()
        lines.append(f"{role}: {s}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalResult:
    """Structured output returned by LightRAG."""
    answer: str
    entities_context: List[Dict[str, Any]]
    relations_context: List[Dict[str, Any]]
    all_chunks: List[Dict[str, Any]]
    hl_keywords: List[str]
    ll_keywords: List[str]


@dataclass(frozen=True)
class GraphSnapshot:
    """A cached view of the knowledge graph stored in SQLite."""
    graph: nx.Graph


# ---------------------------------------------------------------------------
# Storage adapter (reusing from pathrag.py pattern)
# ---------------------------------------------------------------------------

class StorageAdapter:
    """High level helper around the ingestion Storage facade."""

    def __init__(self, paths: Optional[StoragePaths] = None):
        self._storage = Storage(paths=paths)
        self._storage.init()
        self._graph_snapshot: Optional[GraphSnapshot] = None

    @property
    def graph(self) -> nx.Graph:
        if self._graph_snapshot is None:
            self._graph_snapshot = self._load_graph()
        return self._graph_snapshot.graph

    def refresh_graph(self) -> None:
        """Reload the graph from SQLite."""
        self._graph_snapshot = self._load_graph()

    def _load_graph(self) -> GraphSnapshot:
        """Load graph from storage into NetworkX."""
        graph = nx.Graph()

        with self._storage.graphdb.connect() as con:
            node_rows = con.execute(
                "SELECT name, type, description, source_id, filepath FROM nodes;"
            ).fetchall()
            edge_rows = con.execute(
                "SELECT source_name, target_name, weight, description, keywords, "
                "source_id, filepath FROM edges;"
            ).fetchall()

        # Add nodes with chunk_uuid list
        for name, type_, description, source_id, filepath in node_rows:
            node_id = (name or "").strip()
            if not node_id:
                continue

            # Parse source_id to get chunk_uuids
            chunk_uuids = []
            if source_id:
                chunk_uuids = [s.strip() for s in source_id.split("||") if s.strip()]

            graph.add_node(
                node_id,
                type=(type_ or "unknown").strip() or "unknown",
                description=(description or "").strip(),
                source_id=(source_id or "").strip(),
                filepath=(filepath or "").strip(),
                chunk_uuids=chunk_uuids,
            )

        # Add edges with chunk_uuid list
        for row in edge_rows:
            source, target, weight, description, keywords, source_id, filepath = row
            src_id = (source or "").strip()
            tgt_id = (target or "").strip()
            if not src_id or not tgt_id:
                continue
            if src_id not in graph or tgt_id not in graph:
                LOGGER.debug("Skipping edge with missing endpoints: %s -> %s", src_id, tgt_id)
                continue

            # Parse source_id to get chunk_uuids
            chunk_uuids = []
            if source_id:
                chunk_uuids = [s.strip() for s in source_id.split("||") if s.strip()]

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

        LOGGER.debug(
            "Loaded graph snapshot with %d nodes and %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return GraphSnapshot(graph=graph)

    def query_entities(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query entity vector index and return entity information with IDs."""
        results = self._storage.entity_vectors.query(text=text, n_results=limit) or []
        matches: List[Dict[str, Any]] = []
        if not results:
            return matches

        ids = results[0].get("ids", [])
        metadatas = results[0].get("metadatas", [])
        distances = results[0].get("distances", [])

        for i, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            entity_id = ids[i] if i < len(ids) else ""
            distance = distances[i] if i < len(distances) else None
            matches.append({
                "id": entity_id,
                "name": metadata.get("name", entity_id),
                "type": metadata.get("type"),
                "description": metadata.get("description", ""),
                "score": _distance_to_similarity(distance),
            })
        return matches

    def query_relations(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query relation vector index and return relation information with IDs."""
        results = self._storage.relation_vectors.query(text=text, n_results=limit) or []
        matches: List[Dict[str, Any]] = []
        if not results:
            return matches

        ids = results[0].get("ids", [])
        metadatas = results[0].get("metadatas", [])
        distances = results[0].get("distances", [])

        for i, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            relation_id = ids[i] if i < len(ids) else ""
            distance = distances[i] if i < len(distances) else None
            matches.append({
                "id": relation_id,
                "source_name": metadata.get("source_name", ""),
                "target_name": metadata.get("target_name", ""),
                "description": metadata.get("description", ""),
                "keywords": metadata.get("keywords", ""),
                "score": _distance_to_similarity(distance),
            })
        return matches

    def query_chunks(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query chunk vector index and return chunk information."""
        results = self._storage.chunk_vectors.query(text=text, n_results=limit) or []
        matches: List[Dict[str, Any]] = []

        for result in results:
            metadatas = self._as_list(result.get("metadatas"))
            ids = self._as_list(result.get("ids"))
            distances = self._as_list(result.get("distances"))
            documents = self._as_list(result.get("documents"))

            max_len = max(
                (len(seq) for seq in (metadatas, ids, distances, documents) if seq),
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

                matches.append({
                    "chunk_uuid": str(chunk_id),
                    "document_id": str(metadata.get("doc_id", "")),
                    "filename": str(metadata.get("filename", "")),
                    "text": document,
                    "score": _distance_to_similarity(distance),
                })
        return matches

    def get_chunk_by_uuid(self, chunk_uuid: str) -> Optional[Dict[str, Any]]:
        """Get a single chunk by its UUID from the database."""
        chunks = self._storage.chunksdb.get_chunk_by_uuid(chunk_uuid)
        return chunks[0] if chunks else None

    def get_chunks_by_uuids(self, chunk_uuids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get multiple chunks by their UUIDs from the database."""
        chunks = self._storage.chunksdb.get_chunks_by_uuids(chunk_uuids)
        return {c["chunk_uuid"]: c for c in chunks}

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        return [value]


def _distance_to_similarity(value: Any) -> float:
    """Convert distance to similarity score."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            try:
                return max(0.0, 1.0 - float(item))
            except (TypeError, ValueError):
                continue
        return 0.0
    try:
        return max(0.0, 1.0 - float(value))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# LLM bridge
# ---------------------------------------------------------------------------

@dataclass
class RetrieveChat:
    """Lightweight async wrapper around llm.Chat."""
    system_prompt: Optional[str] = None

    async def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Run llm.Chat.generate in a worker thread."""
        chat = Chat.singleton()
        return await asyncio.to_thread(
            chat.generate,
            prompt,
            system=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

async def extract_keywords(
    chat: RetrieveChat,
    query: str,
    conversation_history: Optional[List[Tuple[str, str]]] = None,
    history_turns: int = 4,
) -> Tuple[List[str], List[str]]:
    """
    Extract high-level and low-level keywords from the query.

    Args:
        chat: Chat interface for LLM
        query: User query
        conversation_history: Optional conversation history
        history_turns: Number of conversation turns to include

    Returns:
        Tuple of (hl_keywords, ll_keywords)
    """
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = PROMPTS["DEFAULT_LANGUAGE"]
    history_context = get_conversation_turns(conversation_history, history_turns)

    prompt = PROMPTS["keywords_extraction"].format(
        query=query,
        examples=examples,
        language=language,
        history=history_context
    )

    response = await chat.generate(
        prompt,
        temperature=settings.retrieval.llm_temperature,
        max_tokens=settings.retrieval.llm_max_tokens,
    )

    # Parse JSON response
    try:
        result = json.loads(response.strip())
        hl_keywords = result.get("high_level_keywords", [])
        ll_keywords = result.get("low_level_keywords", [])
        return hl_keywords, ll_keywords
    except json.JSONDecodeError as e:
        LOGGER.warning(f"Failed to parse keywords JSON: {e}. Response: {response}")
        return [], []


# ---------------------------------------------------------------------------
# Context building functions
# ---------------------------------------------------------------------------

def get_node_data(
    adapter: StorageAdapter,
    ll_keywords: List[str],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Query entity vectors with low-level keywords and build node-based context.

    Returns:
        Tuple of (node_datas, use_relations, entities_context1, relations_context1)
    """
    # Query entity vectors with concatenated ll_keywords
    ll_query = ", ".join(ll_keywords) if ll_keywords else ""
    if not ll_query:
        return [], [], [], []

    entity_matches = adapter.query_entities(ll_query, limit=top_k)

    graph = adapter.graph
    node_datas = []

    # Build node_datas with rank based on degree
    for match in entity_matches:
        name = match["name"]
        if name not in graph:
            continue

        node_info = dict(graph.nodes[name])
        node_info["name"] = name
        node_info["score"] = match["score"]

        # Calculate total degree (rank)
        degree = graph.degree(name)
        node_info["rank"] = degree

        node_datas.append(node_info)

    # Sort by rank (degree) descending
    node_datas.sort(key=lambda x: x["rank"], reverse=True)

    # Build use_relations: find all unique edges from node_datas
    edge_set = set()
    use_relations = []

    for node in node_datas:
        name = node["name"]
        for neighbor in graph.neighbors(name):
            edge_key = tuple(sorted([name, neighbor]))
            if edge_key not in edge_set:
                edge_set.add(edge_key)

                edge_data = graph.get_edge_data(name, neighbor) or {}
                edge_info = dict(edge_data)
                edge_info["u_source"] = name
                edge_info["u_target"] = neighbor
                edge_info["source_name"] = name
                edge_info["target_name"] = neighbor

                # Calculate edge rank as sum of node ranks
                source_rank = node["rank"]
                target_rank = graph.degree(neighbor) if neighbor in graph else 0
                edge_info["rank"] = source_rank + target_rank

                use_relations.append(edge_info)

    # Sort use_relations by (rank, weight) descending
    use_relations.sort(key=lambda x: (x.get("rank", 0), x.get("weight", 0)), reverse=True)

    # Build entities_context1
    entities_context1 = []
    for idx, node in enumerate(node_datas):
        entities_context1.append({
            "id": idx + 1,
            "entity": node.get("name", ""),
            "type": node.get("type", ""),
            "description": node.get("description", ""),
            "file_path": node.get("filepath", ""),
        })

    # Build relations_context1
    relations_context1 = []
    for idx, edge in enumerate(use_relations):
        relations_context1.append({
            "id": idx + 1,
            "entity1": edge.get("u_source", ""),
            "entity2": edge.get("u_target", ""),
            "type": edge.get("type", ""),
            "description": edge.get("description", ""),
            "file_path": edge.get("filepath", ""),
        })

    return node_datas, use_relations, entities_context1, relations_context1


def get_edge_data(
    adapter: StorageAdapter,
    hl_keywords: List[str],
    top_k: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Query relation vectors with high-level keywords and build edge-based context.

    Returns:
        Tuple of (edge_datas, use_entities, entities_context2, relations_context2)
    """
    # Query relation vectors with concatenated hl_keywords
    hl_query = ", ".join(hl_keywords) if hl_keywords else ""
    if not hl_query:
        return [], [], [], []

    relation_matches = adapter.query_relations(hl_query, limit=top_k)

    graph = adapter.graph
    edge_datas = []

    # Build node_rank_map from node_datas
    node_ids = set([relation_matches[i]["source_name"] for i in range(len(relation_matches))] +
                   [relation_matches[i]["target_name"] for i in range(len(relation_matches))])
    node_rank_map = {id: graph.degree(id) for id in node_ids if id in graph}

    # Build edge_datas with rank based on sum of node ranks
    for match in relation_matches:
        source = match["source_name"]
        target = match["target_name"]

        if not graph.has_edge(source, target):
            continue

        edge_info = dict(graph.get_edge_data(source, target) or {})
        edge_info["u_source"] = source
        edge_info["u_target"] = target
        edge_info["source_name"] = source
        edge_info["target_name"] = target
        edge_info["score"] = match["score"]

        # Calculate edge rank as sum of node ranks
        source_rank = node_rank_map.get(source, graph.degree(source) if source in graph else 0)
        target_rank = node_rank_map.get(target, graph.degree(target) if target in graph else 0)
        edge_info["rank"] = source_rank + target_rank

        edge_datas.append(edge_info)

    # Sort edge_datas by (rank, weight) descending
    edge_datas.sort(key=lambda x: (x.get("rank", 0), x.get("weight", 0)), reverse=True)

    # Build use_entities: find unique nodes in edge_datas
    node_set = set()
    use_entities = []

    for edge in edge_datas:
        for node_name in [edge["u_source"], edge["u_target"]]:
            if node_name not in node_set and node_name in graph:
                node_set.add(node_name)

                node_info = dict(graph.nodes[node_name])
                node_info["name"] = node_name
                node_info["rank"] = node_rank_map.get(node_name, graph.degree(node_name))

                use_entities.append(node_info)

    # Build entities_context2
    entities_context2 = []
    for idx, node in enumerate(use_entities):
        entities_context2.append({
            "id": idx + 1,
            "entity": node.get("name", ""),
            "type": node.get("type", ""),
            "description": node.get("description", ""),
            "file_path": node.get("filepath", ""),
        })

    # Build relations_context2
    relations_context2 = []
    for idx, edge in enumerate(edge_datas):
        relations_context2.append({
            "id": idx + 1,
            "entity1": edge.get("u_source", ""),
            "entity2": edge.get("u_target", ""),
            "type": edge.get("type", ""),
            "description": edge.get("description", ""),
            "file_path": edge.get("filepath", ""),
        })

    return edge_datas, use_entities, entities_context2, relations_context2


def get_vector_context(
    adapter: StorageAdapter,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Get top-k chunks from vector search.

    Returns:
        List of chunks with source_type="vector"
    """
    chunk_matches = adapter.query_chunks(query, limit=top_k)

    all_chunks = []
    for chunk in chunk_matches:
        all_chunks.append({
            "id": chunk["chunk_uuid"],
            "text": chunk["text"],
            "source_type": "vector",
            "score": chunk.get("score", 0.0),
        })

    return all_chunks


def extract_chunks_from_nodes(
    adapter: StorageAdapter,
    node_datas: List[Dict[str, Any]],
    top_k_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Extract chunks from node data.

    For each node, take top_k chunk_uuid. Calculate relation_score based on
    neighboring nodes sharing the same chunk_uuid.

    Returns:
        List of chunks with source_type="entity"
    """
    graph = adapter.graph
    chunk_data = []

    for node_idx, node in enumerate(node_datas):
        name = node["name"]
        chunk_uuids = node.get("chunk_uuids", [])

        # Take top_k chunk_uuids
        for chunk_uuid in chunk_uuids:#[:top_k_chunks]:
            # Calculate relation_score: count neighbors with same chunk_uuid
            relation_score = 0
            if name in graph:
                for neighbor in graph.neighbors(name):
                    neighbor_data = graph.nodes.get(neighbor, {})
                    neighbor_chunks = neighbor_data.get("chunk_uuids", [])
                    if chunk_uuid in neighbor_chunks:
                        relation_score += 1

            chunk_data.append({
                "chunk_uuid": chunk_uuid,
                "index": node_idx,
                "relation_score": relation_score,
            })

    # Sort by index (ascending) then relation_score (descending)
    chunk_data.sort(key=lambda x: (x["index"], -x["relation_score"]))

    # Fetch actual chunk content
    result_chunks = []
    for item in chunk_data:
        chunk = adapter.get_chunk_by_uuid(item["chunk_uuid"])
        if chunk:
            result_chunks.append({
                "id": chunk["chunk_uuid"],
                "text": chunk.get("text", ""),
                "order": item["index"],
                "relation": item["relation_score"],
                "source_type": "entity",
            })

    return result_chunks


def extract_chunks_from_edges(
    adapter: StorageAdapter,
    edge_datas: List[Dict[str, Any]],
    top_k_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Extract chunks from edge data.

    For each edge, take top_k chunk_uuid. Create a dict with chunk_uuid as key
    and edge index as value (first occurrence only).

    Returns:
        List of chunks with source_type="relationship"
    """
    chunk_index_map = {}

    for edge_idx, edge in enumerate(edge_datas):
        chunk_uuids = edge.get("chunk_uuids", [])

        # Take top_k chunk_uuids
        for chunk_uuid in chunk_uuids:#[:top_k_chunks]:
            # Only assign index on first occurrence
            if chunk_uuid not in chunk_index_map:
                chunk_index_map[chunk_uuid] = edge_idx

    # Fetch actual chunk content
    result_chunks = []
    for chunk_uuid, index in chunk_index_map.items():
        chunk = adapter.get_chunk_by_uuid(chunk_uuid)
        if chunk:
            result_chunks.append({
                "id": chunk["chunk_uuid"],
                "text": chunk.get("text", ""),
                "order": index,
                "source_type": "relationship",
            })

    return result_chunks


def rerank(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    top_n: int,
    text_key: str = "text"
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved documents using FlashRank.
    
    Args:
        query: The original user query
        retrieved_docs: List of retrieved document chunks
        top_n: Number of top documents to return after reranking
        text_key: Key in document dict containing text (default: "text")
    
    Returns:
        Reranked list of document indices
    """
    if not retrieved_docs:
        return []
    
    # Initialize ranker
    ranker = Ranker(
        model_name=settings.retrieval.rerank_model_name,
        cache_dir=settings.retrieval.rerank_cache_dir
    )
    
    # Create rerank request
    rerank_request = RerankRequest(
        query=query, 
        passages=[{"text": doc.get(text_key, "")} for doc in retrieved_docs]
    )
    
    # Get reranked results
    results = ranker.rerank(rerank_request)
    
    # Map back to original indices
    reranked_idx = []
    text_list = [doc.get(text_key, "") for doc in retrieved_docs]
    for result in results[:top_n]:
        idx = text_list.index(result["text"])
        reranked_idx.append(idx)
    
    return reranked_idx


def build_context(
    adapter: StorageAdapter,
    query: str,
    hl_keywords: List[str],
    ll_keywords: List[str],
    retrieval_mode: str,
    top_k_entities: int,
    top_k_relations: int,
    top_k_chunks: int,
    top_k_chunk_per_entity: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build complete context for the query.

    Returns:
        Tuple of (entities_context, relations_context, all_chunks)
    """

    # 2.a) Get node data
    node_datas, use_relations, entities_context1, relations_context1 = get_node_data(
        adapter, ll_keywords, top_k_entities
    ) if retrieval_mode in ("local", "hybrid", "mix") else ([], [], [], [])

    # 2.b) Get edge data
    edge_datas, use_entities, entities_context2, relations_context2 = get_edge_data(
        adapter, hl_keywords, top_k_relations
    ) if retrieval_mode in ("global", "hybrid", "mix") else ([], [], [], [])

    # 2.c) Get vector context
    vector_chunks = get_vector_context(adapter, query, top_k_chunks) if retrieval_mode in ("mix", "naive") else []

    # Combine contexts
    seen = {e["entity"] for e in entities_context1}
    unseen = [e for e in entities_context2 if e["entity"] not in seen]
    entities_context = entities_context1 + unseen
    for idx, entity in enumerate(entities_context):
        entity["id"] = str(idx + 1)

    seen = {(r["entity1"], r["entity2"]) for r in relations_context1}
    unseen = [r for r in relations_context2 if (r["entity1"], r["entity2"]) not in seen]
    relations_context = relations_context1 + unseen
    for idx, relation in enumerate(relations_context):
        relation["id"] = str(idx + 1)

    # Get original node and edge datas (unique only)
    original_node_datas = list(node_datas)
    node_names = {n["name"] for n in original_node_datas}
    for node in use_entities:
        if node["name"] not in node_names:
            original_node_datas.append(node)
            node_names.add(node["name"])

    original_edge_datas = list(use_relations)
    edge_keys = {tuple(sorted([e["u_source"], e["u_target"]])) for e in original_edge_datas}
    for edge in edge_datas:
        edge_key = tuple(sorted([edge["u_source"], edge["u_target"]]))
        if edge_key not in edge_keys:
            original_edge_datas.append(edge)
            edge_keys.add(edge_key)

    # 3.a) Extract chunks from nodes
    entity_chunks = extract_chunks_from_nodes(adapter, original_node_datas, top_k_chunk_per_entity)

    # 3.b) Extract chunks from edges
    relationship_chunks = extract_chunks_from_edges(adapter, original_edge_datas, top_k_chunk_per_entity)

    # 4) Combine and deduplicate chunks
    all_chunks = list(vector_chunks)
    all_chunks.extend(entity_chunks)
    all_chunks.extend(relationship_chunks)
    
    # Deduplicate by chunk UUID
    seen_uuids = set()
    deduplicated_chunks = []
    for chunk in all_chunks:
        chunk_id = chunk.get("id")
        if chunk_id and chunk_id not in seen_uuids:
            seen_uuids.add(chunk_id)
            deduplicated_chunks.append(chunk)
    
    #Rerank by top_k
    if settings.retrieval.enable_rerank and query and deduplicated_chunks:
        rerank_top_k = settings.retrieval.rerank_top_k or len(deduplicated_chunks)
        deduplicated_chunk_ids = rerank(
            query=query,
            retrieved_docs=deduplicated_chunks,
            top_n=rerank_top_k,
        )
        print(f"Reranked chunks: kept {len(deduplicated_chunk_ids)} chunks after reranking, it was {len(deduplicated_chunks)} before reranking")
        deduplicated_chunks = [deduplicated_chunks[idx] for idx in deduplicated_chunk_ids]

    # Apply chunk_top_k limit if needed
    if settings.retrieval.chunk_top_k is not None and settings.retrieval.chunk_top_k > 0:
        if len(deduplicated_chunks) > settings.retrieval.chunk_top_k:
            deduplicated_chunks = deduplicated_chunks[: settings.retrieval.chunk_top_k]
            LOGGER.debug(
                f"Chunk top-k limiting: kept {len(deduplicated_chunks)} chunks (chunk_top_k={settings.retrieval.chunk_top_k})"
            )

    # Truncate chunk text by chunk_window_tokens
    # if settings.retrieval.truncate_chunks and settings.retrieval.chunk_window_tokens > 0:
    #     model = settings.retrieval.tiktoken_model
    #     deduplicated_chunks = truncate_by_tokens(deduplicated_chunks, settings.retrieval.chunk_window_tokens, model)
    #     print(f"Truncated chunks: kept {len(deduplicated_chunks)} chunks after truncation")
    

    return entities_context, relations_context, deduplicated_chunks


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def lightrag_prompt(
    entities_context: List[Dict[str, Any]],
    relations_context: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
    history: Optional[List[Tuple[str, str]]]=None,
) -> str:
    """Generate a system prompt for LightRAG."""

    kg_context = f"""-----Entities(KG)-----

```json
{json.dumps(entities_context, ensure_ascii=False)}
```

-----Relationships(KG)-----

```json
{json.dumps(relations_context, ensure_ascii=False)}
```

"""
    naive_context = f"""-----Document Chunks(DC)-----
```json
{json.dumps(all_chunks, ensure_ascii=False)}
```

"""
    
    context = naive_context if settings.retrieval.light_mode == "naive" else kg_context + naive_context
    history_context = get_conversation_turns(history)
    user_prompt = PROMPTS["DEFAULT_USER_PROMPT"]
    sys_prompt_template = PROMPTS["rag_response"] if settings.retrieval.light_mode != "naive" else PROMPTS["rag_response_naive"]
    sys_prompt = sys_prompt_template.format(
        context_data=context,
        response_type=settings.retrieval.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    return sys_prompt


# ---------------------------------------------------------------------------
# LightRAG entry point
# ---------------------------------------------------------------------------

class LightRAG:
    """
    LightRAG retriever that integrates storage and LLM to answer questions.

    Example usage:
        rag = LightRAG(storage_paths=StoragePaths(...), system_prompt="You are a helpful assistant.")
        result = rag.retrieve("What is the capital of France?")
    """

    def __init__(
        self,
        *,
        storage_paths: Optional[StoragePaths] = None,
        system_prompt: Optional[str] = None,
        log_file: str = "LightRAG.log",
    ) -> None:
        set_logger(log_file)
        LOGGER.info("Initialising LightRAG retriever")
        self._storage = StorageAdapter(paths=storage_paths)
        self._chat = RetrieveChat(system_prompt=system_prompt)

    async def aretrieve(
        self,
        question: str,
        conversation_history: Optional[List[Tuple[str, str]]] = None,
    ) -> RetrievalResult:
        """Asynchronous retrieval method."""
        LOGGER.info("Running LightRAG retrieval for query: %s", question)

        # 1) Extract keywords
        hl_keywords, ll_keywords = await extract_keywords(
            self._chat,
            question,
            conversation_history,
            settings.retrieval.history_turns
        )    

        LOGGER.debug(f"Extracted keywords - HL: {hl_keywords}, LL: {ll_keywords}")
        retrieval_mode = settings.retrieval.light_mode
        # Handle empty keywords
        if hl_keywords == [] and ll_keywords == []:
            LOGGER.warning("low_level_keywords and high_level_keywords is empty")
            return PROMPTS["fail_response"]
        if ll_keywords == [] and retrieval_mode in ["local", "hybrid"]:
            LOGGER.warning(f"low_level_keywords is empty, switching from {retrieval_mode} mode to global mode")
            retrieval_mode = "global"
        if hl_keywords == [] and retrieval_mode in ["global", "hybrid"]:
            LOGGER.warning(f"high_level_keywords is empty, switching from {retrieval_mode} mode to local mode")
            retrieval_mode = "local"

        # 2-4) Build context
        entities_context, relations_context, all_chunks = build_context(
            self._storage,
            question,
            hl_keywords,
            ll_keywords,
            retrieval_mode=retrieval_mode,
            top_k_entities=settings.retrieval.entity_top_k,
            top_k_relations=settings.retrieval.relation_top_k,
            top_k_chunks=settings.retrieval.chunk_top_k,
            top_k_chunk_per_entity=3,  # configurable
        )

        # Format context for prompt
        lightrag_sys_prompt = lightrag_prompt(
            entities_context=entities_context,
            relations_context=relations_context,
            all_chunks=all_chunks,
            history=conversation_history,
        )

        # Generate answer
        chat = RetrieveChat(system_prompt=lightrag_sys_prompt)
        answer = await chat.generate(
            prompt=question,
            max_tokens=settings.retrieval.llm_max_tokens,
            temperature=settings.retrieval.llm_temperature,
        )

        return RetrievalResult(
            answer=answer,
            entities_context=entities_context,
            relations_context=relations_context,
            all_chunks=all_chunks,
            hl_keywords=hl_keywords,
            ll_keywords=ll_keywords,
        )

    def retrieve(
        self,
        question: str,
        conversation_history: Optional[List[Tuple[str, str]]] = None,
    ) -> RetrievalResult:
        """Synchronous helper that creates an event loop if needed."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aretrieve(question, conversation_history))
        else:
            raise RuntimeError(
                "retrieve() cannot be used when an event loop is already running. "
                "Use 'await aretrieve(...)' instead."
            )


__all__ = [
    "LightRAG",
    "StorageAdapter",
    "StoragePaths",
    "RetrievalResult",
]

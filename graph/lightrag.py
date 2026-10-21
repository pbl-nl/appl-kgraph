"""LightRAG retriever integrated with the ingestion storage backend."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import networkx as nx
from flashrank import Ranker, RerankRequest

from settings import settings, RetrievalSettings
from db_storage import Storage
from db_storage import StoragePaths
from llm import Chat
from prompts import PROMPTS
from logging_utils import configure_file_logger
from graph_pickle import load_or_build_graph_snapshot
from project_paths import ProjectPaths
from answer_generation import AsyncTextAnswerGenerator, generate_answer_async, record_chat_audit
from query_analysis import AsyncKeywordQueryAnalyzer, analyze_query_async
import retrieval as stages
from schemas import AnswerResult, retrieval_candidates_from_legacy
from context_serialization import context_from_candidates, light_prompt_rows


LOGGER = logging.getLogger("LightRAG")


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


def _chunk_matches_allowed(chunk: Dict[str, Any], allowed_documents: Optional[set[str]]) -> bool:
    if not allowed_documents:
        return True
    filename = str(chunk.get("filename", "") or "").strip()
    document_id = str(chunk.get("document_id", "") or "").strip()
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
    entities, relations, chunks = light_prompt_rows(result.context)
    return "\n".join(["== ENTITY CONTEXT ==", json.dumps(entities, ensure_ascii=False),
                      "== RELATION CONTEXT ==", json.dumps(relations, ensure_ascii=False),
                      "== ALL CHUNKS ==", json.dumps(chunks, ensure_ascii=False)])

# ---------------------------------------------------------------------------
# Logging and token helpers
# ---------------------------------------------------------------------------

def set_logger(log_file: Path) -> None:
    configure_file_logger(
        "LightRAG",
        log_file=log_file,
        level=settings.logging.internal_log_level,
        enabled=settings.logging.internal_logging_enabled,
    )


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
    if not history or max_turns <= 0:
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
class GraphSnapshot:
    """A cached view of the knowledge graph stored in SQLite."""
    graph: nx.Graph


# ---------------------------------------------------------------------------
# Storage adapter (reusing from pathrag.py pattern)
# ---------------------------------------------------------------------------

class StorageAdapter:
    """High level helper around the ingestion Storage facade."""

    def __init__(
        self,
        paths: Optional[StoragePaths] = None,
        *,
        graph_pickle_path: Optional[Path] = None,
    ):
        self._storage = Storage(paths=paths)
        self._storage.init()
        self._graph_snapshot: Optional[GraphSnapshot] = None
        self._graph_pickle_path = graph_pickle_path

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

    def query_entities(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query entity vector index and return entity information with IDs."""
        results = self._storage.search_entities(text=text, n_results=limit) or []
        matches: List[Dict[str, Any]] = []
        if not results:
            return matches

        for metadata in results:
            if not isinstance(metadata, dict):
                continue
            matches.append({
                "id": metadata.get("name", ""),
                "name": metadata.get("name", ""),
                "type": metadata.get("type"),
                "description": metadata.get("description", ""),
                "score": float(metadata.get("score", 0.0) or 0.0),
            })
        return matches

    def query_relations(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query relation vector index and return relation information with IDs."""
        results = self._storage.search_relations(text=text, n_results=limit) or []
        matches: List[Dict[str, Any]] = []
        if not results:
            return matches

        for metadata in results:
            if not isinstance(metadata, dict):
                continue
            matches.append({
                "id": f"{metadata.get('source_name', '')}::{metadata.get('target_name', '')}",
                "source_name": metadata.get("source_name", ""),
                "target_name": metadata.get("target_name", ""),
                "description": metadata.get("description", ""),
                "keywords": metadata.get("keywords", ""),
                "score": float(metadata.get("score", 0.0) or 0.0),
            })
        return matches

    def query_chunks(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query chunk vector index and return chunk information."""
        results = self._storage.search_chunks(text=text, n_results=limit) or []
        matches: List[Dict[str, Any]] = []
        for metadata in results:
            if not isinstance(metadata, dict):
                continue
            matches.append({
                **metadata,
                "chunk_uuid": str(metadata.get("chunk_uuid", "")),
                "document_id": str(metadata.get("doc_id", "")),
                "filename": str(metadata.get("filename", "")),
                "text": str(metadata.get("text", "")),
                "score": float(metadata.get("score", 0.0) or 0.0),
            })
        return matches

    def get_chunk_by_uuid(self, chunk_uuid: str) -> Optional[Dict[str, Any]]:
        """Get a single chunk by its UUID from the database."""
        chunks = self._storage.chunksdb.get_chunk_by_uuid(chunk_uuid)
        return chunks[0] if chunks else None

    def close(self) -> None:
        close_method = getattr(self._storage, "close", None)
        if callable(close_method):
            close_method()
        self._graph_snapshot = None


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
    *, config: RetrievalSettings | None = None,
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
    config = config or settings.retrieval
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = settings.prompts.default_language
    history_context = get_conversation_turns(conversation_history, history_turns)

    prompt = PROMPTS["keywords_extraction"].format(
        query=query,
        examples=examples,
        language=language,
        history=history_context
    )

    response = await chat.generate(
        prompt,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
    )

    # Parse JSON response
    try:
        result = json.loads(response.strip())
        if not isinstance(result, dict):
            return [], []
        hl_keywords = result.get("high_level_keywords", [])
        ll_keywords = result.get("low_level_keywords", [])
        if not all(isinstance(values, list) and all(isinstance(value, str) for value in values)
                   for values in (hl_keywords, ll_keywords)):
            return [], []
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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Query entity vectors with low-level keywords and build node-based context.

    Returns:
        Tuple of (node_datas, use_relations) before context construction
    """
    # Query entity vectors with concatenated ll_keywords
    ll_query = ", ".join(ll_keywords) if ll_keywords else ""
    if not ll_query:
        return [], []

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

    return node_datas, use_relations


def get_edge_data(
    adapter: StorageAdapter,
    hl_keywords: List[str],
    top_k: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Query relation vectors with high-level keywords and build edge-based context.

    Returns:
        Tuple of (edge_datas, use_entities) before context construction
    """
    # Query relation vectors with concatenated hl_keywords
    hl_query = ", ".join(hl_keywords) if hl_keywords else ""
    if not hl_query:
        return [], []

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

    return edge_datas, use_entities


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
            **chunk,
            "id": chunk["chunk_uuid"],
            "chunk_uuid": chunk["chunk_uuid"],
            "document_id": chunk.get("document_id", ""),
            "filename": chunk.get("filename", ""),
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
        for chunk_uuid in chunk_uuids[:top_k_chunks]:
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
                **chunk,
                "id": chunk["chunk_uuid"],
                "chunk_uuid": chunk["chunk_uuid"],
                "document_id": chunk.get("doc_id", ""),
                "filename": chunk.get("filename", ""),
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
        for chunk_uuid in chunk_uuids[:top_k_chunks]:
            # Only assign index on first occurrence
            if chunk_uuid not in chunk_index_map:
                chunk_index_map[chunk_uuid] = edge_idx

    # Fetch actual chunk content
    result_chunks = []
    for chunk_uuid, index in chunk_index_map.items():
        chunk = adapter.get_chunk_by_uuid(chunk_uuid)
        if chunk:
            result_chunks.append({
                **chunk,
                "id": chunk["chunk_uuid"],
                "chunk_uuid": chunk["chunk_uuid"],
                "document_id": chunk.get("doc_id", ""),
                "filename": chunk.get("filename", ""),
                "text": chunk.get("text", ""),
                "order": index,
                "source_type": "relationship",
            })

    return result_chunks


def rerank(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    top_n: int,
    text_key: str = "text",
    *, config: RetrievalSettings | None = None,
) -> List[int]:
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
    
    # Initialize ranker using the query settings captured before retrieval.
    config = config or settings.retrieval
    ranker = Ranker(
        model_name=config.rerank_model_name,
        cache_dir=config.rerank_cache_dir
    )
    
    # Create rerank request
    rerank_request = RerankRequest(
        query=query, 
        passages=[{"id": index, "text": doc.get(text_key, "")} for index, doc in enumerate(retrieved_docs)]
    )
    
    # Get reranked results
    results = ranker.rerank(rerank_request)
    
    # IDs preserve distinct evidence records even when their text is identical.
    return [int(result["id"]) for result in results[:top_n]]



class LightCandidateRetriever:
    """Vector/graph evidence lookup; leaves deduplication and formatting to context."""
    def __init__(self, storage):
        self.storage = storage

    def retrieve(self, plan):
        config = RetrievalSettings(**plan.settings)
        if plan.metadata.get("empty_keywords"):
            return retrieval_candidates_from_legacy(plan, {})
        adapter = self.storage
        nodes, local_edges = get_node_data(adapter, plan.low_level_keywords, config.entity_top_k) if plan.mode in ("local", "hybrid", "mix") else ([], [])
        edges, global_nodes = get_edge_data(adapter, plan.high_level_keywords, config.relation_top_k) if plan.mode in ("global", "hybrid", "mix") else ([], [])
        nodes = nodes + global_nodes
        edges = local_edges + edges
        allowed = _normalise_document_filter(plan.metadata.get("allowed_document_names"))
        if allowed:
            nodes = [row for row in nodes if _filepath_matches_allowed(row.get("filepath"), allowed)]
            edges = [row for row in edges if _filepath_matches_allowed(row.get("filepath"), allowed)]
        chunks = get_vector_context(adapter, plan.query, config.chunk_top_k) if plan.mode in ("mix", "naive") else []
        chunks += extract_chunks_from_nodes(adapter, nodes, config.top_k_chunk_per_entity)
        chunks += extract_chunks_from_edges(adapter, edges, config.top_k_chunk_per_entity)
        if allowed:
            chunks = [row for row in chunks if _chunk_matches_allowed(row, allowed)]
        return retrieval_candidates_from_legacy(plan, {
            "entities": [{**row, "source_ids": tuple(row.get("chunk_uuids", ())), "score": row.get("score", 0.0)} for row in nodes],
            "relations": [{**row, "source_ids": tuple(row.get("chunk_uuids", ())), "score": row.get("score", 0.0)} for row in edges],
            "chunks": chunks,
        })


class LightContextBuilder:
    """Pure deterministic context assembly from supplied evidence."""
    def build(self, plan, candidates):
        def unique(items, key):
            seen = set()
            result = []
            for item in items:
                identity = key(item)
                if identity not in seen:
                    seen.add(identity)
                    result.append(item)
            return result
        candidates = replace(candidates,
            entities=unique(candidates.entities, lambda item: item.name),
            relations=unique(candidates.relations, lambda item: tuple(sorted((item.source_name, item.target_name)))),
            chunks=unique(candidates.chunks, lambda item: item.chunk_uuid))
        return context_from_candidates(candidates)


class LightContextReranker:
    """Rank supplied context independently, then apply the final chunk limit."""
    def rerank(self, context):
        config = RetrievalSettings(**context.query_plan.settings)
        chunks = list(context.chunks)
        if config.enable_rerank and context.query_plan.query and chunks:
            rows = light_prompt_rows(context)[2]
            indices = rerank(context.query_plan.query, rows, config.rerank_top_k or len(chunks), config=config)
            chunks = [chunks[index] for index in indices]
        limit = config.chunk_top_k
        if limit is not None and limit > 0:
            chunks = chunks[:limit]
        return replace(context, chunks=chunks)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def lightrag_prompt(
    entities_context: List[Dict[str, Any]],
    relations_context: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
    history: Optional[List[Tuple[str, str]]]=None,
    *, config: RetrievalSettings, mode: str,
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
    
    context = naive_context if mode == "naive" else kg_context + naive_context
    history_context = get_conversation_turns(history, config.history_turns)
    user_prompt = settings.prompts.default_user_prompt
    sys_prompt_template = (
        PROMPTS["lightrag_response"]
        if mode != "naive"
        else PROMPTS["naive_rag_response"]
    )
    sys_prompt = sys_prompt_template.format(
        context_data=context,
        content_data=context,
        response_type=config.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    return sys_prompt


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
# LightRAG entry point
# ---------------------------------------------------------------------------

async def generate_light_answer(query, context, history, *, chat_factory=None):
    if context.query_plan.metadata.get("empty_keywords"):
        return PROMPTS["fail_response"]
    config = RetrievalSettings(**context.query_plan.settings)
    entities, relations, chunks = light_prompt_rows(context)
    prompt = lightrag_prompt(entities, relations, chunks, history=history, config=config, mode=context.query_plan.mode)
    chat = (chat_factory or RetrieveChat)(system_prompt=prompt)
    return await chat.generate(prompt=query, max_tokens=config.llm_max_tokens,
                               temperature=config.llm_temperature)


def make_answer_generator(chat_factory=None):
    """Create an answer adapter that needs supplied context and a model, without storage."""
    return AsyncTextAnswerGenerator(
        lambda query, context, history: generate_light_answer(query, context, history, chat_factory=chat_factory),
        model=_retrieval_model_metadata(),
    )


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
                project_paths.lightrag_log_file
                if project_paths is not None
                else Path("LightRAG.log")
            )
        )
        set_logger(effective_log_file)
        LOGGER.info("Initialising LightRAG retriever")
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

    async def aretrieve(self, question, conversation_history=None, allowed_document_names=None) -> AnswerResult:
        config = settings.retrieval
        analyzer = self._query_analyzer or AsyncKeywordQueryAnalyzer(
            lambda query, history: extract_keywords(self._chat, query, history, config.history_turns, config=config),
            mode=config.light_mode, settings=asdict(config))
        plan = await analyze_query_async(question, conversation_history or [], analyzer)
        # Naive retrieval does not require graph keywords.
        empty = not plan.high_level_keywords and not plan.low_level_keywords and plan.mode != "naive"
        plan = replace(plan, metadata={**plan.metadata, "empty_keywords": empty,
            "allowed_document_names": list(allowed_document_names) if allowed_document_names is not None else None})
        candidates = stages.retrieve_candidates(plan, self._candidate_retriever or LightCandidateRetriever(self._storage))
        context = stages.build_context(plan, candidates, self._context_builder or LightContextBuilder())
        context = stages.rerank_context(context, self._reranker or LightContextReranker())
        result = await generate_answer_async(question, context, plan.history,
            self._answer_generator or make_answer_generator())
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
        else:
            raise RuntimeError(
                "retrieve() cannot be used when an event loop is already running. "
                "Use 'await aretrieve(...)' instead."
            )


__all__ = [
    "LightRAG",
    "StorageAdapter",
    "StoragePaths",
    "AnswerResult",
]

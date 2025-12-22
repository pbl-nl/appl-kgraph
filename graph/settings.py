# graph/settings.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Literal
from dotenv import load_dotenv


# ─────────────────────────────────────────────────────────────
# Small helpers to parse environment variables robustly
# ─────────────────────────────────────────────────────────────

def _strip_quotes(val: Optional[str]) -> Optional[str]:
    """
    Removes surrounding quotes from environment variable values.

    Args:
        val (Optional[str]): The value to process.

    Returns:
        Optional[str]: The value with quotes stripped, or None if input was None.
    """
    if val is None:
        return None
    v = val.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v

def env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Reads a string environment variable with quote stripping.

    Args:
        key (str): The environment variable name.
        default (Optional[str], optional): Default value if not found. Defaults to None.

    Returns:
        Optional[str]: The environment variable value or default.
    """
    val = os.getenv(key)
    return _strip_quotes(val) if val is not None else default

def env_int(key: str, default: int) -> int:
    """
    Reads an integer environment variable with fallback to default.

    Args:
        key (str): The environment variable name.
        default (int): Default value if not found or invalid.

    Returns:
        int: The parsed integer value or default.
    """
    val = env_str(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default

def env_float(key: str, default: float) -> float:
    """
    Reads a float environment variable with fallback to default.

    Args:
        key (str): The environment variable name.
        default (float): Default value if not found or invalid.

    Returns:
        float: The parsed float value or default.
    """
    val = env_str(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default

def env_bool(key: str, default: bool) -> bool:
    """
    Reads a boolean environment variable with fallback to default.

    Args:
        key (str): The environment variable name.
        default (bool): Default value if not found.

    Returns:
        bool: True if value is in {"1", "true", "yes", "y", "t"}, otherwise default.
    """
    val = env_str(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "t"}

def env_list(key: str, default_csv: str, sep: str = ",") -> List[str]:
    """
    Reads a delimited list environment variable.

    Args:
        key (str): The environment variable name.
        default_csv (str): Default comma-separated value string.
        sep (str, optional): Delimiter character. Defaults to ",".

    Returns:
        List[str]: List of trimmed non-empty values.
    """
    raw = env_str(key, default_csv) or ""
    return [x.strip() for x in raw.split(sep) if x.strip()]


# ─────────────────────────────────────────────────────────────
# Settings sections
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProviderSettings:
    """
    Provider selection + credentials.
    Choose provider via: LLM_PROVIDER = "openai" or "azure"
    """
    provider: Literal["openai", "azure"] = "openai"

    # OpenAI (direct)
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None         # optional (for proxies / compatible servers)
    openai_llm_model: Optional[str] = None        # e.g. gpt-4o-mini
    openai_embeddings_model: Optional[str] = None # e.g. text-embedding-3-small

    # Azure OpenAI
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2024-02-15-preview"
    azure_llm_deployment: Optional[str] = None
    azure_embeddings_deployment: Optional[str] = None


@dataclass(frozen=True)
class ChatGenerationSettings:
    """
    Default knobs for chat completions used across the app.
    """
    temperature: float = 0.0
    max_tokens: int = 2048

@dataclass(frozen=True)
class LLMPerformanceSettings:
    """
    Performance-related knobs for LLM calls.
    """
    max_concurrency: int = 6              # num of parallel requests
    cache_enabled: bool = True
    cache_max_age_hours: int = 720        # 30 days

@dataclass(frozen=True)
class EmbeddingSettings:
    """
    Embeddings model + batch behavior for vectorization.
    (Model names come from ProviderSettings; this holds cross-cutting knobs.)
    """
    batch_size: int = 64

@dataclass(frozen=True)
class PromptFormattingSettings:
    """
    Formatting conventions your prompts expect & rely on.
    Used by extractor prompts and downstream parsers.
    """
    default_language: str = "English"
    tuple_delimiter: str = "<|>"
    record_delimiter: str = "##"
    completion_delimiter: str = "<|COMPLETE|>"
    default_entity_types: List[str] = None  # filled in loader

@dataclass(frozen=True)
class IngestionMergeSettings:
    """
    How we concatenate multi-source fields before optional summarization.
    delimiter: separates repeated descriptions/keywords/source_ids/filepaths
    description_segment_limit: threshold after which we summarize with LLM
    """
    delimiter: str = "||"
    description_segment_limit: int = 5

@dataclass(frozen=True)
class ChunkingSettings:
    """
    Default chunking policy for page-aware, sentence-preserving chunker.
    """
    max_chars: int = 1200
    overlap_chars: int = 200
    include_overlap_in_limit: bool = True
    join_with: str = " "

@dataclass(frozen=True)
class StoragePaths:
    """
    Where we store SQLite DBs and Chroma collections.
    """
    documents_db: str = "./storage/documents.sqlite"
    chunks_db: str = "./storage/chunks.sqlite"
    graph_db: str = "./storage/graph.sqlite"

    chroma_chunks: str = "./storage/chroma_chunks"
    chroma_entities: str = "./storage/chroma_entities"
    chroma_relations: str = "./storage/chroma_relations"

@dataclass(frozen=True)
class RetrievalSettings:
    """
    Settings for retrieval operations (e.g. how many results to return).
    """
    entity_top_k: int = 5
    relation_top_k: int = 5
    chunk_top_k: int = 6
    graph_depth: int = 2
    graph_windows: int = 3
    chunk_windows: int = 3
    graph_window_tokens: int = 512
    chunk_window_tokens: int = 512
    tiktoken_model: str = "gpt-4o-mini"  # for token counting
    llm_max_tokens: int = 512
    llm_temperature: float = 0.0
    history_turns: int = 4

    # --- Hybrid/global (i.e. do we use paths & relations?) ---
    hybrid_use_paths_for_global: bool = True # whether to build global context from paths
    hybrid_use_relations_for_global: bool = False   # whether to build global context from relations
    global_max_windows: int = 4 # max global context windows
    global_window_tokens: int = 512 # token cap per global window

    # --- Local toggles (i.e. do we use chunks and local neighborhoods?) ---
    use_local_chunks: bool = True # whether to use local chunks
    use_local_graph: bool = True # whether to use local graph
    local_max_windows: int = 6 # cap total local windows

    # --- PathRAG-specific retrieval settings ---
    path_use_top_entities: int = 5   # limit the number of entity seeds considered
    path_max_depth: int = 3          # search up to 3 hops
    path_threshold: float = 0.3      # propagation threshold
    path_alpha: float = 0.8          # propagation decay
    path_max_windows: int = 5        # how many path windows to emit
    path_window_tokens: int = 512    # tokens per path window

    # --- LightRAG-specific retrieval settings ---
    light_mode: str = "mix"       # 'local', 'global', 'hybrid', 'mix', 'naive'
    response_type: str = "Single Paragraph" #'Multiple Paragraphs', 'Single Paragraph'
    rerank_top_k: int = 20
    enable_rerank: bool = True
    rerank_cache_dir: str = "./flashrank_model"  # directory for FlashRank model cache
    rerank_model_name: str = "ms-marco-MultiBERT-L-12"  # model name for reranking
    truncate_chunks: bool = False  # whether to truncate chunks by token limit


@dataclass(frozen=True)
class Settings:
    """
    Full application settings bundle.
    """
    provider: ProviderSettings
    chat: ChatGenerationSettings
    llmperf: LLMPerformanceSettings
    embeddings: EmbeddingSettings
    prompts: PromptFormattingSettings
    ingestion: IngestionMergeSettings
    chunking: ChunkingSettings
    storage: StoragePaths
    retrieval: RetrievalSettings



# ─────────────────────────────────────────────────────────────
# Loader / validator
# ─────────────────────────────────────────────────────────────

def load_settings() -> Settings:
    """
    Loads and validates all application settings from environment variables.

    Reads from .env file, parses configuration for provider, LLM, embeddings, chunking,
    storage paths, and retrieval settings. Validates required fields based on provider.

    Returns:
        Settings: A fully configured Settings object.

    Raises:
        RuntimeError: If required environment variables are missing or invalid.
    """
    load_dotenv()  # called once at startup

    # Provider selection
    provider_name = (env_str("LLM_PROVIDER", "openai") or "openai").strip().lower()
    if provider_name not in {"openai", "azure"}:
        raise RuntimeError("LLM_PROVIDER must be 'openai' or 'azure'.")

    provider = ProviderSettings(
        provider=provider_name,  # type: ignore[arg-type]
        # OpenAI
        openai_api_key=env_str("OPENAI_API_KEY"),
        openai_base_url=env_str("OPENAI_BASE_URL"),
        openai_llm_model=env_str("OPENAI_LLM_MODEL"),
        openai_embeddings_model=env_str("OPENAI_EMBEDDINGS_MODEL"),
        # Azure
        azure_api_key=env_str("AZURE_OPENAI_API_KEY"),
        azure_endpoint=env_str("AZURE_OPENAI_ENDPOINT"),
        azure_api_version=env_str("AZURE_OPENAI_API_VERSION", "2024-02-15-preview") or "2024-02-15-preview",
        azure_llm_deployment=env_str("AZURE_OPENAI_LLM_DEPLOYMENT_NAME"),
        azure_embeddings_deployment=env_str("AZURE_OPENAI_EMB_DEPLOYMENT_NAME"),
    )

    # Validate provider-specific required fields
    if provider.provider == "openai":
        if not provider.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        if not provider.openai_llm_model:
            raise RuntimeError("OPENAI_LLM_MODEL is required when LLM_PROVIDER=openai.")
        if not provider.openai_embeddings_model:
            raise RuntimeError("OPENAI_EMBEDDINGS_MODEL is required when LLM_PROVIDER=openai.")
    else:
        # azure
        missing = []
        if not provider.azure_api_key: missing.append("AZURE_OPENAI_API_KEY")
        if not provider.azure_endpoint: missing.append("AZURE_OPENAI_ENDPOINT")
        if not provider.azure_llm_deployment: missing.append("AZURE_OPENAI_LLM_DEPLOYMENT_NAME")
        if not provider.azure_embeddings_deployment: missing.append("AZURE_OPENAI_EMB_DEPLOYMENT_NAME")
        if missing:
            raise RuntimeError(f"When LLM_PROVIDER=azure, set required variables: {', '.join(missing)}")

    chat = ChatGenerationSettings(
        temperature=env_float("CHAT_TEMPERATURE", 0.0),
        max_tokens=env_int("CHAT_MAX_TOKENS", 2048),
    )

    llmperf = LLMPerformanceSettings(
        max_concurrency=env_int("LLM_MAX_CONCURRENCY", 6),
        cache_enabled=env_bool("LLM_CACHE_ENABLED", True),
        cache_max_age_hours=env_int("LLM_CACHE_MAX_AGE_HOURS", 720),
    )

    embeddings = EmbeddingSettings(
        batch_size=env_int("EMBEDDING_BATCH_SIZE", 64),
    )

    prompts = PromptFormattingSettings(
        default_language=env_str("PROMPT_DEFAULT_LANGUAGE", "English") or "English",
        tuple_delimiter=env_str("PROMPT_TUPLE_DELIMITER", "<|>") or "<|>",
        record_delimiter=env_str("PROMPT_RECORD_DELIMITER", "##") or "##",
        completion_delimiter=env_str("PROMPT_COMPLETION_DELIMITER", "<|COMPLETE|>") or "<|COMPLETE|>",
        default_entity_types=env_list(
            "PROMPT_DEFAULT_ENTITY_TYPES",
            "organization,person,geo,event,category"
        ),
    )

    ingestion = IngestionMergeSettings(
        delimiter=env_str("MERGE_DELIMITER", "||") or "||",
        description_segment_limit=env_int("DESCRIPTION_SEGMENT_LIMIT", 5),
    )

    chunking = ChunkingSettings(
        max_chars=env_int("CHUNK_MAX_CHARS", 1200),
        overlap_chars=env_int("CHUNK_OVERLAP_CHARS", 200),
        include_overlap_in_limit=env_bool("CHUNK_INCLUDE_OVERLAP_IN_LIMIT", True),
        join_with=env_str("CHUNK_JOIN_WITH", " ") or " ",
    )

    storage = StoragePaths(
        documents_db=env_str("DOCUMENTS_DB_PATH", "./storage/documents.sqlite") or "./storage/documents.sqlite",
        chunks_db=env_str("CHUNKS_DB_PATH", "./storage/chunks.sqlite") or "./storage/chunks.sqlite",
        graph_db=env_str("GRAPH_DB_PATH", "./storage/graph.sqlite") or "./storage/graph.sqlite",
        chroma_chunks=env_str("CHROMA_CHUNKS_PATH", "./storage/chroma_chunks") or "./storage/chroma_chunks",
        chroma_entities=env_str("CHROMA_ENTITIES_PATH", "./storage/chroma_entities") or "./storage/chroma_entities",
        chroma_relations=env_str("CHROMA_RELATIONS_PATH", "./storage/chroma_relations") or "./storage/chroma_relations",
    )

    retrieval = RetrievalSettings(
        entity_top_k=env_int("RETRIEVAL_ENTITY_TOP_K", 5),
        relation_top_k=env_int("RETRIEVAL_RELATION_TOP_K", 5),
        chunk_top_k=env_int("RETRIEVAL_CHUNK_TOP_K", 6),
        graph_depth=env_int("RETRIEVAL_GRAPH_DEPTH", 2),
        graph_windows=env_int("RETRIEVAL_GRAPH_WINDOWS", 3),
        chunk_windows=env_int("RETRIEVAL_CHUNK_WINDOWS", 3),
        graph_window_tokens=env_int("RETRIEVAL_GRAPH_WINDOW_TOKENS", 512),
        chunk_window_tokens=env_int("RETRIEVAL_CHUNK_WINDOW_TOKENS", 512),
        tiktoken_model = env_str("RETRIEVAL_TIKTOKEN_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        llm_max_tokens = env_int("RETRIEVAL_LLM_MAX_TOKENS", 512),
        llm_temperature = env_float("RETRIEVAL_LLM_TEMPERATURE", 0.0),
        history_turns= env_int("RETRIEVAL_HISTORY_TURNS", 4),
        # Hybrid/global
        hybrid_use_paths_for_global = env_bool("RETRIEVAL_HYBRID_USE_PATHS_FOR_GLOBAL", True),
        hybrid_use_relations_for_global = env_bool("RETRIEVAL_HYBRID_USE_RELATIONS_FOR_GLOBAL", False),
        global_max_windows = env_int("RETRIEVAL_GLOBAL_MAX_WINDOWS", 4),
        global_window_tokens = env_int("RETRIEVAL_GLOBAL_WINDOW_TOKENS", 512),
        # Local
        use_local_chunks = env_bool("RETRIEVAL_USE_LOCAL_CHUNKS", True),
        use_local_graph = env_bool("RETRIEVAL_USE_LOCAL_GRAPH", True),
        local_max_windows = env_int("RETRIEVAL_LOCAL_MAX_WINDOWS", 6),
        # PathRAG-specific
        path_use_top_entities = env_int("RETRIEVAL_PATH_USE_TOP_ENTITIES", 5),
        path_max_depth = env_int("RETRIEVAL_PATH_MAX_DEPTH", 3),
        path_threshold = env_float("RETRIEVAL_PATH_THRESHOLD", 0.3),
        path_alpha = env_float("RETRIEVAL_PATH_ALPHA", 0.8),
        path_max_windows = env_int("RETRIEVAL_PATH_MAX_WINDOWS", 5),
        path_window_tokens = env_int("RETRIEVAL_PATH_WINDOW_TOKENS", 512),
        # LightRAG-specific
        light_mode = env_str("RETRIEVAL_LIGHT_MODE", "mix"),
        response_type = env_str("RETRIEVAL_RESPONSE_TYPE", "Single Paragraphs"),
        enable_rerank = env_bool("RETRIEVAL_ENABLE_RERANK", True),
        rerank_top_k = env_int("RETRIEVAL_RERANK_TOP_K", 20),
        rerank_cache_dir = env_str("RETRIEVAL_RERANK_CACHE_DIR", "./flashrank_model"),
        rerank_model_name = env_str("RETRIEVAL_RERANK_MODEL_NAME", "ms-marco-MultiBERT-L-12"),
        truncate_chunks = env_bool("RETRIEVAL_TRUNCATE_CHUNKS", False),
    )

    return Settings(
        provider=provider,
        chat=chat,
        llmperf=llmperf,
        embeddings=embeddings,
        prompts=prompts,
        ingestion=ingestion,
        chunking=chunking,
        storage=storage,
        retrieval=retrieval,
    )


# Optional: convenience singleton (you can prefer dependency injection instead)
settings = load_settings()

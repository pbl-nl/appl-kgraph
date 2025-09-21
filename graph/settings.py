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
    if val is None:
        return None
    v = val.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v

def env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    return _strip_quotes(val) if val is not None else default

def env_int(key: str, default: int) -> int:
    val = env_str(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default

def env_float(key: str, default: float) -> float:
    val = env_str(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default

def env_bool(key: str, default: bool) -> bool:
    val = env_str(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "t"}

def env_list(key: str, default_csv: str, sep: str = ",") -> List[str]:
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

# ─────────────────────────────────────────────────────────────
# Loader / validator
# ─────────────────────────────────────────────────────────────

def load_settings() -> Settings:
    """
    Load .env once, parse & validate config, return a Settings object.
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

    return Settings(
        provider=provider,
        chat=chat,
        llmperf=llmperf,
        embeddings=embeddings,
        prompts=prompts,
        ingestion=ingestion,
        chunking=chunking,
        storage=storage,
    )


# Optional: convenience singleton (you can prefer dependency injection instead)
settings = load_settings()

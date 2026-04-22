from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from dotenv import load_dotenv

import utils as ut


VALID_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md", ".html"]


@dataclass(frozen=True)
class ProviderSettings:
    provider: Literal["openai", "azure"] = "openai"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_llm_model: Optional[str] = None
    openai_embeddings_model: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2024-02-15-preview"
    azure_llm_deployment: Optional[str] = None
    azure_embeddings_deployment: Optional[str] = None


@dataclass(frozen=True)
class ChatGenerationSettings:
    temperature: float = 0.0
    completion_max_tokens: int = 2048

    @property
    def max_tokens(self) -> int:
        return self.completion_max_tokens


@dataclass(frozen=True)
class LLMPerformanceSettings:
    max_concurrency: int = 6
    cache_enabled: bool = True
    cache_max_age_hours: int = 720


@dataclass(frozen=True)
class EmbeddingSettings:
    batch_size: int = 64


@dataclass(frozen=True)
class PromptFormattingSettings:
    default_language: str = "English"
    tuple_delimiter: str = "<|>"
    record_delimiter: str = "##"
    completion_delimiter: str = "<|COMPLETE|>"
    default_entity_types: List[str] = None  # type: ignore[assignment]
    default_user_prompt: str = "n/a"


@dataclass(frozen=True)
class IngestionMergeSettings:
    delimiter: str = "||"
    description_segment_limit: int = 5


@dataclass(frozen=True)
class ChunkingSettings:
    max_chars: int = 1200
    overlap_chars: int = 200
    include_overlap_in_limit: bool = True
    join_with: str = " "


@dataclass(frozen=True)
class StoragePaths:
    documents_db: str = "./storage/documents.sqlite"
    chunks_db: str = "./storage/chunks.sqlite"
    graph_db: str = "./storage/graph.sqlite"
    chroma_chunks: str = "./storage/chroma_chunks"
    chroma_entities: str = "./storage/chroma_entities"
    chroma_relations: str = "./storage/chroma_relations"


@dataclass(frozen=True)
class ProjectSettings:
    artifacts_dirname: str = ".appl-kgraph"
    storage_dirname: str = "storage"
    logs_dirname: str = "logs"
    qa_logs_dirname: str = "qa"
    audits_dirname: str = "audits"
    extraction_audits_dirname: str = "extraction"


@dataclass(frozen=True)
class ExtractionSettings:
    use_chunk_language: bool = True
    detect_chunk_language: bool = False
    audit_second_pass_enabled: bool = False


@dataclass(frozen=True)
class LoggingSettings:
    ingestion_enabled: bool = True
    ingestion_level: str = "INFO"
    retrieval_enabled: bool = True
    retrieval_level: str = "INFO"
    qa_enabled: bool = True


@dataclass(frozen=True)
class RetrievalSettings:
    entity_top_k: int = 5
    relation_top_k: int = 5
    chunk_top_k: int = 6
    graph_depth: int = 2
    graph_windows: int = 3
    chunk_windows: int = 3
    graph_window_tokens: int = 512
    chunk_window_tokens: int = 512
    tiktoken_model: str = "gpt-4o-mini"
    answer_max_tokens: int = 512
    llm_temperature: float = 0.0
    history_turns: int = 4
    hybrid_use_paths_for_global: bool = True
    hybrid_use_relations_for_global: bool = False
    global_max_windows: int = 4
    global_window_tokens: int = 512
    use_local_chunks: bool = True
    use_local_graph: bool = True
    local_max_windows: int = 6
    path_use_top_entities: int = 5
    path_max_depth: int = 3
    path_threshold: float = 0.3
    path_alpha: float = 0.8
    path_max_windows: int = 5
    path_window_tokens: int = 512
    light_mode: str = "mix"
    response_type: str = "Single Paragraph"
    rerank_top_k: int = 20
    enable_rerank: bool = True
    rerank_cache_dir: str = "./flashrank_model"
    rerank_model_name: str = "ms-marco-MultiBERT-L-12"
    truncate_chunks: bool = False

    @property
    def llm_max_tokens(self) -> int:
        return self.answer_max_tokens


@dataclass(frozen=True)
class Settings:
    provider: ProviderSettings
    chat: ChatGenerationSettings
    llmperf: LLMPerformanceSettings
    embeddings: EmbeddingSettings
    prompts: PromptFormattingSettings
    ingestion: IngestionMergeSettings
    chunking: ChunkingSettings
    storage: StoragePaths
    project: ProjectSettings
    extraction: ExtractionSettings
    logging: LoggingSettings
    retrieval: RetrievalSettings


def load_settings() -> Settings:
    load_dotenv()

    provider_name = (ut.env_str("LLM_PROVIDER", "openai") or "openai").strip().lower()
    if provider_name not in {"openai", "azure"}:
        raise RuntimeError("LLM_PROVIDER must be 'openai' or 'azure'.")

    provider = ProviderSettings(
        provider=provider_name,  # type: ignore[arg-type]
        openai_api_key=ut.env_str("OPENAI_API_KEY"),
        openai_base_url=ut.env_str("OPENAI_BASE_URL"),
        openai_llm_model=ut.env_str("OPENAI_LLM_MODEL"),
        openai_embeddings_model=ut.env_str("OPENAI_EMBEDDINGS_MODEL"),
        azure_api_key=ut.env_str("AZURE_OPENAI_API_KEY"),
        azure_endpoint=ut.env_str("AZURE_OPENAI_ENDPOINT"),
        azure_api_version=ut.env_str("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        or "2024-02-15-preview",
        azure_llm_deployment=ut.env_str("AZURE_OPENAI_LLM_DEPLOYMENT_NAME"),
        azure_embeddings_deployment=ut.env_str("AZURE_OPENAI_EMB_DEPLOYMENT_NAME"),
    )

    if provider.provider == "openai":
        if not provider.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        if not provider.openai_llm_model:
            raise RuntimeError("OPENAI_LLM_MODEL is required when LLM_PROVIDER=openai.")
        if not provider.openai_embeddings_model:
            raise RuntimeError("OPENAI_EMBEDDINGS_MODEL is required when LLM_PROVIDER=openai.")
    else:
        missing = []
        if not provider.azure_api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not provider.azure_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not provider.azure_llm_deployment:
            missing.append("AZURE_OPENAI_LLM_DEPLOYMENT_NAME")
        if not provider.azure_embeddings_deployment:
            missing.append("AZURE_OPENAI_EMB_DEPLOYMENT_NAME")
        if missing:
            raise RuntimeError(
                f"When LLM_PROVIDER=azure, set required variables: {', '.join(missing)}"
            )

    chat = ChatGenerationSettings(
        temperature=ut.env_float("CHAT_TEMPERATURE", 0.0),
        completion_max_tokens=ut.env_int("CHAT_MAX_TOKENS", 2048),
    )

    llmperf = LLMPerformanceSettings(
        max_concurrency=ut.env_int("LLM_MAX_CONCURRENCY", 6),
        cache_enabled=ut.env_bool("LLM_CACHE_ENABLED", True),
        cache_max_age_hours=ut.env_int("LLM_CACHE_MAX_AGE_HOURS", 720),
    )

    embeddings = EmbeddingSettings(
        batch_size=ut.env_int("EMBEDDING_BATCH_SIZE", 64),
    )

    prompts = PromptFormattingSettings(
        default_language=ut.env_str("PROMPT_DEFAULT_LANGUAGE", "English") or "English",
        tuple_delimiter=ut.env_str("PROMPT_TUPLE_DELIMITER", "<|>") or "<|>",
        record_delimiter=ut.env_str("PROMPT_RECORD_DELIMITER", "##") or "##",
        completion_delimiter=ut.env_str("PROMPT_COMPLETION_DELIMITER", "<|COMPLETE|>")
        or "<|COMPLETE|>",
        default_entity_types=ut.env_list(
            "PROMPT_DEFAULT_ENTITY_TYPES",
            "organization,person,geo,event,category",
        ),
        default_user_prompt=ut.env_str("PROMPT_DEFAULT_USER_PROMPT", "n/a") or "n/a",
    )

    ingestion = IngestionMergeSettings(
        delimiter=ut.env_str("MERGE_DELIMITER", "||") or "||",
        description_segment_limit=ut.env_int("DESCRIPTION_SEGMENT_LIMIT", 5),
    )

    chunking = ChunkingSettings(
        max_chars=ut.env_int("CHUNK_MAX_CHARS", 1200),
        overlap_chars=ut.env_int("CHUNK_OVERLAP_CHARS", 200),
        include_overlap_in_limit=ut.env_bool("CHUNK_INCLUDE_OVERLAP_IN_LIMIT", True),
        join_with=ut.env_str("CHUNK_JOIN_WITH", " ") or " ",
    )

    storage = StoragePaths(
        documents_db=ut.env_str("DOCUMENTS_DB_PATH", "./storage/documents.sqlite")
        or "./storage/documents.sqlite",
        chunks_db=ut.env_str("CHUNKS_DB_PATH", "./storage/chunks.sqlite")
        or "./storage/chunks.sqlite",
        graph_db=ut.env_str("GRAPH_DB_PATH", "./storage/graph.sqlite")
        or "./storage/graph.sqlite",
        chroma_chunks=ut.env_str("CHROMA_CHUNKS_PATH", "./storage/chroma_chunks")
        or "./storage/chroma_chunks",
        chroma_entities=ut.env_str("CHROMA_ENTITIES_PATH", "./storage/chroma_entities")
        or "./storage/chroma_entities",
        chroma_relations=ut.env_str("CHROMA_RELATIONS_PATH", "./storage/chroma_relations")
        or "./storage/chroma_relations",
    )

    project = ProjectSettings(
        artifacts_dirname=ut.env_str("PROJECT_ARTIFACTS_DIRNAME", ".appl-kgraph")
        or ".appl-kgraph",
        storage_dirname=ut.env_str("PROJECT_STORAGE_DIRNAME", "storage") or "storage",
        logs_dirname=ut.env_str("PROJECT_LOGS_DIRNAME", "logs") or "logs",
        qa_logs_dirname=ut.env_str("PROJECT_QA_LOGS_DIRNAME", "qa") or "qa",
        audits_dirname=ut.env_str("PROJECT_AUDITS_DIRNAME", "audits") or "audits",
        extraction_audits_dirname=ut.env_str(
            "PROJECT_EXTRACTION_AUDITS_DIRNAME", "extraction"
        )
        or "extraction",
    )

    extraction = ExtractionSettings(
        use_chunk_language=ut.env_bool("EXTRACTION_USE_CHUNK_LANGUAGE", True),
        detect_chunk_language=ut.env_bool("EXTRACTION_DETECT_CHUNK_LANGUAGE", False),
        audit_second_pass_enabled=ut.env_bool(
            "EXTRACTION_AUDIT_SECOND_PASS_ENABLED",
            False,
        ),
    )

    logging_settings = LoggingSettings(
        ingestion_enabled=ut.env_bool("INGESTION_LOG_ENABLED", True),
        ingestion_level=ut.env_str("INGESTION_LOG_LEVEL", "INFO") or "INFO",
        retrieval_enabled=ut.env_bool("RETRIEVAL_LOG_ENABLED", True),
        retrieval_level=ut.env_str("RETRIEVAL_LOG_LEVEL", "INFO") or "INFO",
        qa_enabled=ut.env_bool("QA_LOG_ENABLED", True),
    )

    retrieval = RetrievalSettings(
        entity_top_k=ut.env_int("RETRIEVAL_ENTITY_TOP_K", 5),
        relation_top_k=ut.env_int("RETRIEVAL_RELATION_TOP_K", 5),
        chunk_top_k=ut.env_int("RETRIEVAL_CHUNK_TOP_K", 6),
        graph_depth=ut.env_int("RETRIEVAL_GRAPH_DEPTH", 2),
        graph_windows=ut.env_int("RETRIEVAL_GRAPH_WINDOWS", 3),
        chunk_windows=ut.env_int("RETRIEVAL_CHUNK_WINDOWS", 3),
        graph_window_tokens=ut.env_int("RETRIEVAL_GRAPH_WINDOW_TOKENS", 512),
        chunk_window_tokens=ut.env_int("RETRIEVAL_CHUNK_WINDOW_TOKENS", 512),
        tiktoken_model=ut.env_str("RETRIEVAL_TIKTOKEN_MODEL", "gpt-4o-mini")
        or "gpt-4o-mini",
        answer_max_tokens=ut.env_int("RETRIEVAL_LLM_MAX_TOKENS", 512),
        llm_temperature=ut.env_float("RETRIEVAL_LLM_TEMPERATURE", 0.0),
        history_turns=ut.env_int("RETRIEVAL_HISTORY_TURNS", 4),
        hybrid_use_paths_for_global=ut.env_bool(
            "RETRIEVAL_HYBRID_USE_PATHS_FOR_GLOBAL",
            True,
        ),
        hybrid_use_relations_for_global=ut.env_bool(
            "RETRIEVAL_HYBRID_USE_RELATIONS_FOR_GLOBAL",
            False,
        ),
        global_max_windows=ut.env_int("RETRIEVAL_GLOBAL_MAX_WINDOWS", 4),
        global_window_tokens=ut.env_int("RETRIEVAL_GLOBAL_WINDOW_TOKENS", 512),
        use_local_chunks=ut.env_bool("RETRIEVAL_USE_LOCAL_CHUNKS", True),
        use_local_graph=ut.env_bool("RETRIEVAL_USE_LOCAL_GRAPH", True),
        local_max_windows=ut.env_int("RETRIEVAL_LOCAL_MAX_WINDOWS", 6),
        path_use_top_entities=ut.env_int("RETRIEVAL_PATH_USE_TOP_ENTITIES", 5),
        path_max_depth=ut.env_int("RETRIEVAL_PATH_MAX_DEPTH", 3),
        path_threshold=ut.env_float("RETRIEVAL_PATH_THRESHOLD", 0.3),
        path_alpha=ut.env_float("RETRIEVAL_PATH_ALPHA", 0.8),
        path_max_windows=ut.env_int("RETRIEVAL_PATH_MAX_WINDOWS", 5),
        path_window_tokens=ut.env_int("RETRIEVAL_PATH_WINDOW_TOKENS", 512),
        light_mode=ut.env_str("RETRIEVAL_LIGHT_MODE", "mix") or "mix",
        response_type=ut.env_str("RETRIEVAL_RESPONSE_TYPE", "Single Paragraph")
        or "Single Paragraph",
        enable_rerank=ut.env_bool("RETRIEVAL_ENABLE_RERANK", True),
        rerank_top_k=ut.env_int("RETRIEVAL_RERANK_TOP_K", 20),
        rerank_cache_dir=ut.env_str(
            "RETRIEVAL_RERANK_CACHE_DIR",
            "./flashrank_model",
        )
        or "./flashrank_model",
        rerank_model_name=ut.env_str(
            "RETRIEVAL_RERANK_MODEL_NAME",
            "ms-marco-MultiBERT-L-12",
        )
        or "ms-marco-MultiBERT-L-12",
        truncate_chunks=ut.env_bool("RETRIEVAL_TRUNCATE_CHUNKS", False),
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
        project=project,
        extraction=extraction,
        logging=logging_settings,
        retrieval=retrieval,
    )


settings = load_settings()

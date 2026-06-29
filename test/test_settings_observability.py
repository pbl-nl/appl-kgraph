import os
import sys
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_EMBEDDINGS_MODEL", "test-embed")

sys.path.append(str(Path(__file__).resolve().parent.parent / "graph"))

import settings as settings_module


OBSERVABILITY_KEYS = [
    "AUDIT_ENABLED",
    "VERBOSITY_ENABLED",
    "INTERNAL_LOGGING_ENABLED",
    "INTERNAL_LOG_LEVEL",
    "PROJECT_AUDIT_LOGS_DIRNAME",
    "PROJECT_DIAGNOSTICS_DIRNAME",
    "PROJECT_EXTRACTION_DIAGNOSTICS_DIRNAME",
    "EXTRACTION_VALIDATION_SECOND_PASS_ENABLED",
    "RETRIEVAL_TOP_K_CHUNK_PER_ENTITY",
    "QA_LOG_ENABLED",
    "INGESTION_LOG_ENABLED",
    "INGESTION_LOG_LEVEL",
    "RETRIEVAL_LOG_ENABLED",
    "RETRIEVAL_LOG_LEVEL",
    "PROJECT_QA_LOGS_DIRNAME",
    "PROJECT_AUDITS_DIRNAME",
    "PROJECT_EXTRACTION_AUDITS_DIRNAME",
    "EXTRACTION_AUDIT_SECOND_PASS_ENABLED",
]


def _load_with_env(monkeypatch, **env):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_LLM_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_EMBEDDINGS_MODEL", "test-embed")
    for key in OBSERVABILITY_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return settings_module.load_settings()


def test_observability_settings_use_new_environment_variables(monkeypatch):
    settings = _load_with_env(
        monkeypatch,
        AUDIT_ENABLED="false",
        VERBOSITY_ENABLED="0",
        INTERNAL_LOGGING_ENABLED="no",
        INTERNAL_LOG_LEVEL="DEBUG",
        PROJECT_AUDIT_LOGS_DIRNAME="chat-audit",
        PROJECT_DIAGNOSTICS_DIRNAME="dev-diagnostics",
        PROJECT_EXTRACTION_DIAGNOSTICS_DIRNAME="validation",
        EXTRACTION_VALIDATION_SECOND_PASS_ENABLED="true",
        RETRIEVAL_TOP_K_CHUNK_PER_ENTITY="9",
    )

    assert settings.logging.audit_enabled is False
    assert settings.logging.verbosity_enabled is False
    assert settings.logging.internal_logging_enabled is False
    assert settings.logging.internal_log_level == "DEBUG"
    assert settings.project.audit_logs_dirname == "chat-audit"
    assert settings.project.diagnostics_dirname == "dev-diagnostics"
    assert settings.project.extraction_diagnostics_dirname == "validation"
    assert settings.extraction.validation_second_pass_enabled is True
    assert settings.retrieval.top_k_chunk_per_entity == 9


def test_legacy_observability_environment_variables_are_ignored(monkeypatch):
    settings = _load_with_env(
        monkeypatch,
        QA_LOG_ENABLED="false",
        INGESTION_LOG_ENABLED="0",
        RETRIEVAL_LOG_ENABLED="0",
        INGESTION_LOG_LEVEL="WARNING",
        PROJECT_QA_LOGS_DIRNAME="qa-legacy",
        PROJECT_AUDITS_DIRNAME="diagnostics-legacy",
        PROJECT_EXTRACTION_AUDITS_DIRNAME="extraction-legacy",
        EXTRACTION_AUDIT_SECOND_PASS_ENABLED="true",
    )

    assert settings.logging.audit_enabled is True
    assert settings.logging.internal_logging_enabled is True
    assert settings.logging.internal_log_level == "INFO"
    assert settings.project.audit_logs_dirname == "audit"
    assert settings.project.diagnostics_dirname == "diagnostics"
    assert settings.project.extraction_diagnostics_dirname == "extraction"
    assert settings.extraction.validation_second_pass_enabled is False
    assert not hasattr(settings.logging, "qa_enabled")
    assert not hasattr(settings.logging, "ingestion_enabled")
    assert not hasattr(settings.logging, "retrieval_enabled")
    assert not hasattr(settings.extraction, "audit_second_pass_enabled")

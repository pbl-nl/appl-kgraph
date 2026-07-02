from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from project_paths import ProjectPaths, ensure_project_dirs
from settings import settings


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def build_audit_settings_snapshot() -> Dict[str, Any]:
    provider_model = (
        settings.provider.azure_llm_deployment
        if settings.provider.provider == "azure"
        else settings.provider.openai_llm_model
    )
    embeddings_model = (
        settings.provider.azure_embeddings_deployment
        if settings.provider.provider == "azure"
        else settings.provider.openai_embeddings_model
    )
    return _json_safe(
        {
            "provider": {
                "provider": settings.provider.provider,
                "llm_model": provider_model,
                "embeddings_model": embeddings_model,
                "azure_api_version": settings.provider.azure_api_version
                if settings.provider.provider == "azure"
                else None,
            },
            "chat": asdict(settings.chat),
            "llmperf": asdict(settings.llmperf),
            "embeddings": asdict(settings.embeddings),
            "prompts": {
                "default_language": settings.prompts.default_language,
                "default_entity_types": settings.prompts.default_entity_types,
                "default_user_prompt": settings.prompts.default_user_prompt,
            },
            "ingestion": asdict(settings.ingestion),
            "chunking": asdict(settings.chunking),
            "project": asdict(settings.project),
            "extraction": asdict(settings.extraction),
            "logging": asdict(settings.logging),
            "retrieval": asdict(settings.retrieval),
        }
    )


def write_audit_log(
    *,
    project_paths: Optional[ProjectPaths],
    retriever_name: str,
    payload: Dict[str, Any],
) -> Optional[Path]:
    if project_paths is None or not settings.logging.audit_enabled:
        return None

    ensure_project_dirs(project_paths)
    timestamp = datetime.now(timezone.utc)
    filename = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{retriever_name}_{uuid4().hex[:8]}.json"
    target = project_paths.audit_logs_dir / filename

    body = {
        "retriever": retriever_name,
        "timestamp": timestamp.isoformat(),
        "settings": build_audit_settings_snapshot(),
        **_json_safe(payload),
    }
    target.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return target

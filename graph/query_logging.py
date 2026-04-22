from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from project_paths import ProjectPaths, ensure_project_dirs


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def write_query_log(
    *,
    project_paths: Optional[ProjectPaths],
    retriever_name: str,
    payload: Dict[str, Any],
) -> Optional[Path]:
    if project_paths is None:
        return None

    ensure_project_dirs(project_paths)
    timestamp = datetime.now(timezone.utc)
    filename = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{retriever_name}_{uuid4().hex[:8]}.json"
    target = project_paths.qa_logs_dir / filename

    body = {
        "retriever": retriever_name,
        "timestamp": timestamp.isoformat(),
        **_json_safe(payload),
    }
    target.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return target

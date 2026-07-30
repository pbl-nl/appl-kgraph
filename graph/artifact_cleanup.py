from __future__ import annotations

import shutil
import time
from typing import List, Optional

try:
    from graph.project_paths import ProjectPaths
except ModuleNotFoundError:
    from project_paths import ProjectPaths


def remove_tree_with_retries(path, attempts: int = 5, base_delay_seconds: float = 0.2) -> None:
    last_error: Optional[Exception] = None
    for index in range(attempts):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except OSError as exc:
            last_error = exc
            if index == attempts - 1:
                break
            time.sleep(base_delay_seconds * (index + 1))

    if last_error is not None:
        raise last_error


def cleanup_graph_artifacts_best_effort(project_paths: ProjectPaths) -> Optional[str]:
    errors: List[str] = []

    try:
        remove_tree_with_retries(project_paths.knowledge_graph_dir)
    except OSError as exc:
        errors.append(f"knowledge_graph: {exc}")

    try:
        remove_tree_with_retries(project_paths.storage_root)
        project_paths.storage_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.append(f"storage: {exc}")

    if errors:
        return " | ".join(errors)
    return None

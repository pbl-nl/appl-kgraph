from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from schemas import DocumentRef
from settings import VALID_EXTENSIONS, settings


def discover_documents(
    project_root: Path,
    *,
    valid_extensions: Optional[Iterable[str]] = None,
) -> List[DocumentRef]:
    """Discover supported source documents below a document collection root."""

    root = Path(project_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    allowed = {extension.lower() for extension in (valid_extensions or VALID_EXTENSIONS)}
    artifacts_root = root / settings.project.artifacts_dirname
    paths = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if artifacts_root in path.parents:
            continue
        if path.suffix.lower() not in allowed:
            continue
        paths.append(path)

    paths.sort(key=lambda item: str(item).lower())
    return [DocumentRef(path=path, root=root) for path in paths]

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from schemas import DocumentRef
from settings import VALID_EXTENSIONS, settings


def is_temporary_document(path: Path) -> bool:
    """Return whether a path matches the ingestion temporary-file policy."""

    name = path.name.lower()
    return (
        name.startswith("~$") and name.endswith((".docx", ".doc"))
    ) or (name.endswith((".tmp", ".temp")) and "word" in name)


def discover_documents(
    project_root: Path,
    *,
    valid_extensions: Optional[Iterable[str]] = None,
) -> List[DocumentRef]:
    """Discover supported source documents below a document collection root."""

    root = Path(project_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    extensions = VALID_EXTENSIONS if valid_extensions is None else valid_extensions
    allowed = {extension.lower() for extension in extensions}
    artifacts_root = root / settings.project.artifacts_dirname
    paths = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if artifacts_root in path.parents:
            continue
        if is_temporary_document(path):
            continue
        if path.suffix.lower() not in allowed:
            continue
        paths.append(path)

    paths.sort(key=lambda item: str(item).lower())
    return [DocumentRef(path=path, root=root) for path in paths]

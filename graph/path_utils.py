"""Interpret persisted path metadata independently of the current host OS."""

from pathlib import PurePosixPath, PureWindowsPath


def stored_path_name(value: str) -> str:
    """Read a Windows or POSIX basename; do not translate filesystem locations.

    Absolute POSIX paths may contain literal backslashes in their filenames.
    Relative paths containing backslashes are treated as Windows paths.
    """
    if value.startswith("/"):
        return PurePosixPath(value).name
    return PureWindowsPath(value).name

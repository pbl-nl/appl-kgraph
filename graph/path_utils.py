"""Helpers for stored metadata paths, which may originate on another OS."""

from pathlib import PureWindowsPath


def source_filename(path: str) -> str:
    """Read a filename from either Windows or POSIX metadata without filesystem IO."""
    return PureWindowsPath(path).name

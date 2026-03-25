from __future__ import annotations

import os
from typing import List, Optional

from langdetect import LangDetectException, detect


LANGUAGE_NAME_MAP = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese Traditional",
}


def detect_language(text: str, num_chars: int = 1000) -> str:
    """
    Detect a language code from the leading portion of a text.
    """
    text_snippet = text[:num_chars] if len(text) > num_chars else text

    if not text_snippet.strip():
        return "unknown"

    try:
        return detect(text_snippet)
    except LangDetectException as exc:
        if "No features in text" in str(exc):
            return "unknown"
    return "unknown"


def normalize_language_name(language: Optional[str], default: str = "English") -> str:
    """
    Convert a language code or free-form language string into a prompt-friendly name.
    """
    if not language:
        return default

    candidate = str(language).strip()
    if not candidate:
        return default

    lowered = candidate.lower()
    if lowered == "unknown":
        return default
    if lowered in LANGUAGE_NAME_MAP:
        return LANGUAGE_NAME_MAP[lowered]
    if len(candidate) <= 3 and candidate.islower():
        return default
    return candidate[:1].upper() + candidate[1:]


def _strip_quotes(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    stripped = val.strip()
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return stripped[1:-1]
    return stripped


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
    return [item.strip() for item in raw.split(sep) if item.strip()]

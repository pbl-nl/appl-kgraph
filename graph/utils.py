import os
from typing import List, Optional, Literal
from langdetect import detect, LangDetectException

def detect_language(text: str, num_chars: int = 1000) -> str:
    """
    Detects the language of a text based on a sample of its characters.

    Args:
        text (str): The input text to analyze for language detection.
        num_chars (int, optional): The number of characters from the beginning
            of the text to use for detection. Defaults to 1000.

    Returns:
        str: A language code (e.g., 'en' for English, 'fr' for French) or 'unknown'
            if the language cannot be detected or if the text is empty.
    """
    text_snippet = text[:num_chars] if len(text) > num_chars else text

    if not text_snippet.strip():
        # Handle the case where the text snippet is empty or only contains whitespace
        return 'unknown'
    try:
        return detect(text_snippet)
    except LangDetectException as e:
        if 'No features in text' in str(e):
            # Handle the specific error where no features are found in the text
            return 'unknown'
    # Default return statement to ensure the function always returns a value
    return 'unknown'

# ─────────────────────────────────────────────────────────────
# Small helpers to parse environment variables robustly
# ─────────────────────────────────────────────────────────────

def _strip_quotes(val: Optional[str]) -> Optional[str]:
    """
    Removes surrounding quotes from environment variable values.

    Args:
        val (Optional[str]): The value to process.

    Returns:
        Optional[str]: The value with quotes stripped, or None if input was None.
    """
    if val is None:
        return None
    v = val.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v

def env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Reads a string environment variable with quote stripping.

    Args:
        key (str): The environment variable name.
        default (Optional[str], optional): Default value if not found. Defaults to None.

    Returns:
        Optional[str]: The environment variable value or default.
    """
    val = os.getenv(key)
    return _strip_quotes(val) if val is not None else default

def env_int(key: str, default: int) -> int:
    """
    Reads an integer environment variable with fallback to default.

    Args:
        key (str): The environment variable name.
        default (int): Default value if not found or invalid.

    Returns:
        int: The parsed integer value or default.
    """
    val = env_str(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default

def env_float(key: str, default: float) -> float:
    """
    Reads a float environment variable with fallback to default.

    Args:
        key (str): The environment variable name.
        default (float): Default value if not found or invalid.

    Returns:
        float: The parsed float value or default.
    """
    val = env_str(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default

def env_bool(key: str, default: bool) -> bool:
    """
    Reads a boolean environment variable with fallback to default.

    Args:
        key (str): The environment variable name.
        default (bool): Default value if not found.

    Returns:
        bool: True if value is in {"1", "true", "yes", "y", "t"}, otherwise default.
    """
    val = env_str(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "t"}

def env_list(key: str, default_csv: str, sep: str = ",") -> List[str]:
    """
    Reads a delimited list environment variable.

    Args:
        key (str): The environment variable name.
        default_csv (str): Default comma-separated value string.
        sep (str, optional): Delimiter character. Defaults to ",".

    Returns:
        List[str]: List of trimmed non-empty values.
    """
    raw = env_str(key, default_csv) or ""
    return [x.strip() for x in raw.split(sep) if x.strip()]

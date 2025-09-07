from __future__ import annotations
from typing import Callable
from langdetect import LangDetectException, detect


def detect_language(text: str, number_of_characters: int = 1000) -> str:
    """
    Detects language based on the first X number of characters
    """
    text_snippet = text[:number_of_characters] if len(text) > number_of_characters else text

    if not text_snippet.strip():
        # Handle the case where the text snippet is empty or only contains whitespace
        return 'unknown'
    try:
        return detect(text_snippet)
    except LangDetectException as e:
        if 'No features in text' in str(e):
            # Handle the specific error where no features are found in the text
            return 'unknown'
        
# --- Bounded merge with late summarization ---
def simple_summarize(text: str, max_chars: int = 600) -> str:
    """
    Heuristic fallback if no LLM is wired: keep unique, informative lines,
    then trim to max_chars. Replace with your LLM-backed summarizer if available.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen, uniq = set(), []
    for ln in lines:
        if ln not in seen:
            seen.add(ln); uniq.append(ln)
    joined = " ".join(uniq)
    return joined[:max_chars] + ("…" if len(joined) > max_chars else "")

def merge_description_bounded(
    old: str | None,
    new: str | None,
    *,
    concat_until: int = 800,     # concat freely until this length
    hard_cap: int = 1600,        # never exceed this length
    summarizer: Callable[[str, int], str] | None = None,
) -> str:
    """
    Concatenate old+new until `concat_until`. Beyond that, summarize the combined text.
    Always enforce `hard_cap` at the end. Idempotent and de-duplicates lines.
    """
    old, new = (old or "").strip(), (new or "").strip()
    if not old: return (new[:hard_cap] + ("…" if len(new) > hard_cap else ""))
    if not new: return (old[:hard_cap] + ("…" if len(old) > hard_cap else ""))

    # de-duplicate by lines while preserving order
    def _dedup_lines(text: str) -> list[str]:
        parts = [p.strip() for p in text.splitlines() if p.strip()]
        seen, out = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p); out.append(p)
        return out

    merged = "\n".join(_dedup_lines(old) + _dedup_lines(new))
    if len(merged) <= concat_until:
        return merged[:hard_cap] + ("…" if len(merged) > hard_cap else "")

    # over threshold: summarize
    if summarizer is None:
        summarizer = simple_summarize
    summarized = summarizer(merged, max_chars=hard_cap)
    return summarized
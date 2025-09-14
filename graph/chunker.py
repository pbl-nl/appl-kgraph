"""
Page-aware, sentence-preserving chunker.

Key features:
- Accepts PDF pages as a list of tuples: [(page_number, text), ...].
- Builds chunks up to `max_chars` without breaking sentences.
- If a chunk reaches the end of a page but hasn't filled to `max_chars`,
  it continues to the next page to try to fill capacity (still respecting sentence boundaries).
- Supports sentence-level overlap between adjacent chunks. Overlap is composed only of complete
  sentences (never partial). If a new chunk starts at the beginning of page 2, the overlap can
  come from the tail of page 1 (again, full sentences only).
- If a single sentence is longer than `max_chars`, it is placed alone in a chunk (not split).

Usage:
    from page_aware_chunker import chunk_parsed_pages

    chunks = chunk_parsed_pages(
        pages=[(0, "Page 1 text..."), (1, "Page 2 text...")],
        max_chars=1200,
        overlap_chars=200,
        include_overlap_in_limit=True,
    )

Each item in `chunks` is a dict:
    {
        "chunk_id": int,
        "text": str,
        "char_count": int,
        "start_page": int,
        "end_page": int,
        "sentence_span": [  # (global_sentence_index, page_number, sentence_text)
            (12, 0, "First sentence."),
            ...
        ],
        "overlap_from_previous": bool
    }
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Iterable
from settings import settings


# Avoid variable-width lookbehind (unsupported by Python's 're').
# We capture the end-of-sentence boundary instead.
_SENT_END_SPLIT = re.compile(
    r'([.!?]["\')\]]*)(\s+)(?=[A-Za-z0-9"\(])'
)


def _normalize_ws(text: str) -> str:
    # Collapse whitespace/newlines to single spaces and strip ends
    return re.sub(r'\s+', ' ', (text or '').strip())


def _split_sentences(text: str) -> List[str]:
    text = _normalize_ws(text)
    if not text:
        return []
    sentences: List[str] = []
    start = 0
    for m in _SENT_END_SPLIT.finditer(text):
        end = m.end(1)  # end at punctuation/quotes
        piece = text[start:end].strip()
        if piece:
            sentences.append(piece)
        start = m.end(0)  # skip the whitespace following EoS
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


@dataclass
class Sentence:
    page: int
    text: str
    idx: int  # global index among all sentences


def _flatten_pages_to_sentences(pages: List[Tuple[int, str]]) -> List[Sentence]:
    # Ensure pages are processed in ascending page order
    pages_sorted = sorted(pages, key=lambda x: x[0])
    sentences: List[Sentence] = []
    idx = 0
    for page_no, page_text in pages_sorted:
        for s in _split_sentences(page_text):
            sentences.append(Sentence(page=page_no, text=s, idx=idx))
            idx += 1
    return sentences


def _sentences_len(sentences: Iterable[Sentence], join_with: str = " ") -> int:
    # Return the total chars if we join the sentences with single spaces
    total = 0
    first = True
    for s in sentences:
        if not first:
            total += len(join_with)
        total += len(s.text)
        first = False
    return total


def chunk_parsed_pages(
    pages: List[Tuple[int, str]],
    max_chars: int = settings.chunking.max_chars,
    overlap_chars: int = settings.chunking.overlap_chars,
    include_overlap_in_limit: bool = settings.chunking.include_overlap_in_limit,
    join_with: str = settings.chunking.join_with,
) -> List[Dict[str, Any]]:
    """
    Chunk a parsed file represented as list of (page_number, text) tuples.

    Rules:
    - Do not break sentences.
    - When a page ends, continue on the next page to fill the chunk up to max_chars.
    - Include overlap between chunks using ONLY whole sentences, up to overlap_chars.
      Overlap is drawn from the tail of the previous chunk; may come from previous pages.
    - If a single sentence is longer than max_chars, place it as its own chunk.

    Parameters
    ----------
    pages : list[(int, str)]
        For PDFs: [(page_no, text), ...]. For non-PDFs, just use a single tuple like [(0, full_text)].
    max_chars : int
        Target maximum characters per chunk (including joiners/spaces).
    overlap_chars : int
        Desired overlap size in characters (composed of *whole sentences only*).
    include_overlap_in_limit : bool
        If True, the overlap counts toward `max_chars`. If False, a chunk can exceed `max_chars`
        by up to `overlap_chars` due to the prefix overlap.
    join_with : str
        String used to join sentences inside a chunk.

    Returns
    -------
    list of dict
        See module docstring for the schema.
    """
    sentences = _flatten_pages_to_sentences(pages)
    chunks: List[Dict[str, Any]] = []
    if not sentences:
        return chunks

    i = 0  # index of the next *new* sentence to place
    prev_chunk_sentence_indices: List[int] = []

    while i < len(sentences):
        # Determine overlap sentences (from tail of previous chunk) within overlap_chars
        overlap: List[Sentence] = []
        if prev_chunk_sentence_indices and overlap_chars > 0:
            # Walk backwards over previous chunk's sentence indices, collect until we hit the char budget
            tail = [sentences[k] for k in prev_chunk_sentence_indices]
            total = 0
            tmp: List[Sentence] = []
            for s in reversed(tail):
                add_len = len(s.text) if not tmp else (len(join_with) + len(s.text))
                if total + add_len <= overlap_chars:
                    tmp.append(s)
                    total += add_len
                else:
                    break
            overlap = list(reversed(tmp))  # restore chronological order

        # Compute initial char count for capacity accounting
        overlap_len = _sentences_len(overlap, join_with) if overlap else 0
        char_count = overlap_len if include_overlap_in_limit else 0

        # Add new sentences until we hit max_chars (respect whole sentences)
        new_sents: List[Sentence] = []
        while i < len(sentences):
            s = sentences[i]
            add_len = (len(s.text) if (not overlap and not new_sents) else len(join_with) + len(s.text))

            # If adding this sentence would exceed max_chars and we already have some new_sents,
            # then stop to keep within the limit.
            if char_count + add_len > max_chars and len(new_sents) > 0:
                break

            # If we'd exceed but we have *no* new_sents yet, we still take this sentence
            # to avoid splitting it (even if it goes over max_chars).
            if char_count + add_len > max_chars and len(new_sents) == 0:
                new_sents.append(s)
                i += 1
                char_count += add_len
                break

            # Otherwise we can add it safely
            new_sents.append(s)
            i += 1
            char_count += add_len

        # Build this chunk
        all_sents = overlap + new_sents
        if not all_sents:
            # Safety: if nothing got added (shouldn't happen), advance by one sentence
            all_sents = [sentences[i]]
            i += 1

        text = join_with.join(s.text for s in all_sents)
        start_page = min(s.page for s in all_sents)
        end_page = max(s.page for s in all_sents)
        chunk_sentence_indices = [s.idx for s in all_sents]

        chunk = {
            "chunk_id": len(chunks),
            "text": text,
            "char_count": len(text),
            "start_page": start_page,
            "end_page": end_page,
            # No need for below in storage
            "sentence_span": [(s.idx, s.page, s.text) for s in all_sents],
            "overlap_from_previous": bool(overlap),
            "overlap_chars_effective": overlap_len,
            "included_new_sentence_count": len(new_sents),
            "include_overlap_in_limit": include_overlap_in_limit,
            "max_chars_target": max_chars,
        }
        chunks.append(chunk)

        # Prepare for next iteration
        prev_chunk_sentence_indices = chunk_sentence_indices

    return chunks

# !! Currently not used, but could be used for unit testing.
def chunk_text(
    text: str,
    max_chars: int = settings.chunking.max_chars,
    overlap_chars: int = settings.chunking.overlap_chars,
    include_overlap_in_limit: bool = settings.chunking.include_overlap_in_limit,
    join_with: str = settings.chunking.join_with
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper to chunk a plain text string (non-PDF) using the same logic.
    Internally treats the text as a single-page document with page number 0.
    """
    return chunk_parsed_pages(
        pages=[(0, text)],
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        include_overlap_in_limit=include_overlap_in_limit,
        join_with=join_with,
    )


if __name__ == "__main__":
    # Minimal self-test / demonstration.
    sample_pages = [
        (0, "This is page one. It has two sentences. Now a third sentence ends here! "
            "And a very long sentence follows that might exceed the maximum chunk size if set very small "
            "but we will still keep it intact because we never split sentences."),
        (1, "Page two begins. We continue adding sentences so the chunk can fill to capacity across pages. "
            "The end is near? Yes. End."),
    ]

    # Example run
    chunks = chunk_parsed_pages(sample_pages, max_chars=120, overlap_chars=25, include_overlap_in_limit=True)
    print(f"Produced {len(chunks)} chunks:\n")
    for ch in chunks:
        print(f"[{ch['chunk_id']}] p{ch['start_page']}–p{ch['end_page']} ({ch['char_count']} chars)"
              f" | overlap={ch['overlap_chars_effective']}")
        print(ch["text"])
        print("---")
from __future__ import annotations

import re
import json
from dataclasses import dataclass
from prompts import PROMPTS
from llm import Chat
from typing import Any, Dict, Iterable, List, Optional, Tuple
from settings import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from storage import Storage
import hashlib

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _sha256(s: str) -> str:
    """Stable SHA-256 for cache keys (text & prompt)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ─────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────

def build_entity_relation_prompt(
    text: str,
    language: Optional[str] = None,
    entity_types: Optional[Iterable[str]] = None,
) -> str:
    """
    Fill PROMPTS['entity_extraction'] for a single chunk of text. Also formats the
    examples so they DO NOT contain '{record_delimiter}' literals.
    """
    # 1) Base context for the prompt (without examples yet)
    examples = "\n\n".join(PROMPTS.get("entity_extraction_examples", []))
    ctx = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types) if entity_types else ", ".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        examples="",
        language=language or PROMPTS["DEFAULT_LANGUAGE"],
        input_text=text,
    )

    # 2) Join & format the examples with the SAME ctx so placeholders are replaced
    examples_template = "\n\n".join(PROMPTS.get("entity_extraction_examples", []))
    examples = examples_template.format(**ctx)  # fill in delimiters, entity_types, language
    ctx["examples"] = examples

    # 3) Finally format the main template
    template = PROMPTS["entity_extraction"]
    return template.format(**ctx)

# ─────────────────────────────────────────────────────────────
# Parsing utilities (regex + delimiter tolerant)
# ─────────────────────────────────────────────────────────────

@dataclass
class ParsedOutput:
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    content_keywords: List[str]
    raw_records: List[str]


_FANCY_QUOTES = {
    "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'",
}

def _normalize_quotes(s: str) -> str:
    for k, v in _FANCY_QUOTES.items():
        s = s.replace(k, v)
    return s


def _strip_parens(s: str) -> str:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        return s[1:-1].strip()
    return s


def _strip_quotes(s: str) -> str:
    s = s.strip()
    s = _normalize_quotes(s)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _to_float_or_none(x: str) -> Optional[float]:
    x = x.strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_model_output(
    raw: str,
    tuple_delim: Optional[str] = None,
    record_delim: Optional[str] = None,
    completion_delim: Optional[str] = None,
) -> ParsedOutput:
    """
    Parse the LLM output into entities and relationships using regex/splits.

    Expected generated record shapes (from your prompt):
      ("entity"<|>"name"<|>"type"<|>"description")
      ("relationship"<|>"source"<|>"target"<|>"description"<|>"keywords"<|>"strength")
      ("content_keywords"<|>"kw1, kw2, ...")

    Tolerates both real delimiters and accidental literal
    '{record_delimiter}' / '{tuple_delimiter}' tokens from the model.
    """
    tuple_delim = tuple_delim or PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    record_delim = record_delim or PROMPTS["DEFAULT_RECORD_DELIMITER"]
    completion_delim = completion_delim or PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

    raw = _normalize_quotes(raw)

    # Defensive: handles examples that slipped through or model echoes
    raw = raw.replace("{tuple_delimiter}", tuple_delim) \
             .replace("{record_delimiter}", record_delim) \
             .replace("{completion_delimiter}", completion_delim)
    

    # Truncate at completion delimiter if present
    if completion_delim in raw:
        raw = raw.split(completion_delim, 1)[0]

    # Some models echo headings, keep only after "Output:" if present
    if "Output:" in raw:
        raw = raw.split("Output:", 1)[1]

    # Split into records by record delimiter (tolerate trailing spaces/newlines)
    recs = re.split(rf"{re.escape(record_delim)}\s*", raw)
    recs = [r.strip() for r in recs if r.strip()]
    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    content_keywords: List[str] = []

    for rec in recs:
        body = _strip_parens(rec)
        # Split fields by tuple delimiter
        parts = [p.strip() for p in body.split(tuple_delim)]
        parts = [_strip_quotes(p) for p in parts if p.strip() != ""]

        if not parts:
            continue
        tag = parts[0].lower()

        if tag == "entity" and len(parts) >= 4:
            # ("entity", name, type, description)
            name, etype, desc = parts[1], parts[2], parts[3]
            entities.append({
                "name": name,
                "type": etype,
                "description": desc,
            })

        elif tag == "relationship" and len(parts) >= 6:
            # Supported formats:
            # 8-tuple: ("relationship", source_name, source_type, target_name, target_type, description, keywords, strength)
            # 6-tuple: ("relationship", source_name, target_name, description, keywords, strength)
            if len(parts) >= 8:
                _, src, src_type, tgt, tgt_type, desc, keywords, strength = parts[:8]
            else:
                _, src, tgt, desc, keywords, strength = parts[:6]
                src_type, tgt_type = "", ""  # no typed hints in legacy format
            relationships.append({
                "source_name": src,
                "source_type": src_type,
                "target_name": tgt,
                "target_type": tgt_type,
                "description": desc,
                "keywords": keywords,
                "weight": _to_float_or_none(strength),
            })

        elif tag == "content_keywords" and len(parts) >= 2:
            # ("content_keywords", "kw1, kw2, ...")
            kws = [k.strip() for k in parts[1].split(",") if k.strip()]
            content_keywords.extend(kws)

        else:
            # Unknown tag—keep raw for debugging (raw_records contains it)
            pass

    return ParsedOutput(
        entities=entities,
        relationships=relationships,
        content_keywords=content_keywords,
        raw_records=recs,
    )

# ─────────────────────────────────────────────────────────────
# Public extraction API (chunk-by-chunk + batch)
# ─────────────────────────────────────────────────────────────

def _get_chunk_text(chunk: Dict[str, Any]) -> str:
    for key in ("text", "content", "body"):
        if key in chunk and isinstance(chunk[key], str):
            return chunk[key]
    raise KeyError("Chunk is missing text content. Expected one of keys: 'text', 'content', or 'body'.")


def _require_chunk_uuid(chunk: Dict[str, Any]) -> str:
    if "chunk_uuid" not in chunk or not chunk["chunk_uuid"]:
        raise KeyError("Each chunk MUST include 'chunk_uuid' (used as source_id).")
    return str(chunk["chunk_uuid"])

# !! Currently not used, but could be useful for single-chunk extraction
def extract_entities_relations_for_chunk(
    chunk: Dict[str, Any],
    client: Chat,
    language: Optional[str] = None,
    entity_types: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Run the entity/relationship prompt for a single chunk and parse the result.
    - Sets entity['source_id'] = chunk['chunk_uuid']
    - Sets relation['source_id'] = chunk['chunk_uuid']
    - Also carries 'filepath' (or 'filename') if present
    """
    text = _get_chunk_text(chunk)
    prompt = build_entity_relation_prompt(text=text, language=language, entity_types=entity_types)
    system = "You extract entities and relationships precisely in the required format. Do not add commentary."

    raw = client.generate(prompt=prompt, system=system)
    parsed = parse_model_output(raw)

    # Attach source_id (strictly chunk_uuid) and filepath if provided on the chunk
    source_id = _require_chunk_uuid(chunk)
    filepath = chunk.get("filepath") or chunk.get("filename")

    for e in parsed.entities:
        e["source_id"] = source_id
        e["filepath"] = filepath
    for r in parsed.relationships:
        r["source_id"] = source_id
        r["filepath"] = filepath

    return parsed.entities, parsed.relationships, parsed.content_keywords

def extract_from_chunks(
    chunks: Iterable[Dict[str, Any]],
    language: Optional[str] = None,
    entity_types: Optional[Iterable[str]] = None,
    client: Optional[Chat] = None,
) -> Dict[str, Any]:
    """
    High-level convenience: iterate chunks, call LLM, parse, and return collected results.
    Returns dict with 'entities', 'relationships', 'content_keywords'.

    PERFORMANCE (no functional change, no prompt changes):
    - Reuses a single Chat client (singleton) to keep HTTP sessions hot.
    - Adds a content-addressed LLM cache in SQLite (model + prompt_sha + text_sha).
      *Cache stores RAW model text; on hits we parse it exactly like fresh outputs.*
    - Calls the LLM ONLY for cache misses; hits are stitched back in order.
    - Runs LLM calls for misses concurrently (bounded by settings-based concurrency).

    SETTINGS (instead of .env):
    - Concurrency:  settings.perf.max_concurrency       (fallback 6 if missing)
    - Cache on/off: settings.perf.cache_enabled         (fallback True if missing)
    - Cache TTL:    settings.perf.cache_max_age_hours   (fallback 720 if missing)

    Provenance:
    - Even on cache hits, each extracted record is stamped with this file/chunk's
      source identifiers so cross-file queries remain accurate.

    NOTE:
    - Prompts are built with your existing `build_entity_relation_prompt(...)`.
    - Parsing uses your existing `parse_model_output(...)`.
    - The #TODO you added is preserved below (applied per-chunk after parsing).
    """
    # Pull settings
    MAX_WORKERS = settings.llmperf.max_concurrency
    CACHE_ENABLED = settings.llmperf.cache_enabled
    CACHE_MAX_AGE_HOURS = settings.llmperf.cache_max_age_hours

    # 1) Shared LLM client + storage handles
    chat = client or Chat.singleton()
    storage = Storage()
    storage.init()

    # 2) Materialize chunks to keep stable indexing for stitching results
    chunk_list: List[Dict[str, Any]] = list(chunks)

    # 3) Build the exact prompts for each chunk
    #    and compute cache keys (prompt_sha, text_sha) per chunk.
    prompts: List[str] = []
    keys: List[Tuple[str, str]] = []  # (prompt_sha, text_sha)
    for ch in chunk_list:
        txt = _get_chunk_text(ch)
        prompt = build_entity_relation_prompt(
            text=txt,
            language=language,
            entity_types=entity_types,
        )
        prompts.append(prompt)
        keys.append((_sha256(prompt), _sha256(txt)))

    # 4) Probe cache; mark misses
    model_name = chat.model
    raw_outputs: List[Optional[str]] = [None] * len(chunk_list)
    to_run: List[int] = []

    if CACHE_ENABLED:
        for i, (psha, tsha) in enumerate(keys):
            cached = storage.get_llm_cache(model_name, psha, tsha, CACHE_MAX_AGE_HOURS)
            if cached is not None:
                raw_outputs[i] = cached
            else:
                to_run.append(i)
    else:
        to_run = list(range(len(chunk_list)))

    # 5) Call the LLM ONLY for cache misses, in parallel (bounded)
    def _call_one(i: int) -> str:
        return chat.generate(prompt=prompts[i], system="You extract entities and relationships precisely in the required format. Do not add commentary.")

    if to_run:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_call_one, i): i for i in to_run}
            for fut in as_completed(futs):
                i = futs[fut]
                out = fut.result()
                raw_outputs[i] = out
                if CACHE_ENABLED:
                    psha, tsha = keys[i]
                    storage.put_llm_cache(model_name, psha, tsha, out)

    # 6) Parse outputs and attach per-chunk provenance (identical to your approach)
    all_entities: List[Dict[str, Any]] = []
    all_relationships: List[Dict[str, Any]] = []
    all_keywords: List[str] = []

    for i, ch in enumerate(chunk_list):
        raw = raw_outputs[i] or ""
        parsed = parse_model_output(raw)   # existing parser

        source_id = _require_chunk_uuid(ch)
        filepath = ch.get("filepath") or ch.get("filename")

        for e in parsed.entities:
            e["source_id"] = source_id
            e["filepath"] = filepath
        for r in parsed.relationships:
            r["source_id"] = source_id
            r["filepath"] = filepath

        # TODO: ask whether the extraction is really completed
        # If not send with new prompt to complete

        all_entities.extend(parsed.entities)
        all_relationships.extend(parsed.relationships)
        all_keywords.extend(parsed.content_keywords)

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "content_keywords": sorted(set(all_keywords)),
    }

# ─────────────────────────────────────────────────────────────
# CLI (optional) — quick test driver
# ─────────────────────────────────────────────────────────────

def _load_chunks_from_path(path: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a JSON or JSONL file.
    Each record should be a dict with at least 'text' or 'content' and 'chunk_uuid'.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        # Try JSON array
        obj = json.loads(data)
        if isinstance(obj, list):
            return obj  # type: ignore[return-value]
        raise ValueError("Expected a list of chunks in JSON.")
    except json.JSONDecodeError:
        # Try JSONL
        chunks: List[Dict[str, Any]] = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
        return chunks


def _default_demo_chunks() -> List[Dict[str, Any]]:
    return [{
        "chunk_uuid": "demo-1",
        "text": "Apple launched the Vision Pro with help from Foxconn. Tim Cook presented it in Cupertino during WWDC.",
        "filename": "/demo/path/a.txt"
    }]


def _print_summary(res: Dict[str, Any]) -> None:
    print("\nEntities:")
    for e in res["entities"]:
        print(f"  - {e['name']} [{e['type']}]  src={e.get('source_id')}")
    print("\nRelationships:")
    for r in res["relationships"]:
        w = r.get("weight")
        print(f"  - {r['source_name']} <-> {r['target_name']}  w={w}  src={r.get('source_id')}")
    if res["content_keywords"]:
        print("\nKeywords:", ", ".join(res["content_keywords"]))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract entities and relations from chunks using Azure OpenAI and regex parsing."
    )
    ap.add_argument("--chunks", type=str, default="", help="Path to JSON/JSONL with chunks (each has text/content and chunk_uuid).")
    ap.add_argument("--language", type=str, default="", help="Override output language (default from prompts).")
    ap.add_argument("--entity-types", type=str, default="", help="Comma-separated entity types to enforce.")
    args = ap.parse_args()

    language = args.language or None
    entity_types = [s.strip() for s in args.entity_types.split(",")] if args.entity_types else None

    chunks = _default_demo_chunks() if not args.chunks else _load_chunks_from_path(args.chunks)

    client = Chat.singleton()  # reads env
    result = extract_from_chunks(chunks, language=language, entity_types=entity_types, client=client)

    # Pretty print
    _print_summary(result)

    # Also dump JSON
    print("\n\nJSON Output:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
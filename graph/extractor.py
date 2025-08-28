from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Any, Dict, Iterable, List, Optional, Tuple
load_dotenv()
# ─────────────────────────────────────────────────────────────
# Load prompt templates from your uploaded prompts.py
# ─────────────────────────────────────────────────────────────
try:
    from prompts import PROMPTS  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Could not import PROMPTS from prompts.py. Place prompts.py next to this script.") from e

# ─────────────────────────────────────────────────────────────
# Azure OpenAI chat client (env-driven)
#   Required env vars:
#     - AZURE_OPENAI_ENDPOINT
#     - AZURE_OPENAI_API_KEY
#     - AZURE_OPENAI_CHAT_DEPLOYMENT_NAME   (the chat model deployment name)
#     - AZURE_OPENAI_API_VERSION            (optional, defaults to 2024-02-15-preview)
# ─────────────────────────────────────────────────────────────
try:
    from openai import AzureOpenAI  # type: ignore
except Exception:
    AzureOpenAI = None  # type: ignore


@dataclass
class AzureConfig:
    endpoint: str
    api_key: str
    api_version: str = "2024-02-15-preview"
    chat_deployment: Optional[str] = None  # AZURE_OPENAI_CHAT_DEPLOYMENT_NAME


class AzureChat:
    def __init__(self, cfg: Optional[AzureConfig] = None):
        if AzureOpenAI is None:
            raise RuntimeError("openai package not installed. pip install openai")

        if cfg is None:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            chat_deployment = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME")
            if not endpoint or not api_key or not chat_deployment:
                raise RuntimeError(
                    "Missing Azure env vars. Set AZURE_OPENAI_ENDPOINT, "
                    "AZURE_OPENAI_API_KEY, AZURE_OPENAI_LLM_DEPLOYMENT_NAME"
                )
            cfg = AzureConfig(
                endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                chat_deployment=chat_deployment,
            )
        self.cfg = cfg
        self.client = AzureOpenAI(
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            azure_endpoint=cfg.endpoint,
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.cfg.chat_deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────

def build_entity_relation_prompt(
    text: str,
    language: Optional[str] = None,
    entity_types: Optional[Iterable[str]] = None,
) -> str:
    """
    Fill PROMPTS['entity_extraction'] for a single chunk of text, using your examples and delimiters.
    """
    examples = "\n\n".join(PROMPTS.get("entity_extraction_examples", []))
    ctx = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types) if entity_types else ", ".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        examples=examples,
        language=language or PROMPTS["DEFAULT_LANGUAGE"],
        input_text=text,
    )
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
    """
    tuple_delim = tuple_delim or PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    record_delim = record_delim or PROMPTS["DEFAULT_RECORD_DELIMITER"]
    completion_delim = completion_delim or PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

    raw = _normalize_quotes(raw)

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
            # ("relationship", source, target, description, keywords, strength)
            src, tgt, desc, keywords, strength = parts[1], parts[2], parts[3], parts[4], parts[5]
            relationships.append({
                "source_name": src,
                "target_name": tgt,
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


def extract_entities_relations_for_chunk(
    chunk: Dict[str, Any],
    client: AzureChat,
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
    client: Optional[AzureChat] = None,
) -> Dict[str, Any]:
    """
    High-level convenience: iterate chunks, call LLM, parse, and return collected results.
    Returns dict with 'entities', 'relationships', 'content_keywords'.
    """
    if client is None:
        client = AzureChat()  # read env

    all_entities: List[Dict[str, Any]] = []
    all_relationships: List[Dict[str, Any]] = []
    all_keywords: List[str] = []

    for ch in chunks:
        ents, rels, kws = extract_entities_relations_for_chunk(
            ch, client=client, language=language, entity_types=entity_types
        )
        all_entities.extend(ents)
        all_relationships.extend(rels)
        all_keywords.extend(kws)

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

    client = AzureChat()  # reads env
    result = extract_from_chunks(chunks, language=language, entity_types=entity_types, client=client)

    # Pretty print
    _print_summary(result)

    # Also dump JSON
    print("\n\nJSON Output:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
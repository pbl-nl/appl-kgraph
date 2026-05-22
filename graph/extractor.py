from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from db_storage import Storage
from llm import Chat
from prompts import PROMPTS
from settings import settings
from utils import detect_language, normalize_language_name


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _default_entity_types() -> List[str]:
    return list(settings.prompts.default_entity_types or PROMPTS["DEFAULT_ENTITY_TYPES"])


def build_entity_relation_prompt(
    text: str,
    language: Optional[str] = None,
    entity_types: Optional[Iterable[str]] = None,
) -> str:
    examples_template = "\n\n".join(PROMPTS.get("entity_extraction_examples", []))
    ctx = dict(
        tuple_delimiter=settings.prompts.tuple_delimiter,
        record_delimiter=settings.prompts.record_delimiter,
        completion_delimiter=settings.prompts.completion_delimiter,
        entity_types=", ".join(entity_types or _default_entity_types()),
        examples="",
        language=normalize_language_name(language, settings.prompts.default_language),
        input_text=text,
    )
    ctx["examples"] = examples_template.format(**ctx)
    return PROMPTS["entity_extraction"].format(**ctx)


def build_entity_audit_prompt(
    text: str,
    *,
    initial_extraction: str,
    language: Optional[str] = None,
    entity_types: Optional[Iterable[str]] = None,
) -> str:
    return PROMPTS["entity_extraction_audit"].format(
        input_text=text,
        initial_extraction=initial_extraction,
        language=normalize_language_name(language, settings.prompts.default_language),
        entity_types=", ".join(entity_types or _default_entity_types()),
    )


@dataclass
class ParsedOutput:
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    content_keywords: List[str]
    raw_records: List[str]


_FANCY_QUOTES = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‘": "'",
    "’": "'",
}


def _normalize_quotes(s: str) -> str:
    for key, value in _FANCY_QUOTES.items():
        s = s.replace(key, value)
    return s


def _strip_parens(s: str) -> str:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        return s[1:-1].strip()
    return s


def _strip_quotes(s: str) -> str:
    s = _normalize_quotes(s.strip())
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _to_float_or_none(x: str) -> Optional[float]:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x.strip())
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def parse_model_output(
    raw: str,
    tuple_delim: Optional[str] = None,
    record_delim: Optional[str] = None,
    completion_delim: Optional[str] = None,
) -> ParsedOutput:
    tuple_delim = tuple_delim or settings.prompts.tuple_delimiter
    record_delim = record_delim or settings.prompts.record_delimiter
    completion_delim = completion_delim or settings.prompts.completion_delimiter

    raw = _normalize_quotes(raw)
    raw = raw.replace("{tuple_delimiter}", tuple_delim)
    raw = raw.replace("{record_delimiter}", record_delim)
    raw = raw.replace("{completion_delimiter}", completion_delim)

    if completion_delim in raw:
        raw = raw.split(completion_delim, 1)[0]
    if "Output:" in raw:
        raw = raw.split("Output:", 1)[1]

    records = re.split(rf"{re.escape(record_delim)}\s*", raw)
    records = [record.strip() for record in records if record.strip()]

    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    content_keywords: List[str] = []

    for record in records:
        body = _strip_parens(record)
        parts = [_strip_quotes(part.strip()) for part in body.split(tuple_delim) if part.strip()]
        if not parts:
            continue

        tag = parts[0].lower()
        if tag == "entity" and len(parts) >= 4:
            entities.append(
                {
                    "name": parts[1],
                    "type": parts[2],
                    "description": parts[3],
                }
            )
        elif tag == "relationship" and len(parts) >= 6:
            _, src, tgt, desc, keywords, strength = parts[:6]
            relationships.append(
                {
                    "source_name": src,
                    "target_name": tgt,
                    "description": desc,
                    "keywords": keywords,
                    "weight": _to_float_or_none(strength),
                }
            )
        elif tag == "content_keywords" and len(parts) >= 2:
            content_keywords.extend([part.strip() for part in parts[1].split(",") if part.strip()])

    return ParsedOutput(
        entities=entities,
        relationships=relationships,
        content_keywords=content_keywords,
        raw_records=records,
    )


def _get_chunk_text(chunk: Dict[str, Any]) -> str:
    for key in ("text", "content", "body"):
        if key in chunk and isinstance(chunk[key], str):
            return chunk[key]
    raise KeyError("Chunk is missing text content. Expected one of keys: 'text', 'content', or 'body'.")


def _require_chunk_uuid(chunk: Dict[str, Any]) -> str:
    if "chunk_uuid" not in chunk or not chunk["chunk_uuid"]:
        raise KeyError("Each chunk MUST include 'chunk_uuid' (used as source_id).")
    return str(chunk["chunk_uuid"])


def _resolve_chunk_language(
    chunk: Dict[str, Any],
    *,
    explicit_language: Optional[str] = None,
) -> str:
    if explicit_language:
        return normalize_language_name(explicit_language, settings.prompts.default_language)

    default_language = settings.prompts.default_language
    if settings.extraction.use_chunk_language:
        for key in ("chunk_language", "language", "document_language"):
            if chunk.get(key):
                return normalize_language_name(str(chunk.get(key)), default_language)
        if settings.extraction.detect_chunk_language:
            detected = detect_language(_get_chunk_text(chunk))
            if detected and detected != "unknown":
                return normalize_language_name(detected, default_language)
    elif chunk.get("document_language"):
        return normalize_language_name(str(chunk.get("document_language")), default_language)

    return normalize_language_name(None, default_language)


def _ensure_storage(storage: Optional[Storage]) -> Storage:
    active_storage = storage or Storage()
    active_storage.init()
    return active_storage


def _run_cached_prompts(
    *,
    chat: Chat,
    storage: Storage,
    prompts: List[str],
    text_hash_inputs: List[str],
    system_prompt: str,
) -> List[str]:
    max_workers = settings.llmperf.max_concurrency
    cache_enabled = settings.llmperf.cache_enabled
    cache_max_age_hours = settings.llmperf.cache_max_age_hours
    model_name = chat.model

    keys = [(_sha256(prompt), _sha256(text_hash)) for prompt, text_hash in zip(prompts, text_hash_inputs)]
    raw_outputs: List[Optional[str]] = [None] * len(prompts)
    to_run: List[int] = []

    if cache_enabled:
        for index, (prompt_sha, text_sha) in enumerate(keys):
            cached = storage.get_llm_cache(model_name, prompt_sha, text_sha, cache_max_age_hours)
            if cached is None:
                to_run.append(index)
            else:
                raw_outputs[index] = cached
    else:
        to_run = list(range(len(prompts)))

    def _call_one(index: int) -> str:
        return chat.generate(prompt=prompts[index], system=system_prompt)

    if to_run:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_call_one, index): index for index in to_run}
            for future in as_completed(futures):
                index = futures[future]
                output = future.result()
                raw_outputs[index] = output
                if cache_enabled:
                    prompt_sha, text_sha = keys[index]
                    storage.put_llm_cache(model_name, prompt_sha, text_sha, output)

    return [output or "" for output in raw_outputs]


def _parse_audit_output(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not text:
        return {"missing_entities": [], "missing_relationships": [], "summary": ""}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {
                "missing_entities": [],
                "missing_relationships": [],
                "summary": "",
                "raw_response": text,
            }
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "missing_entities": [],
                "missing_relationships": [],
                "summary": "",
                "raw_response": text,
            }

    if not isinstance(parsed, dict):
        return {"missing_entities": [], "missing_relationships": [], "summary": "", "raw_response": text}

    return {
        "missing_entities": parsed.get("missing_entities", []) or [],
        "missing_relationships": parsed.get("missing_relationships", []) or [],
        "summary": parsed.get("summary", "") or "",
    }


def extract_from_chunks(
    chunks: Iterable[Dict[str, Any]],
    language: Optional[str] = None,
    entity_types: Optional[Iterable[str]] = None,
    client: Optional[Chat] = None,
    storage: Optional[Storage] = None,
    audit_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    chat = client or Chat.singleton()
    active_storage = _ensure_storage(storage)
    chunk_list = list(chunks)
    resolved_entity_types = list(entity_types or _default_entity_types())
    do_audit = settings.extraction.audit_second_pass_enabled if audit_enabled is None else audit_enabled

    chunk_languages = [
        _resolve_chunk_language(chunk, explicit_language=language)
        for chunk in chunk_list
    ]
    chunk_texts = [_get_chunk_text(chunk) for chunk in chunk_list]
    extraction_prompts = [
        build_entity_relation_prompt(
            text=text,
            language=chunk_language,
            entity_types=resolved_entity_types,
        )
        for text, chunk_language in zip(chunk_texts, chunk_languages)
    ]

    extraction_outputs = _run_cached_prompts(
        chat=chat,
        storage=active_storage,
        prompts=extraction_prompts,
        text_hash_inputs=chunk_texts,
        system_prompt="You extract entities and relationships precisely in the required format. Do not add commentary.",
    )

    all_entities: List[Dict[str, Any]] = []
    all_relationships: List[Dict[str, Any]] = []
    all_keywords: List[str] = []
    chunk_results: List[Dict[str, Any]] = []

    for chunk, chunk_language, raw_output in zip(chunk_list, chunk_languages, extraction_outputs):
        parsed = parse_model_output(raw_output)
        source_id = _require_chunk_uuid(chunk)
        filepath = chunk.get("filepath") or chunk.get("filename")

        entities = []
        for entity in parsed.entities:
            stamped = dict(entity)
            stamped["source_id"] = source_id
            stamped["filepath"] = filepath
            entities.append(stamped)

        relationships = []
        for relationship in parsed.relationships:
            stamped = dict(relationship)
            stamped["source_id"] = source_id
            stamped["filepath"] = filepath
            relationships.append(stamped)

        all_entities.extend(entities)
        all_relationships.extend(relationships)
        all_keywords.extend(parsed.content_keywords)
        chunk_results.append(
            {
                "chunk_uuid": source_id,
                "filepath": filepath,
                "language": chunk_language,
                "entities": entities,
                "relationships": relationships,
                "content_keywords": parsed.content_keywords,
                "raw_output": raw_output,
            }
        )

    audits: List[Dict[str, Any]] = []
    if do_audit and chunk_results:
        audit_prompts = []
        audit_hash_inputs = []
        for chunk, chunk_result, chunk_language in zip(chunk_list, chunk_results, chunk_languages):
            extraction_snapshot = json.dumps(
                {
                    "entities": chunk_result["entities"],
                    "relationships": chunk_result["relationships"],
                    "content_keywords": chunk_result["content_keywords"],
                },
                ensure_ascii=False,
            )
            audit_prompts.append(
                build_entity_audit_prompt(
                    _get_chunk_text(chunk),
                    initial_extraction=extraction_snapshot,
                    language=chunk_language,
                    entity_types=resolved_entity_types,
                )
            )
            audit_hash_inputs.append(f"{_get_chunk_text(chunk)}\n{extraction_snapshot}")

        audit_outputs = _run_cached_prompts(
            chat=chat,
            storage=active_storage,
            prompts=audit_prompts,
            text_hash_inputs=audit_hash_inputs,
            system_prompt="You audit extraction completeness. Return JSON only.",
        )

        for chunk_result, raw_audit in zip(chunk_results, audit_outputs):
            parsed_audit = _parse_audit_output(raw_audit)
            audits.append(
                {
                    "chunk_uuid": chunk_result["chunk_uuid"],
                    "filepath": chunk_result["filepath"],
                    "language": chunk_result["language"],
                    **parsed_audit,
                    "raw_output": raw_audit,
                }
            )

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "content_keywords": sorted(set(all_keywords)),
        "chunk_results": chunk_results,
        "audits": audits,
    }


def extract_entities_relations_for_chunk(
    chunk: Dict[str, Any],
    client: Chat,
    language: Optional[str] = None,
    entity_types: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    result = extract_from_chunks(
        [chunk],
        language=language,
        entity_types=entity_types,
        client=client,
        audit_enabled=False,
    )
    return result["entities"], result["relationships"], result["content_keywords"]


def _load_chunks_from_path(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()
    try:
        obj = json.loads(data)
        if isinstance(obj, list):
            return obj
        raise ValueError("Expected a list of chunks in JSON.")
    except json.JSONDecodeError:
        chunks: List[Dict[str, Any]] = []
        for line in data.splitlines():
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
        return chunks


def _default_demo_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "chunk_uuid": "demo-1",
            "text": "Apple launched the Vision Pro with help from Foxconn. Tim Cook presented it in Cupertino during WWDC.",
            "filename": "/demo/path/a.txt",
        }
    ]


def _print_summary(res: Dict[str, Any]) -> None:
    print("\nEntities:")
    for entity in res["entities"]:
        print(f"  - {entity['name']} [{entity['type']}]  src={entity.get('source_id')}")
    print("\nRelationships:")
    for relationship in res["relationships"]:
        print(
            f"  - {relationship['source_name']} <-> {relationship['target_name']} "
            f"w={relationship.get('weight')} src={relationship.get('source_id')}"
        )
    if res["content_keywords"]:
        print("\nKeywords:", ", ".join(res["content_keywords"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract entities and relations from chunks using the configured LLM and regex parsing."
    )
    parser.add_argument("--chunks", type=str, default="", help="Path to JSON/JSONL with chunks.")
    parser.add_argument("--language", type=str, default="", help="Override output language.")
    parser.add_argument("--entity-types", type=str, default="", help="Comma-separated entity types.")
    args = parser.parse_args()

    explicit_language = args.language or None
    entity_types = [s.strip() for s in args.entity_types.split(",")] if args.entity_types else None
    chunks = _default_demo_chunks() if not args.chunks else _load_chunks_from_path(args.chunks)

    result = extract_from_chunks(
        chunks,
        language=explicit_language,
        entity_types=entity_types,
        client=Chat.singleton(),
        audit_enabled=settings.extraction.audit_second_pass_enabled,
    )
    _print_summary(result)
    print("\n\nJSON Output:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DocumentRef:
    path: Path
    root: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawDocument:
    ref: DocumentRef
    pages: Sequence[Tuple[int, str]]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnrichedDocument:
    raw: RawDocument
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_uuid: str
    doc_id: str
    chunk_id: int
    filename: str
    text: str
    char_count: int
    start_page: int
    end_page: int
    filepath: Optional[str] = None
    document_language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Entity:
    name: str
    type: str
    description: str
    source_id: Optional[str] = None
    filepath: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    source_name: str
    target_name: str
    description: str
    keywords: str = ""
    weight: Optional[float] = None
    source_id: Optional[str] = None
    filepath: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    """Canonical graph findings returned by the extraction stage."""

    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    content_keywords: List[str] = field(default_factory=list)
    chunk_results: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphDelta:
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryPlan:
    query: str
    history: Sequence[Tuple[str, str]] = field(default_factory=list)
    high_level_keywords: List[str] = field(default_factory=list)
    low_level_keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityCandidate:
    name: str
    type: Optional[str]
    description: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationCandidate:
    source_name: str
    target_name: str
    description: str
    score: float
    keywords: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkCandidate:
    chunk_uuid: str
    document_id: str
    filename: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalCandidates:
    """Scored records returned by candidate retrieval before context assembly."""

    query_plan: QueryPlan
    entities: List[EntityCandidate] = field(default_factory=list)
    relations: List[RelationCandidate] = field(default_factory=list)
    chunks: List[ChunkCandidate] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedContext:
    query_plan: QueryPlan
    context_windows: List[Any] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    context: RetrievedContext
    model: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


_ENTITY_LEGACY_FIELDS = {
    "name",
    "type",
    "description",
    "source_id",
    "filepath",
    "metadata",
}
_RELATION_LEGACY_FIELDS = {
    "source_name",
    "target_name",
    "description",
    "keywords",
    "weight",
    "source_id",
    "filepath",
    "metadata",
}
_EXTRACTION_LEGACY_FIELDS = {
    "entities",
    "relationships",
    "relations",
    "content_keywords",
    "chunk_results",
    "validation_results",
    "audits",
    "metadata",
}


def _legacy_metadata(
    payload: Mapping[str, Any],
    known_fields: set,
) -> Dict[str, Any]:
    nested = payload.get("metadata") or {}
    if not isinstance(nested, Mapping):
        raise TypeError("legacy metadata must be a mapping")
    metadata = deepcopy(dict(nested))
    for key, value in payload.items():
        if key not in known_fields:
            metadata[key] = deepcopy(value)
    return metadata


def _legacy_list(value: Any, field_name: str) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"legacy {field_name} must be a sequence")
    return deepcopy(list(value))


def _entity_from_legacy(payload: Any) -> Entity:
    if isinstance(payload, Entity):
        return Entity(
            name=payload.name,
            type=payload.type,
            description=payload.description,
            source_id=payload.source_id,
            filepath=payload.filepath,
            metadata=deepcopy(payload.metadata),
        )
    if not isinstance(payload, Mapping):
        raise TypeError("legacy entity must be a mapping")
    return Entity(
        name=payload.get("name", ""),
        type=payload.get("type", "unknown") or "unknown",
        description=payload.get("description", "") or "",
        source_id=payload.get("source_id"),
        filepath=payload.get("filepath"),
        metadata=_legacy_metadata(payload, _ENTITY_LEGACY_FIELDS),
    )


def _relation_from_legacy(payload: Any) -> Relation:
    if isinstance(payload, Relation):
        return Relation(
            source_name=payload.source_name,
            target_name=payload.target_name,
            description=payload.description,
            keywords=payload.keywords,
            weight=payload.weight,
            source_id=payload.source_id,
            filepath=payload.filepath,
            metadata=deepcopy(payload.metadata),
        )
    if not isinstance(payload, Mapping):
        raise TypeError("legacy relation must be a mapping")
    return Relation(
        source_name=payload.get("source_name", ""),
        target_name=payload.get("target_name", ""),
        description=payload.get("description", "") or "",
        keywords=payload.get("keywords", "") or "",
        weight=payload.get("weight"),
        source_id=payload.get("source_id"),
        filepath=payload.get("filepath"),
        metadata=_legacy_metadata(payload, _RELATION_LEGACY_FIELDS),
    )


def extraction_result_from_legacy(payload: Mapping[str, Any]) -> ExtractionResult:
    """Convert the extractor's dictionary result to the canonical contract."""

    if not isinstance(payload, Mapping):
        raise TypeError("legacy extraction result must be a mapping")
    relationships = payload.get("relationships")
    if relationships is None:
        relationships = payload.get("relations")
    diagnostics = payload.get("validation_results") or payload.get("audits")
    return ExtractionResult(
        entities=[
            _entity_from_legacy(item)
            for item in _legacy_list(payload.get("entities"), "entities")
        ],
        relations=[
            _relation_from_legacy(item)
            for item in _legacy_list(relationships, "relationships")
        ],
        content_keywords=_legacy_list(
            payload.get("content_keywords"),
            "content_keywords",
        ),
        chunk_results=_legacy_list(payload.get("chunk_results"), "chunk_results"),
        diagnostics=_legacy_list(diagnostics, "validation_results"),
        metadata=_legacy_metadata(payload, _EXTRACTION_LEGACY_FIELDS),
    )


def _entity_to_legacy(entity: Entity) -> Dict[str, Any]:
    payload = deepcopy(entity.metadata)
    payload.update(
        {
            "name": entity.name,
            "type": entity.type,
            "description": entity.description,
            "source_id": entity.source_id,
            "filepath": entity.filepath,
        }
    )
    return payload


def _relation_to_legacy(relation: Relation) -> Dict[str, Any]:
    payload = deepcopy(relation.metadata)
    payload.update(
        {
            "source_name": relation.source_name,
            "target_name": relation.target_name,
            "description": relation.description,
            "keywords": relation.keywords,
            "weight": relation.weight,
            "source_id": relation.source_id,
            "filepath": relation.filepath,
        }
    )
    return payload


def extraction_result_to_legacy(result: ExtractionResult) -> Dict[str, Any]:
    """Convert the canonical extraction result to the current caller shape."""

    payload = deepcopy(result.metadata)
    diagnostics = deepcopy(result.diagnostics)
    payload.update(
        {
            "entities": [_entity_to_legacy(entity) for entity in result.entities],
            "relationships": [
                _relation_to_legacy(relation) for relation in result.relations
            ],
            "content_keywords": deepcopy(result.content_keywords),
            "chunk_results": deepcopy(result.chunk_results),
            "validation_results": diagnostics,
            "audits": deepcopy(diagnostics),
        }
    )
    return payload

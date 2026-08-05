from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _require_identifier(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_non_negative_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be zero or greater")


def _owned_metadata(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return deepcopy(dict(value))


def _normalized_source_ids(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("source_ids must be a sequence of strings")
    source_ids = tuple(value)
    for source_id in source_ids:
        _require_identifier(source_id, "source_id")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_ids must not contain duplicates")
    return source_ids


def _finite_score(value: Any, field_name: str = "score") -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a number") from exc
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


@dataclass(frozen=True)
class DocumentRef:
    path: Path
    root: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class RawDocument:
    """Parsed source with one stable ID and zero-based contiguous pages."""

    ref: DocumentRef
    doc_id: str
    pages: Tuple[Tuple[int, str], ...]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.doc_id, "doc_id")
        if isinstance(self.pages, (str, bytes)) or not isinstance(
            self.pages,
            Sequence,
        ):
            raise TypeError("pages must be a sequence of (page_number, text) pairs")

        normalized_pages = []
        for expected_number, page in enumerate(self.pages):
            if (
                isinstance(page, (str, bytes))
                or not isinstance(page, Sequence)
                or len(page) != 2
            ):
                raise TypeError("each page must be a (page_number, text) pair")
            page_number, page_text = page
            _require_non_negative_integer(page_number, "page number")
            if page_number != expected_number:
                raise ValueError("page numbers must be zero-based and contiguous")
            if not isinstance(page_text, str):
                raise TypeError("page text must be a string")
            normalized_pages.append((page_number, page_text))

        if not isinstance(self.text, str):
            raise TypeError("document text must be a string")
        if self.text != "\n".join(text for _, text in normalized_pages):
            raise ValueError("document text must equal the newline-joined page text")
        object.__setattr__(self, "pages", tuple(normalized_pages))
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class EnrichedDocument:
    raw: RawDocument
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    transformations: Tuple[str, ...] = field(default_factory=tuple)
    pages: Optional[Tuple[Tuple[int, str], ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.raw, RawDocument):
            raise TypeError("raw must be a RawDocument")
        if not isinstance(self.text, str):
            raise TypeError("enriched text must be a string")
        pages = self.raw.pages if self.pages is None else tuple(self.pages)
        for expected_number, page in enumerate(pages):
            if (
                isinstance(page, (str, bytes))
                or not isinstance(page, Sequence)
                or len(page) != 2
            ):
                raise TypeError("each enriched page must be a (page_number, text) pair")
            page_number, page_text = page
            _require_non_negative_integer(page_number, "enriched page number")
            if page_number != expected_number:
                raise ValueError("enriched page numbers must be zero-based and contiguous")
            if not isinstance(page_text, str):
                raise TypeError("enriched page text must be a string")
        if self.text != "\n".join(page_text for _, page_text in pages):
            raise ValueError("enriched text must equal the newline-joined page text")
        object.__setattr__(self, "pages", tuple(pages))
        transformations = tuple(self.transformations)
        for transformation in transformations:
            _require_identifier(transformation, "transformation name")
        object.__setattr__(self, "transformations", transformations)
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class Chunk:
    """Stored chunk with stable IDs and an inclusive zero-based page range."""

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

    def __post_init__(self) -> None:
        _require_identifier(self.chunk_uuid, "chunk_uuid")
        _require_identifier(self.doc_id, "doc_id")
        _require_non_negative_integer(self.chunk_id, "chunk_id")
        _require_non_negative_integer(self.char_count, "char_count")
        _require_non_negative_integer(self.start_page, "start_page")
        _require_non_negative_integer(self.end_page, "end_page")
        if self.end_page < self.start_page:
            raise ValueError("end_page must be greater than or equal to start_page")
        if not isinstance(self.text, str):
            raise TypeError("chunk text must be a string")
        if self.char_count != len(self.text):
            raise ValueError("char_count must equal the length of chunk text")
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class Entity:
    name: str
    type: str
    description: str
    source_ids: Tuple[str, ...] = field(default_factory=tuple)
    filepath: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.name, "entity name")
        _require_identifier(self.type, "entity type")
        if not isinstance(self.description, str):
            raise TypeError("entity description must be a string")
        object.__setattr__(self, "source_ids", _normalized_source_ids(self.source_ids))
        if self.filepath is not None and not isinstance(self.filepath, str):
            raise TypeError("entity filepath must be a string or None")
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class Relation:
    source_name: str
    target_name: str
    description: str
    keywords: str = ""
    weight: Optional[float] = None
    source_ids: Tuple[str, ...] = field(default_factory=tuple)
    filepath: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.source_name, "relation source_name")
        _require_identifier(self.target_name, "relation target_name")
        if not isinstance(self.description, str):
            raise TypeError("relation description must be a string")
        if not isinstance(self.keywords, str):
            raise TypeError("relation keywords must be a string")
        if self.weight is not None:
            object.__setattr__(self, "weight", _finite_score(self.weight, "weight"))
        object.__setattr__(self, "source_ids", _normalized_source_ids(self.source_ids))
        if self.filepath is not None and not isinstance(self.filepath, str):
            raise TypeError("relation filepath must be a string or None")
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class ExtractionResult:
    """Canonical graph findings returned by the extraction stage."""

    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    content_keywords: List[str] = field(default_factory=list)
    chunk_results: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(isinstance(entity, Entity) for entity in self.entities):
            raise TypeError("entities must contain Entity objects")
        if not all(isinstance(relation, Relation) for relation in self.relations):
            raise TypeError("relations must contain Relation objects")
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class GraphDelta:
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    deleted_entity_names: Tuple[str, ...] = field(default_factory=tuple)
    deleted_relation_pairs: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(isinstance(entity, Entity) for entity in self.entities):
            raise TypeError("entities must contain Entity objects")
        if not all(isinstance(relation, Relation) for relation in self.relations):
            raise TypeError("relations must contain Relation objects")
        entity_names = [entity.name for entity in self.entities]
        if len(set(entity_names)) != len(entity_names):
            raise ValueError("GraphDelta entity names must be unique")
        relation_pairs = [
            tuple(sorted((relation.source_name, relation.target_name)))
            for relation in self.relations
        ]
        if len(set(relation_pairs)) != len(relation_pairs):
            raise ValueError("GraphDelta relation pairs must be unique")
        deleted_entity_names = tuple(self.deleted_entity_names)
        for name in deleted_entity_names:
            _require_identifier(name, "deleted entity name")
        if len(set(deleted_entity_names)) != len(deleted_entity_names):
            raise ValueError("deleted entity names must be unique")
        deleted_relation_pairs = tuple(
            tuple(sorted(pair)) for pair in self.deleted_relation_pairs
        )
        for pair in deleted_relation_pairs:
            if len(pair) != 2:
                raise ValueError("deleted relation pairs must contain two names")
            _require_identifier(pair[0], "deleted relation source name")
            _require_identifier(pair[1], "deleted relation target name")
        if len(set(deleted_relation_pairs)) != len(deleted_relation_pairs):
            raise ValueError("deleted relation pairs must be unique")
        object.__setattr__(self, "deleted_entity_names", deleted_entity_names)
        object.__setattr__(self, "deleted_relation_pairs", deleted_relation_pairs)
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class QueryPlan:
    query: str
    history: Sequence[Tuple[str, str]] = field(default_factory=list)
    high_level_keywords: List[str] = field(default_factory=list)
    low_level_keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class EntityCandidate:
    name: str
    type: Optional[str]
    description: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.name, "entity candidate name")
        if self.type is not None and not isinstance(self.type, str):
            raise TypeError("entity candidate type must be a string or None")
        if not isinstance(self.description, str):
            raise TypeError("entity candidate description must be a string")
        object.__setattr__(self, "score", _finite_score(self.score))
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class RelationCandidate:
    source_name: str
    target_name: str
    description: str
    score: float
    keywords: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.source_name, "relation candidate source_name")
        _require_identifier(self.target_name, "relation candidate target_name")
        if not isinstance(self.description, str):
            raise TypeError("relation candidate description must be a string")
        if not isinstance(self.keywords, str):
            raise TypeError("relation candidate keywords must be a string")
        object.__setattr__(self, "score", _finite_score(self.score))
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class ChunkCandidate:
    chunk_uuid: str
    document_id: str
    filename: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.chunk_uuid, "chunk candidate chunk_uuid")
        _require_identifier(self.document_id, "chunk candidate document_id")
        if not isinstance(self.filename, str):
            raise TypeError("chunk candidate filename must be a string")
        if not isinstance(self.text, str):
            raise TypeError("chunk candidate text must be a string")
        object.__setattr__(self, "score", _finite_score(self.score))
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class RetrievalCandidates:
    """Scored records returned by candidate retrieval before context assembly."""

    query_plan: QueryPlan
    entities: List[EntityCandidate] = field(default_factory=list)
    relations: List[RelationCandidate] = field(default_factory=list)
    chunks: List[ChunkCandidate] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query_plan, QueryPlan):
            raise TypeError("query_plan must be a QueryPlan")
        if not all(isinstance(item, EntityCandidate) for item in self.entities):
            raise TypeError("entities must contain EntityCandidate objects")
        if not all(isinstance(item, RelationCandidate) for item in self.relations):
            raise TypeError("relations must contain RelationCandidate objects")
        if not all(isinstance(item, ChunkCandidate) for item in self.chunks):
            raise TypeError("chunks must contain ChunkCandidate objects")
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class RetrievedContext:
    query_plan: QueryPlan
    context_windows: List[Any] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query_plan, QueryPlan):
            raise TypeError("query_plan must be a QueryPlan")
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    context: RetrievedContext
    model: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise TypeError("answer must be a string")
        if not isinstance(self.context, RetrievedContext):
            raise TypeError("context must be a RetrievedContext")
        if not isinstance(self.model, Mapping):
            raise TypeError("model must be a mapping")
        object.__setattr__(self, "model", deepcopy(dict(self.model)))
        object.__setattr__(self, "metadata", _owned_metadata(self.metadata))


_ENTITY_LEGACY_FIELDS = {
    "name",
    "type",
    "description",
    "source_id",
    "source_ids",
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
    "source_ids",
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
_ENTITY_CANDIDATE_LEGACY_FIELDS = {
    "name",
    "type",
    "description",
    "score",
    "metadata",
}
_RELATION_CANDIDATE_LEGACY_FIELDS = {
    "source_name",
    "target_name",
    "description",
    "keywords",
    "score",
    "metadata",
}
_CHUNK_CANDIDATE_LEGACY_FIELDS = {
    "chunk_uuid",
    "document_id",
    "doc_id",
    "filename",
    "text",
    "score",
    "metadata",
}
_RETRIEVAL_CANDIDATE_LEGACY_FIELDS = {
    "entities",
    "entity_matches",
    "relations",
    "relation_matches",
    "chunks",
    "chunk_matches",
    "metadata",
}


def _legacy_metadata(
    payload: Mapping[str, Any],
    known_fields: set,
) -> Dict[str, Any]:
    nested = payload.get("metadata")
    if nested is None:
        nested = {}
    elif not isinstance(nested, Mapping):
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


def _source_ids_from_legacy(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    value = payload.get("source_ids")
    if value is None:
        value = payload.get("source_id")
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        values = tuple(part.strip() for part in value.split("||") if part.strip())
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        values = tuple(value)
    else:
        raise TypeError("legacy source_id must be a string or sequence")
    return _normalized_source_ids(values)


def _entity_from_legacy(payload: Any) -> Entity:
    if isinstance(payload, Entity):
        return Entity(
            name=payload.name,
            type=payload.type,
            description=payload.description,
            source_ids=payload.source_ids,
            filepath=payload.filepath,
            metadata=deepcopy(payload.metadata),
        )
    if not isinstance(payload, Mapping):
        raise TypeError("legacy entity must be a mapping")
    return Entity(
        name=payload.get("name", ""),
        type=payload.get("type", "unknown") or "unknown",
        description=payload.get("description", "") or "",
        source_ids=_source_ids_from_legacy(payload),
        filepath=payload.get("filepath") or None,
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
            source_ids=payload.source_ids,
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
        source_ids=_source_ids_from_legacy(payload),
        filepath=payload.get("filepath") or None,
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
            "source_id": "||".join(entity.source_ids) or None,
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
            "source_id": "||".join(relation.source_ids) or None,
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


def _candidate_mapping(payload: Any, candidate_type: str) -> Mapping[str, Any]:
    if is_dataclass(payload) and not isinstance(payload, type):
        return asdict(payload)
    if isinstance(payload, Mapping):
        return payload
    raise TypeError(f"legacy {candidate_type} candidate must be a mapping or dataclass")


def _entity_candidate_from_legacy(payload: Any) -> EntityCandidate:
    candidate = _candidate_mapping(payload, "entity")
    return EntityCandidate(
        name=candidate.get("name", ""),
        type=candidate.get("type"),
        description=candidate.get("description", "") or "",
        score=float(candidate.get("score", 0.0) or 0.0),
        metadata=_legacy_metadata(candidate, _ENTITY_CANDIDATE_LEGACY_FIELDS),
    )


def _relation_candidate_from_legacy(payload: Any) -> RelationCandidate:
    candidate = _candidate_mapping(payload, "relation")
    return RelationCandidate(
        source_name=candidate.get("source_name", ""),
        target_name=candidate.get("target_name", ""),
        description=candidate.get("description", "") or "",
        keywords=candidate.get("keywords", "") or "",
        score=float(candidate.get("score", 0.0) or 0.0),
        metadata=_legacy_metadata(candidate, _RELATION_CANDIDATE_LEGACY_FIELDS),
    )


def _chunk_candidate_from_legacy(payload: Any) -> ChunkCandidate:
    candidate = _candidate_mapping(payload, "chunk")
    document_id = candidate.get("document_id")
    if document_id is None:
        document_id = candidate.get("doc_id", "")
    return ChunkCandidate(
        chunk_uuid=candidate.get("chunk_uuid", ""),
        document_id=document_id,
        filename=candidate.get("filename", "") or "",
        text=candidate.get("text", "") or "",
        score=float(candidate.get("score", 0.0) or 0.0),
        metadata=_legacy_metadata(candidate, _CHUNK_CANDIDATE_LEGACY_FIELDS),
    )


def retrieval_candidates_from_legacy(
    query_plan: QueryPlan,
    payload: Mapping[str, Any],
) -> RetrievalCandidates:
    """Convert PathRAG or LightRAG candidate payloads to shared contracts."""

    if not isinstance(payload, Mapping):
        raise TypeError("legacy retrieval candidates must be a mapping")
    entities = payload.get("entity_matches")
    if entities is None:
        entities = payload.get("entities")
    relations = payload.get("relation_matches")
    if relations is None:
        relations = payload.get("relations")
    chunks = payload.get("chunk_matches")
    if chunks is None:
        chunks = payload.get("chunks")
    return RetrievalCandidates(
        query_plan=query_plan,
        entities=[
            _entity_candidate_from_legacy(item)
            for item in _legacy_list(entities, "entity candidates")
        ],
        relations=[
            _relation_candidate_from_legacy(item)
            for item in _legacy_list(relations, "relation candidates")
        ],
        chunks=[
            _chunk_candidate_from_legacy(item)
            for item in _legacy_list(chunks, "chunk candidates")
        ],
        metadata=_legacy_metadata(payload, _RETRIEVAL_CANDIDATE_LEGACY_FIELDS),
    )


def _entity_candidate_to_legacy(candidate: EntityCandidate) -> Dict[str, Any]:
    payload = deepcopy(candidate.metadata)
    payload.update(
        {
            "name": candidate.name,
            "type": candidate.type,
            "description": candidate.description,
            "score": candidate.score,
        }
    )
    return payload


def _relation_candidate_to_legacy(candidate: RelationCandidate) -> Dict[str, Any]:
    payload = deepcopy(candidate.metadata)
    payload.update(
        {
            "source_name": candidate.source_name,
            "target_name": candidate.target_name,
            "description": candidate.description,
            "keywords": candidate.keywords,
            "score": candidate.score,
        }
    )
    return payload


def _chunk_candidate_to_legacy(candidate: ChunkCandidate) -> Dict[str, Any]:
    payload = deepcopy(candidate.metadata)
    payload.update(
        {
            "chunk_uuid": candidate.chunk_uuid,
            "document_id": candidate.document_id,
            "filename": candidate.filename,
            "text": candidate.text,
            "score": candidate.score,
        }
    )
    return payload


def retrieval_candidates_to_legacy(
    candidates: RetrievalCandidates,
) -> Dict[str, Any]:
    """Convert shared candidates to PathRAG's current dictionary result keys."""

    payload = deepcopy(candidates.metadata)
    payload.update(
        {
            "entity_matches": [
                _entity_candidate_to_legacy(candidate)
                for candidate in candidates.entities
            ],
            "relation_matches": [
                _relation_candidate_to_legacy(candidate)
                for candidate in candidates.relations
            ],
            "chunk_matches": [
                _chunk_candidate_to_legacy(candidate)
                for candidate in candidates.chunks
            ],
        }
    )
    return payload

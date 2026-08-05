from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from schemas import Entity, ExtractionResult, GraphDelta, Relation


DEFAULT_DELIMITER = "||"


@dataclass(frozen=True)
class GraphView:
    """Read-only graph state supplied to the normalization connector."""

    entities: Sequence[Entity] = ()
    relations: Sequence[Relation] = ()

    def __post_init__(self) -> None:
        entities = tuple(self.entities)
        relations = tuple(self.relations)
        if not all(isinstance(entity, Entity) for entity in entities):
            raise TypeError("entities must contain Entity objects")
        if not all(isinstance(relation, Relation) for relation in relations):
            raise TypeError("relations must contain Relation objects")
        if len({entity.name for entity in entities}) != len(entities):
            raise ValueError("GraphView entity names must be unique")
        pairs = [_pair(relation.source_name, relation.target_name) for relation in relations]
        if len(set(pairs)) != len(pairs):
            raise ValueError("GraphView relation pairs must be unique")
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "relations", relations)


def _pair(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((source, target)))


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _text_parts(values: Iterable[str], delimiter: str) -> tuple[str, ...]:
    return _ordered_unique(
        part.strip()
        for value in values
        for part in (value or "").split(delimiter)
        if part.strip()
    )


def _keyword_parts(values: Iterable[str], delimiter: str) -> tuple[str, ...]:
    return _ordered_unique(
        part.strip()
        for value in values
        for segment in (value or "").split(delimiter)
        for part in segment.split(",")
        if part.strip()
    )


def _resolve_type(entities: Sequence[Entity], existing: Entity | None) -> str:
    votes = Counter(
        entity.type.strip().lower()
        for entity in entities
        if entity.type.strip() and entity.type.strip().lower() != "unknown"
    )
    canonical = {
        entity.type.strip().lower(): entity.type.strip()
        for entity in entities
        if entity.type.strip() and entity.type.strip().lower() != "unknown"
    }
    if existing is not None and existing.type.strip().lower() != "unknown":
        key = existing.type.strip().lower()
        votes[key] += 1
        canonical[key] = existing.type.strip()
    if not votes:
        return "unknown"
    highest = max(votes.values())
    contenders = sorted(key for key, count in votes.items() if count == highest)
    if existing is not None and existing.type.strip().lower() in contenders:
        return existing.type.strip()
    return canonical[contenders[0]]


def _merge_metadata(existing, incoming):
    metadata = dict(existing.metadata) if existing is not None else {}
    for item in incoming:
        metadata.update(item.metadata)
    return metadata


def normalize_graph(
    extraction: ExtractionResult,
    existing: GraphView,
    *,
    delimiter: str = DEFAULT_DELIMITER,
) -> GraphDelta:
    """Normalize extracted findings into deterministic graph upserts."""

    if not isinstance(extraction, ExtractionResult):
        raise TypeError("extraction must be an ExtractionResult")
    if not isinstance(existing, GraphView):
        raise TypeError("existing must be a GraphView")
    if not isinstance(delimiter, str) or not delimiter:
        raise ValueError("delimiter must not be empty")

    existing_entities = {entity.name: entity for entity in existing.entities}
    incoming_entities: dict[str, list[Entity]] = {}
    for entity in extraction.entities:
        incoming_entities.setdefault(entity.name, []).append(entity)

    known_names = set(existing_entities) | set(incoming_entities)
    referenced_names = {
        name
        for relation in extraction.relations
        for name in (relation.source_name, relation.target_name)
    }
    for missing_name in referenced_names - known_names:
        incoming_entities[missing_name] = [
            Entity(name=missing_name, type="unknown", description="")
        ]

    normalized_entities = []
    for name in sorted(incoming_entities):
        incoming = incoming_entities[name]
        stored = existing_entities.get(name)
        all_entities = ([stored] if stored is not None else []) + incoming
        descriptions = _text_parts(
            (entity.description for entity in all_entities),
            delimiter,
        )
        source_ids = _ordered_unique(
            source_id
            for entity in all_entities
            for source_id in entity.source_ids
        )
        filepaths = _text_parts(
            (entity.filepath or "" for entity in all_entities),
            delimiter,
        )
        normalized_entities.append(
            Entity(
                name=name,
                type=_resolve_type(incoming, stored),
                description=delimiter.join(descriptions),
                source_ids=source_ids,
                filepath=delimiter.join(filepaths) or None,
                metadata=_merge_metadata(stored, incoming),
            )
        )

    existing_relations = {
        _pair(relation.source_name, relation.target_name): relation
        for relation in existing.relations
    }
    incoming_relations: dict[tuple[str, str], list[Relation]] = {}
    for relation in extraction.relations:
        incoming_relations.setdefault(
            _pair(relation.source_name, relation.target_name),
            [],
        ).append(relation)

    normalized_relations = []
    for pair in sorted(incoming_relations):
        incoming = incoming_relations[pair]
        stored = existing_relations.get(pair)
        all_relations = ([stored] if stored is not None else []) + incoming
        weights = [
            relation.weight
            for relation in all_relations
            if relation.weight is not None
        ]
        normalized_relations.append(
            Relation(
                source_name=pair[0],
                target_name=pair[1],
                description=delimiter.join(
                    _text_parts(
                        (relation.description for relation in all_relations),
                        delimiter,
                    )
                ),
                keywords=delimiter.join(
                    _keyword_parts(
                        (relation.keywords for relation in all_relations),
                        delimiter,
                    )
                ),
                weight=sum(weights) if weights else 0.0,
                source_ids=_ordered_unique(
                    source_id
                    for relation in all_relations
                    for source_id in relation.source_ids
                ),
                filepath=delimiter.join(
                    _text_parts(
                        (relation.filepath or "" for relation in all_relations),
                        delimiter,
                    )
                )
                or None,
                metadata=_merge_metadata(stored, incoming),
            )
        )

    return GraphDelta(
        entities=normalized_entities,
        relations=normalized_relations,
        metadata={"delimiter": delimiter},
    )

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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

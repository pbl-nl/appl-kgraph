from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from schemas import AnswerResult, RetrievedContext


ConversationHistory = Sequence[tuple[str, str]]


@dataclass(frozen=True)
class GeneratedAnswer:
    """Raw answer-generator output before it is joined to retrieval context."""

    text: str
    model: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("generated answer text must be a string")
        if not isinstance(self.model, Mapping):
            raise TypeError("generated answer model must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("generated answer metadata must be a mapping")
        object.__setattr__(self, "model", deepcopy(dict(self.model)))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


class AnswerGenerator(Protocol):
    """Synchronous grounded-answer implementation."""

    def generate(
        self,
        query: str,
        context: RetrievedContext,
        history: ConversationHistory,
    ) -> GeneratedAnswer:
        ...


class AsyncAnswerGenerator(Protocol):
    """Asynchronous grounded-answer implementation."""

    def generate(
        self,
        query: str,
        context: RetrievedContext,
        history: ConversationHistory,
    ) -> Awaitable[GeneratedAnswer]:
        ...


class ChatAuditRecorder(Protocol):
    """Post-answer chatbot reproducibility side effect."""

    def record(self, result: AnswerResult, project_paths: Any) -> Path | None:
        ...


@dataclass(frozen=True)
class TextAnswerGenerator:
    """Adapter for a synchronous text-returning model call."""

    generate_text: Callable[[str, RetrievedContext, ConversationHistory], str]
    model: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def generate(
        self,
        query: str,
        context: RetrievedContext,
        history: ConversationHistory,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            text=self.generate_text(query, context, history),
            model=self.model,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class AsyncTextAnswerGenerator:
    """Adapter for an asynchronous text-returning model call."""

    generate_text: Callable[
        [str, RetrievedContext, ConversationHistory],
        Awaitable[str],
    ]
    model: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    async def generate(
        self,
        query: str,
        context: RetrievedContext,
        history: ConversationHistory,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            text=await self.generate_text(query, context, history),
            model=self.model,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class LegacyAuditRecorder:
    """Adapter for query_logging.write_audit_log()."""

    write_audit_log: Callable[..., Path | None]

    def record(self, result: AnswerResult, project_paths: Any) -> Path | None:
        plan = result.context.query_plan
        return self.write_audit_log(
            project_paths=project_paths,
            retriever_name=plan.strategy,
            payload={
                "question": plan.query,
                "answer": result.answer,
                "conversation_history": list(plan.history),
                "retrieval_metadata": {
                    "strategy": plan.strategy,
                    "mode": plan.mode,
                    "settings": plan.settings,
                    "plan_metadata": plan.metadata,
                    "context_metadata": result.context.metadata,
                },
                "model": result.model,
                "high_level_keywords": plan.high_level_keywords,
                "low_level_keywords": plan.low_level_keywords,
                "seed_entities": list(plan.seed_entities),
                "context_windows": [
                    asdict(window) for window in result.context.context_windows
                ],
                "retrieved_entities": [
                    asdict(entity) for entity in result.context.entities
                ],
                "retrieved_relationships": [
                    asdict(relation) for relation in result.context.relations
                ],
                "retrieved_chunks": [
                    asdict(chunk) for chunk in result.context.chunks
                ],
                "answer_metadata": result.metadata,
            },
        )


def generate_answer(
    query: str,
    context: RetrievedContext,
    history: ConversationHistory,
    generator: AnswerGenerator,
) -> AnswerResult:
    """Generate a grounded answer through a synchronous typed connector."""

    normalized_history = _validate_inputs(query, context, history)
    generated = generator.generate(query, context, normalized_history)
    return _answer_result(generated, context)


async def generate_answer_async(
    query: str,
    context: RetrievedContext,
    history: ConversationHistory,
    generator: AsyncAnswerGenerator,
) -> AnswerResult:
    """Generate a grounded answer through an asynchronous typed connector."""

    normalized_history = _validate_inputs(query, context, history)
    generated = await generator.generate(query, context, normalized_history)
    return _answer_result(generated, context)


def record_chat_audit(
    result: AnswerResult,
    project_paths: Any,
    recorder: ChatAuditRecorder | None = None,
) -> Path | None:
    """Record one completed answer as a separate post-answer side effect."""

    if not isinstance(result, AnswerResult):
        raise TypeError("result must be an AnswerResult")
    if recorder is None:
        from query_logging import write_audit_log

        recorder = LegacyAuditRecorder(write_audit_log)
    target = recorder.record(result, project_paths)
    if target is not None and not isinstance(target, Path):
        raise TypeError("audit recorder must return Path or None")
    return target


def _validate_inputs(query, context, history):
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(context, RetrievedContext):
        raise TypeError("context must be a RetrievedContext")
    if context.query_plan.query != query:
        raise ValueError("query must match the retrieved context query plan")
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("history must be a sequence")
    normalized_history = tuple(tuple(turn) for turn in history)
    if normalized_history != tuple(context.query_plan.history):
        raise ValueError("history must match the retrieved context query plan")
    return normalized_history


def _answer_result(generated, context):
    if not isinstance(generated, GeneratedAnswer):
        raise TypeError("generator must return a GeneratedAnswer")
    return AnswerResult(
        answer=generated.text,
        context=context,
        model=dict(generated.model),
        metadata=dict(generated.metadata),
    )

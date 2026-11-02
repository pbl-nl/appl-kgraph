import asyncio
from pathlib import Path

import pytest

from answer_generation import (
    AsyncTextAnswerGenerator,
    GeneratedAnswer,
    LegacyAuditRecorder,
    TextAnswerGenerator,
    generate_answer,
    generate_answer_async,
    record_chat_audit,
)
from schemas import AnswerResult, ContextWindow, QueryPlan, RetrievedContext


def _context():
    plan = QueryPlan(
        query="Question",
        history=(("user", "Earlier"),),
        high_level_keywords=["theme"],
        low_level_keywords=["Alpha"],
        strategy="lightrag",
        mode="hybrid",
        settings={"chunk_top_k": 5},
    )
    return RetrievedContext(
        query_plan=plan,
        context_windows=[
            ContextWindow(
                label="chunk::one",
                text="Evidence",
                score=0.9,
                source_refs=("chunk-1",),
            )
        ],
        metadata={"token_count": 10},
    )


def test_generate_answer_returns_answer_with_context_and_model_metadata():
    context = _context()
    calls = []

    def generate_text(query, received_context, history):
        calls.append((query, received_context, history))
        return "Grounded answer"

    result = generate_answer(
        "Question",
        context,
        [("user", "Earlier")],
        TextAnswerGenerator(
            generate_text,
            model={"provider": "fake", "model_name": "model-1"},
            metadata={"prompt_version": "v1"},
        ),
    )

    assert result == AnswerResult(
        answer="Grounded answer",
        context=context,
        model={"provider": "fake", "model_name": "model-1"},
        metadata={"prompt_version": "v1"},
    )
    assert calls == [("Question", context, (("user", "Earlier"),))]


def test_async_generate_answer_uses_same_result_contract():
    context = _context()

    async def generate_text(query, received_context, history):
        return "Async grounded answer"

    result = asyncio.run(
        generate_answer_async(
            "Question",
            context,
            [("user", "Earlier")],
            AsyncTextAnswerGenerator(generate_text, model={"provider": "fake"}),
        )
    )

    assert result.answer == "Async grounded answer"
    assert result.model == {"provider": "fake"}
    assert result.context is context


def test_audit_adapter_emits_reproducibility_payload():
    context = _context()
    result = AnswerResult(
        answer="Grounded answer",
        context=context,
        model={"provider": "fake"},
        metadata={"prompt_version": "v1"},
    )
    calls = []
    expected_path = Path("audit.json")

    def write_audit_log(**kwargs):
        calls.append(kwargs)
        return expected_path

    target = record_chat_audit(
        result,
        "project-paths",
        LegacyAuditRecorder(write_audit_log),
    )

    assert target == expected_path
    assert calls[0]["project_paths"] == "project-paths"
    assert calls[0]["retriever_name"] == "lightrag"
    payload = calls[0]["payload"]
    assert payload["question"] == "Question"
    assert payload["conversation_history"] == [("user", "Earlier")]
    assert payload["answer"] == "Grounded answer"
    assert payload["model"] == {"provider": "fake"}
    assert payload["retrieval_metadata"]["mode"] == "hybrid"
    assert payload["context_windows"][0]["source_refs"] == ("chunk-1",)
    assert payload["answer_metadata"] == {"prompt_version": "v1"}


def test_audit_connector_allows_disabled_recorder_result():
    class DisabledRecorder:
        def record(self, result, project_paths):
            return None

    result = AnswerResult(answer="Answer", context=_context())

    assert record_chat_audit(result, object(), DisabledRecorder()) is None


def test_answer_connector_rejects_changed_query_or_history():
    context = _context()
    generator = TextAnswerGenerator(lambda *args: "Answer")

    with pytest.raises(ValueError, match="query"):
        generate_answer("Different", context, [("user", "Earlier")], generator)
    with pytest.raises(ValueError, match="history"):
        generate_answer("Question", context, [], generator)


def test_answer_connector_rejects_invalid_generator_output():
    class InvalidGenerator:
        def generate(self, query, context, history):
            return "plain string"

    with pytest.raises(TypeError, match="GeneratedAnswer"):
        generate_answer(
            "Question",
            _context(),
            [("user", "Earlier")],
            InvalidGenerator(),
        )


def test_audit_connector_rejects_invalid_output_path():
    class InvalidRecorder:
        def record(self, result, project_paths):
            return "audit.json"

    with pytest.raises(TypeError, match="Path or None"):
        record_chat_audit(
            AnswerResult(answer="Answer", context=_context()),
            object(),
            InvalidRecorder(),
        )


def test_generated_answer_owns_metadata():
    metadata = {"nested": {"values": ["original"]}}
    generated = GeneratedAnswer(text="Answer", metadata=metadata)

    metadata["nested"]["values"].append("changed")

    assert generated.metadata == {"nested": {"values": ["original"]}}

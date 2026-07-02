"""
DocBench-style evaluation of PathRAG and LightRAG.

Usage:
    python evaluate.py <benchmark_root> [--retriever {pathrag,lightrag,both}]

benchmark_root must contain numbered subfolders, each with:
  - one PDF file, the document to query
  - one JSONL file, questions in DocBench format
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from ingestion import ingest_paths
from lightrag import LightRAG
from llm import Chat
from pathrag import PathRAG
from project_paths import resolve_project_paths
from query_logging import build_audit_settings_snapshot
from settings import settings


_JUDGE_SYSTEM = (
    "You are an evaluation judge for a document-question-answering benchmark. "
    "Reply with only 'yes' or 'no'."
)
_JUDGE_PROMPT = """\
Decide whether the generated answer correctly answers the question given the reference answer.

Question: {question}
Reference answer: {reference}
Generated answer: {generated}

Is the generated answer correct? Reply with only "yes" or "no"."""


def _emit(message: str, *, force: bool = False) -> None:
    if force or settings.logging.verbosity_enabled:
        print(message)


def _write_audit_jsonl(log_path: Path, entry: dict) -> None:
    if not settings.logging.audit_enabled:
        return
    entry["ts"] = datetime.datetime.now().isoformat()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _llm_judge(
    chat: Chat,
    question: str,
    reference: str,
    generated: str,
    log_path: Path,
    subfolder: str,
    question_index: int,
) -> Optional[bool]:
    prompt = _JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        generated=generated,
    )
    t0 = time.monotonic()
    try:
        verdict = chat.generate(prompt, system=_JUDGE_SYSTEM, temperature=0.0, max_tokens=4)
        result = verdict.strip().lower().startswith("y")
    except Exception as exc:
        _emit(f"    [judge error] {exc}", force=True)
        _write_audit_jsonl(
            log_path,
            {
                "call": "judge",
                "subfolder": subfolder,
                "question_index": question_index,
                "question": question,
                "reference": reference,
                "generated": generated,
                "response": None,
                "verdict": None,
                "error": str(exc),
                "duration_ms": int((time.monotonic() - t0) * 1000),
            },
        )
        return None
    _write_audit_jsonl(
        log_path,
        {
            "call": "judge",
            "subfolder": subfolder,
            "question_index": question_index,
            "question": question,
            "reference": reference,
            "generated": generated,
            "response": verdict.strip(),
            "verdict": result,
            "duration_ms": int((time.monotonic() - t0) * 1000),
        },
    )
    return result


def _find_file(folder: Path, suffix: str) -> Optional[Path]:
    matches = list(folder.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _load_questions(jsonl_path: Path) -> List[dict]:
    rows = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def _run_subfolder(
    subfolder: Path,
    retriever: str,
    chat: Chat,
    log_path: Path,
) -> List[dict]:
    pdf = _find_file(subfolder, ".pdf")
    jsonl = _find_file(subfolder, ".jsonl")

    if pdf is None:
        _emit(f"  [{subfolder.name}] No PDF found - skipping.")
        return []
    if jsonl is None:
        _emit(f"  [{subfolder.name}] No JSONL found - skipping.")
        return []

    _emit(f"\n[{subfolder.name}] Ingesting {pdf.name} ...")
    ingest_paths([pdf], documents_root=subfolder)

    project_paths = resolve_project_paths(subfolder)
    pathrag = PathRAG(project_paths=project_paths, system_prompt="") if retriever in ("pathrag", "both") else None
    lightrag = LightRAG(project_paths=project_paths, system_prompt="") if retriever in ("lightrag", "both") else None

    questions = _load_questions(jsonl)
    _emit(f"[{subfolder.name}] {len(questions)} questions ...")

    results: List[dict] = []
    for idx, qa in enumerate(questions, start=1):
        question = qa["question"]
        reference = qa["answer"]
        row: dict = {
            "subfolder": subfolder.name,
            "question_index": idx,
            "question": question,
            "type": qa.get("type", ""),
            "reference_answer": reference,
            "evidence": qa.get("evidence", ""),
        }

        _emit(f"  Q{idx}/{len(questions)}: {question[:70]}...")

        if pathrag is not None:
            t0 = time.monotonic()
            try:
                result = await pathrag.aretrieve(question)
                row["pathrag_answer"] = result.answer
                error = None
            except Exception as exc:
                row["pathrag_answer"] = f"ERROR: {exc}"
                error = str(exc)
            _write_audit_jsonl(
                log_path,
                {
                    "call": "retrieval",
                    "retriever": "pathrag",
                    "subfolder": subfolder.name,
                    "question_index": idx,
                    "question": question,
                    "answer": row["pathrag_answer"],
                    "error": error,
                    "duration_ms": int((time.monotonic() - t0) * 1000),
                },
            )
            row["pathrag_correct"] = _llm_judge(
                chat,
                question,
                reference,
                row["pathrag_answer"],
                log_path,
                subfolder.name,
                idx,
            )
            mark = "ok" if row["pathrag_correct"] else ("wrong" if row["pathrag_correct"] is False else "judge-err")
            _emit(f"    PathRAG  -> {mark}")

        if lightrag is not None:
            t0 = time.monotonic()
            try:
                result = await lightrag.aretrieve(question)
                row["lightrag_answer"] = result.answer
                error = None
            except Exception as exc:
                row["lightrag_answer"] = f"ERROR: {exc}"
                error = str(exc)
            _write_audit_jsonl(
                log_path,
                {
                    "call": "retrieval",
                    "retriever": "lightrag",
                    "subfolder": subfolder.name,
                    "question_index": idx,
                    "question": question,
                    "answer": row["lightrag_answer"],
                    "error": error,
                    "duration_ms": int((time.monotonic() - t0) * 1000),
                },
            )
            row["lightrag_correct"] = _llm_judge(
                chat,
                question,
                reference,
                row["lightrag_answer"],
                log_path,
                subfolder.name,
                idx,
            )
            mark = "ok" if row["lightrag_correct"] else ("wrong" if row["lightrag_correct"] is False else "judge-err")
            _emit(f"    LightRAG -> {mark}")

        results.append(row)

    return results


def _print_summary(rows: List[dict], retriever: str) -> None:
    _emit("\n" + "=" * 60, force=True)
    _emit("EVALUATION SUMMARY", force=True)
    _emit("=" * 60, force=True)

    retrievers = [r for r in ("pathrag", "lightrag") if retriever in (r, "both")]
    by_type: dict[str, dict[str, dict]] = {}
    totals: dict[str, dict] = {r: {"correct": 0, "total": 0} for r in retrievers}

    for row in rows:
        q_type = row.get("type") or "unknown"
        if q_type not in by_type:
            by_type[q_type] = {r: {"correct": 0, "total": 0} for r in retrievers}
        for r in retrievers:
            val = row.get(f"{r}_correct")
            if val is not None:
                by_type[q_type][r]["total"] += 1
                totals[r]["total"] += 1
                if val:
                    by_type[q_type][r]["correct"] += 1
                    totals[r]["correct"] += 1

    for q_type, counts in sorted(by_type.items()):
        _emit(f"\n  {q_type}", force=True)
        for r in retrievers:
            n = counts[r]["total"]
            if n:
                pct = 100 * counts[r]["correct"] / n
                _emit(f"    {r.upper():<10s}: {counts[r]['correct']}/{n}  ({pct:.1f}%)", force=True)

    _emit("\n  OVERALL", force=True)
    for r in retrievers:
        n = totals[r]["total"]
        if n:
            pct = 100 * totals[r]["correct"] / n
            _emit(f"    {r.upper():<10s}: {totals[r]['correct']}/{n}  ({pct:.1f}%)", force=True)


async def _main_async(benchmark_root: Path, retriever: str) -> None:
    input_dir = benchmark_root / "input"
    output_dir = benchmark_root / "output"

    if not input_dir.is_dir():
        _emit(f"ERROR: input folder not found at {input_dir}", force=True)
        sys.exit(1)

    if settings.logging.audit_enabled:
        output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"eval_results_{timestamp}.jsonl"
    settings_path = output_dir / f"eval_settings_{timestamp}.json"
    log_path = output_dir / f"eval_llm_log_{timestamp}.jsonl"

    subfolders = sorted(
        [d for d in input_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not subfolders:
        _emit(f"No subfolders found in {input_dir}", force=True)
        sys.exit(1)

    run_settings = {
        "timestamp": timestamp,
        "benchmark_root": str(benchmark_root),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "retriever": retriever,
        "subfolders": [d.name for d in subfolders],
        "settings": build_audit_settings_snapshot(),
    }

    if settings.logging.audit_enabled:
        settings_path.write_text(json.dumps(run_settings, indent=2, ensure_ascii=False), encoding="utf-8")

    _emit(f"Input      : {input_dir}")
    if settings.logging.audit_enabled:
        _emit(f"Output     : {results_path}")
        _emit(f"LLM log    : {log_path}")
        _emit(f"Settings   : {settings_path}")
    else:
        _emit("Audit logs : disabled")
    _emit(f"Retriever  : {retriever}")
    _emit(f"Subfolders : {[d.name for d in subfolders]}")

    chat = Chat()
    all_rows: List[dict] = []

    for subfolder in subfolders:
        rows = await _run_subfolder(subfolder, retriever, chat, log_path)
        all_rows.extend(rows)

        if settings.logging.audit_enabled:
            with results_path.open("w", encoding="utf-8") as fh:
                for row in all_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if settings.logging.audit_enabled:
        _emit(f"\nResults written to {results_path}  ({len(all_rows)} rows)", force=True)
    else:
        _emit(f"\nAudit disabled; evaluated {len(all_rows)} rows without writing result files.", force=True)
    _print_summary(all_rows, retriever)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PathRAG / LightRAG on a DocBench-style benchmark."
    )
    default_benchmark = str(Path(__file__).parent.parent / "benchmark")
    parser.add_argument(
        "benchmark_root",
        nargs="?",
        default=default_benchmark,
        help="Benchmark folder containing an 'input' subfolder (default: <repo>/benchmark)",
    )
    parser.add_argument(
        "--retriever",
        choices=["pathrag", "lightrag", "both"],
        default="both",
        help="Which retriever(s) to evaluate (default: both)",
    )
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    if not benchmark_root.is_dir():
        _emit(f"ERROR: {benchmark_root} is not a directory.", force=True)
        sys.exit(1)

    asyncio.run(_main_async(benchmark_root, args.retriever))


if __name__ == "__main__":
    main()

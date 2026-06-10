"""
DocBench-style evaluation of PathRAG and LightRAG.

Usage:
    python evaluate.py <benchmark_root> [--retriever {pathrag,lightrag,both}] [--output results.jsonl]

benchmark_root must contain numbered subfolders (1, 2, ..., 10), each with:
  - one PDF file  (the document to query)
  - one JSONL file (questions in DocBench format)
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
from pathrag import PathRAG
from project_paths import resolve_project_paths
from llm import Chat


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


def _log(log_path: Path, entry: dict) -> None:
    entry["ts"] = datetime.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as fh:
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
        print(f"    [judge error] {exc}")
        _log(log_path, {
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
        })
        return None
    _log(log_path, {
        "call": "judge",
        "subfolder": subfolder,
        "question_index": question_index,
        "question": question,
        "reference": reference,
        "generated": generated,
        "response": verdict.strip(),
        "verdict": result,
        "duration_ms": int((time.monotonic() - t0) * 1000),
    })
    return result


def _find_file(folder: Path, suffix: str) -> Optional[Path]:
    matches = list(folder.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _load_questions(jsonl_path: Path) -> List[dict]:
    rows = []
    with open(jsonl_path, encoding="utf-8") as fh:
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
        print(f"  [{subfolder.name}] No PDF found — skipping.")
        return []
    if jsonl is None:
        print(f"  [{subfolder.name}] No JSONL found — skipping.")
        return []

    print(f"\n[{subfolder.name}] Ingesting {pdf.name} ...")
    ingest_paths([pdf], documents_root=subfolder)

    project_paths = resolve_project_paths(subfolder)
    pathrag = PathRAG(project_paths=project_paths, system_prompt="") if retriever in ("pathrag", "both") else None
    lightrag = LightRAG(project_paths=project_paths, system_prompt="") if retriever in ("lightrag", "both") else None

    questions = _load_questions(jsonl)
    print(f"[{subfolder.name}] {len(questions)} questions ...")

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

        print(f"  Q{idx}/{len(questions)}: {question[:70]}...")

        if pathrag is not None:
            t0 = time.monotonic()
            try:
                result = await pathrag.aretrieve(question)
                row["pathrag_answer"] = result.answer
                error = None
            except Exception as exc:
                row["pathrag_answer"] = f"ERROR: {exc}"
                error = str(exc)
            _log(log_path, {
                "call": "retrieval",
                "retriever": "pathrag",
                "subfolder": subfolder.name,
                "question_index": idx,
                "question": question,
                "answer": row["pathrag_answer"],
                "error": error,
                "duration_ms": int((time.monotonic() - t0) * 1000),
            })
            row["pathrag_correct"] = _llm_judge(
                chat, question, reference, row["pathrag_answer"],
                log_path, subfolder.name, idx,
            )
            mark = "ok" if row["pathrag_correct"] else ("wrong" if row["pathrag_correct"] is False else "judge-err")
            print(f"    PathRAG  → {mark}")

        if lightrag is not None:
            t0 = time.monotonic()
            try:
                result = await lightrag.aretrieve(question)
                row["lightrag_answer"] = result.answer
                error = None
            except Exception as exc:
                row["lightrag_answer"] = f"ERROR: {exc}"
                error = str(exc)
            _log(log_path, {
                "call": "retrieval",
                "retriever": "lightrag",
                "subfolder": subfolder.name,
                "question_index": idx,
                "question": question,
                "answer": row["lightrag_answer"],
                "error": error,
                "duration_ms": int((time.monotonic() - t0) * 1000),
            })
            row["lightrag_correct"] = _llm_judge(
                chat, question, reference, row["lightrag_answer"],
                log_path, subfolder.name, idx,
            )
            mark = "ok" if row["lightrag_correct"] else ("wrong" if row["lightrag_correct"] is False else "judge-err")
            print(f"    LightRAG → {mark}")

        results.append(row)

    return results


def _print_summary(rows: List[dict], retriever: str) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

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
        print(f"\n  {q_type}")
        for r in retrievers:
            n = counts[r]["total"]
            if n:
                pct = 100 * counts[r]["correct"] / n
                print(f"    {r.upper():<10s}: {counts[r]['correct']}/{n}  ({pct:.1f}%)")

    print(f"\n  OVERALL")
    for r in retrievers:
        n = totals[r]["total"]
        if n:
            pct = 100 * totals[r]["correct"] / n
            print(f"    {r.upper():<10s}: {totals[r]['correct']}/{n}  ({pct:.1f}%)")


async def _main_async(benchmark_root: Path, retriever: str) -> None:
    import datetime
    from settings import settings as app_settings

    input_dir = benchmark_root / "input"
    output_dir = benchmark_root / "output"

    if not input_dir.is_dir():
        print(f"ERROR: input folder not found at {input_dir}")
        sys.exit(1)

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
        print(f"No subfolders found in {input_dir}")
        sys.exit(1)

    run_settings = {
        "timestamp": timestamp,
        "benchmark_root": str(benchmark_root),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "retriever": retriever,
        "subfolders": [d.name for d in subfolders],
        "llm_provider": app_settings.provider.provider,
        "llm_model": (
            app_settings.provider.openai_llm_model
            if app_settings.provider.provider == "openai"
            else app_settings.provider.azure_llm_deployment
        ),
        "retrieval": {
            "entity_top_k": app_settings.retrieval.entity_top_k,
            "relation_top_k": app_settings.retrieval.relation_top_k,
            "chunk_top_k": app_settings.retrieval.chunk_top_k,
            "light_mode": app_settings.retrieval.light_mode,
            "path_max_depth": app_settings.retrieval.path_max_depth,
            "enable_rerank": app_settings.retrieval.enable_rerank,
        },
    }

    settings_path.write_text(json.dumps(run_settings, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input      : {input_dir}")
    print(f"Output     : {results_path}")
    print(f"LLM log    : {log_path}")
    print(f"Settings   : {settings_path}")
    print(f"Retriever  : {retriever}")
    print(f"Subfolders : {[d.name for d in subfolders]}")

    chat = Chat()
    all_rows: List[dict] = []

    for subfolder in subfolders:
        rows = await _run_subfolder(subfolder, retriever, chat, log_path)
        all_rows.extend(rows)

        # flush after each subfolder so partial results survive interruption
        with open(results_path, "w", encoding="utf-8") as fh:
            for row in all_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nResults written to {results_path}  ({len(all_rows)} rows)")
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
        print(f"ERROR: {benchmark_root} is not a directory.")
        sys.exit(1)

    asyncio.run(_main_async(benchmark_root, args.retriever))


if __name__ == "__main__":
    main()

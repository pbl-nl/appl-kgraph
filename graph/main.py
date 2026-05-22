from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple

from ingestion import ingest_paths
from lightrag import LightRAG, RetrievalResult
from pathrag import PathRAG, render_full_context
from project_paths import list_document_paths, resolve_project_paths


async def ask_with_pathrag(
    question: str,
    *,
    documents_root: Path,
    verbose: bool = False,
    conversation_history: Optional[List[Tuple[str, str]]] = None,
) -> None:
    project_paths = resolve_project_paths(documents_root)
    rag = PathRAG(project_paths=project_paths, system_prompt="")
    result = await rag.aretrieve(question, conversation_history=conversation_history)
    print("Answer:\n", result.answer)
    if verbose:
        print(render_full_context(result))
    elif result.context_windows:
        for window in result.context_windows:
            print(f"\n[{window.label}] score={window.score:.2f}\n{window.text}")


async def ask_with_lightrag(
    question: str,
    *,
    documents_root: Path,
    verbose: bool = False,
    history: Optional[List[Tuple[str, str]]] = None,
) -> RetrievalResult:
    project_paths = resolve_project_paths(documents_root)
    rag = LightRAG(project_paths=project_paths, system_prompt="")
    result = await rag.aretrieve(question, conversation_history=history)
    print("Answer:\n", result.answer)
    if verbose:
        from lightrag import render_full_context as render_lightrag_context

        print(render_lightrag_context(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a document folder and query the project-scoped RAG stores.")
    parser.add_argument("documents_root", nargs="?", default="docs", help="Folder containing documents to ingest and query.")
    args = parser.parse_args()

    documents_root = Path(args.documents_root).expanduser().resolve()
    paths = list_document_paths(documents_root)
    if not paths:
        print("No files to ingest.")
        return

    ingest_paths(paths, documents_root=documents_root)

    conversation_history: List[Tuple[str, str]] = []
    query = input("Enter your question: ")
    while query not in ("exit", "quit"):
        print("\n--- PathRAG Response ---\n")
        asyncio.run(
            ask_with_pathrag(
                query,
                documents_root=documents_root,
                verbose=True,
                conversation_history=conversation_history,
            )
        )
        print("\n---\n")
        print("\n--- LightRAG Response ---\n")
        result = asyncio.run(
            ask_with_lightrag(
                query,
                documents_root=documents_root,
                verbose=True,
                history=conversation_history,
            )
        )
        conversation_history.append(("user", query))
        conversation_history.append(("assistant", result.answer))
        print("\n---\n")
        query = input("Enter your next question: ")


if __name__ == "__main__":
    main()

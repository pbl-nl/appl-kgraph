import asyncio
from pathlib import Path

from ingestion import ingest_paths
from pathrag import PathRAG
from fileparser import FileParser
from pathrag import render_full_context

async def ask_with_pathrag(question: str, verbose: bool = False) -> None:
    """
    Asks a question using PathRAG retrieval and prints the answer with context.

    Args:
        question (str): The question to ask.
        verbose (bool, optional): If True, displays full context details. Defaults to False.

    Returns:
        None
    """
    rag = PathRAG(
        system_prompt=""
    )
    result = await rag.aretrieve(question)
    print("Answer:\n", result.answer)

# Always show context windows (your current behavior)
    print(render_full_context(result) if verbose else "")
    if not verbose:
        # Keep your existing brief printout (only windows)
        for window in result.context_windows:
            print(f"\n[{window.label}] score={window.score:.2f}\n{window.text}")

def main():
    """
    Main entry point for document ingestion and Q&A demonstration.

    Ingests documents from the 'docs' directory and runs a sample PathRAG query.
    """
    root = Path('docs')
    paths = FileParser(root).filepaths
    if not paths:
        print("No files to ingest.")
        return
    ingest_paths(paths)
    asyncio.run(ask_with_pathrag("Who are the authors of LayoutParser and do they overlap any of the other articles?", verbose=True))

if __name__ == "__main__":
    main()

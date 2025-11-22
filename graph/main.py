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

    print(render_full_context(result) if verbose else "")
    if not verbose:
        for window in result.context_windows:
            print(f"\n[{window.label}] score={window.score:.2f}\n{window.text}")


async def ask_with_lightrag(question: str, verbose: bool = False, history: list = None) -> object:
    """
    Asks a question using LightRAG retrieval and prints the answer with context.

    Args:
        question (str): The question to ask.
        verbose (bool, optional): If True, displays full context details. Defaults to False.
        history (list, optional): Conversation history. Defaults to None.

    Returns:
        Result object containing the answer and context.
    """
    from lightrag import LightRAG
    from lightrag import render_full_context
    rag = LightRAG(
        system_prompt=""
    )
    result = await rag.aretrieve(question, conversation_history=history)
    print("Answer:\n", result.answer)

    print(render_full_context(result) if verbose else "")
    
    return result

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
    # query = "Who are the authors of LayoutParser and do they overlap any of the other articles?"
    query = input("Enter your question: ")
    conversation_history = []
    while query not in ("exit", "quit"):
        print("\n--- PathRAG Response ---\n")
        asyncio.run(ask_with_pathrag(query, verbose=True))
        print("\n---\n")
        print("\n--- LightRAG Response ---\n")
        result = asyncio.run(ask_with_lightrag(query, verbose=True, history=conversation_history))
        conversation_history.append(("user", query))
        conversation_history.append(("assistant", result.answer))
        print("\n---\n")
        query = input("Enter your question: ")

if __name__ == "__main__":
    main()

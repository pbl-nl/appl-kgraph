# Pull Discussion Summary

Comparison basis:
- Local working tree on `main` at `bf2d9e2`
- Upstream `origin/main` at `d4a40be`
- Date: 2026-03-25

This note summarizes the main code changes relative to the current upstream `main`, why they were made, and what still needs discussion before deciding whether to pull this work.

## What Changed at a High Level

- The codebase was moved from mostly global storage assumptions to a project-aware model.
- Ingestion, retrieval, logs, graph snapshots, and audit outputs can now be scoped to a selected document folder.
- The Gradio app was updated to work with that project-aware model instead of assuming one global storage location.
- Graph pickle support was restored and split into two roles:
  - `kg.pkl` for the editable working graph in the UI
  - `kg_retrieval.pkl` for the retrieval snapshot built from canonical storage
- Structured JSON logging was added for question-answer interactions.
- Extraction was extended to use language information from documents and chunks.
- Clean-ingestion stability on this Windows/Python 3.12 environment was improved by keeping retrieval operational even when native Chroma vector operations are unstable.

## Project-Aware Ingestion and Storage

The main architectural change is that a selected document folder is now treated as a project root for artifacts.

Indented example:

```text
C:\path\to\documents\
  report.pdf
  notes.txt
  .appl-kgraph\
    storage\
      documents.sqlite
      chunks.sqlite
      graph.sqlite
      chroma_chunks\
      chroma_entities\
      chroma_relations\
    knowledge_graph\
      kg.pkl
      kg_retrieval.pkl
    logs\
      ingestion.log
      pathrag.log
      lightrag.log
      qa\
        20260325T....json
    audits\
      extraction\
        report.audit.json
```

Why this matters:
- Separate corpora no longer have to share one global storage directory.
- The UI can later select a project cleanly because the storage layout already supports it.
- Logs, graph artifacts, and audits stay attached to the document set that produced them.

## Most Important Code Changes and Reasons

- `graph/project_paths.py`
  - Added a single resolver for turning a document folder into project-local storage, logs, audits, and knowledge-graph paths.
  - This is the main seam that makes per-project isolation work.

- `graph/ingestion.py`
  - Ingestion now accepts a project/document root and writes to project-local storage.
  - Added structured progress messages for the UI during ingestion.
  - Added retrieval-snapshot writing after ingestion.
  - Added hooks for audit output and more explicit per-file processing stages.

- `graph/app.py`
  - The Gradio app now follows the project-aware storage model.
  - Ingestion status is streamed live in the sidebar instead of only showing a final result.
  - The knowledge graph rendering was fixed to use an iframe wrapper so the PyVis graph actually displays.
  - The earlier graph editing tools were restored:
    - save working graph
    - load saved graph
    - edit node
    - merge nodes
  - The app now distinguishes between:
    - the editable working graph pickle
    - the retrieval graph snapshot

- `graph/graph_pickle.py`
  - Added helper functions for loading, saving, and rebuilding graph pickles from canonical storage.
  - This keeps pickle logic out of the UI and retriever code.

- `graph/pathrag.py` and `graph/lightrag.py`
  - Retrieval now loads the graph from the project-specific retrieval snapshot when available.
  - Both retrievers now log question-answer interactions to structured JSON files.
  - Retrieval was adjusted to work with the project-aware storage model instead of a single global storage root.

- `graph/query_logging.py`
  - Added JSON query logging for QA sessions.
  - This is useful for debugging, auditability, and reviewing retrieval context after the fact.

- `graph/fileparser.py`
  - Text-file parsing was changed to avoid depending on `chardet` for normal `.txt` ingestion.
  - This fixed a real failure seen while ingesting `docs2\UDHR_first_article_all.txt`.

- `graph/extractor.py`, `graph/prompts.py`, and `graph/utils.py`
  - Extraction prompting now has better language handling and cleaner prompt wiring.
  - Utility and prompt behavior is more centralized than before.

- `graph/db_storage.py`
  - Storage was extended to support project-aware paths cleanly.
  - A SQL-backed similarity fallback was added for environments where native Chroma vector operations are unstable.
  - On the current Windows/Python 3.12 setup, that fallback is now also used to avoid clean-ingestion crashes during Chroma vector writes.

- `graph/chunker.py`
  - Overlap behavior was tightened so it does not keep compounding prior overlap unintentionally.

- `graph/settings.py`
  - Settings were expanded for project layout, logging split, and extraction toggles.
  - This is one of the larger refactors in the diff and is functionally useful, but also one of the noisier files to review.

- `graph/logging_utils.py`
  - Added centralized file logger setup so ingestion and retrieval logs can be scoped per project.

- Tests
  - Added coverage for project-path resolution, graph pickle helpers, and chunk overlap behavior.
  - Repaired the existing PathRAG storage adapter test.

## New Files Added

- `graph/project_paths.py`
- `graph/query_logging.py`
- `graph/graph_pickle.py`
- `graph/logging_utils.py`
- `test/test_project_paths.py`
- `test/test_graph_pickle.py`
- `test/test_chunker.py`

## Local-Only Artifacts That Should Not Be Part of a Pull

These exist in the working tree but look like local runtime or test artifacts, not pull candidates:

- `.venv2/`
- `docs/.appl-kgraph/`
- `docs2/`

One tracked non-code change also needs discussion:

- `docs_store/UDHR_first_article_all.txt` is currently deleted in the diff
  - This does not look like part of the architectural work and should be reviewed before any pull request is prepared.

## Why the Changes Are Good

- The system is easier to reason about because project selection now maps directly to storage layout.
- The UI and backend are more aligned than before.
- Retrieval and ingestion leave behind much better operational traces through logs, QA JSON files, and graph snapshots.
- The graph editing workflow was preserved instead of being lost during the project-scoping changes.
- The codebase is better positioned for future UI project selection without another storage redesign.

## Open Issues and Discussion Points

- Native Chroma stability on Windows/Python 3.12
  - On this machine, native Chroma query and native Chroma vector writes both proved unstable.
  - The current solution keeps the app working by relying on SQL-backed similarity instead.
  - This is practical, but it is still a workaround and should be discussed explicitly.

- Performance implications of the fallback
  - The SQL-backed similarity path is slower than native Chroma ANN search on larger corpora.
  - For current functionality it works, but it is not the ideal long-term path if Chroma can be stabilized.

- Duplicate basename handling during ingestion
  - Some ingestion logic still keys documents by filename rather than full filepath.
  - Two files with the same name in different subfolders of one project can still collide.

- Second-pass extraction behavior
  - There is audit support, but second-pass findings are still not cleanly positioned as graph-augmenting extraction.
  - This should be clarified before calling the extraction flow final.

- Working graph vs canonical graph
  - `kg.pkl` is still a working UI artifact.
  - UI graph edits do not currently sync back into canonical retrieval storage.
  - That separation is intentional, but it should be understood by everyone discussing the pull.

- Size and reviewability of the diff
  - `app.py`, `settings.py`, `extractor.py`, and `db_storage.py` carry a large share of the churn.
  - Functionally, many of the changes are reasonable.
  - Review-wise, this may still be easier to land if split into smaller pull requests.

## Suggested Pull Framing

If this work is to be proposed upstream, the cleanest framing would probably be:

- project-aware ingestion and storage
- project-aware UI and graph pickle restoration
- structured QA logging and graph snapshot support
- extraction and chunking cleanups
- Windows/Python 3.12 Chroma safety workaround

That would make it easier to discuss what is foundational, what is convenience/UI, and what is environment-specific stabilization.

# Observability And Modularity Inventory

## Project Aim

`appl-kgraph` is a graph-based RAG pipeline for querying large document collections with both vector retrieval and explicit entity/relation graph retrieval. The codebase is moving toward small, independently testable stages so ingestion, extraction, indexing, retrieval, and answering can be benchmarked and swapped separately.

## Observability Terms

- Audit: chatbot reproducibility records. These include the question, retrieved context, answer, conversation history, retriever settings, and model metadata.
- Verbosity: user-visible progress and status output, including UI progress callbacks, Gradio status text, CLI status details, and optional source/context panels.
- Internal logging: developer diagnostics and non-user-facing operational records, including `.log` files, logger calls, extraction validation sidecars, and context diagnostics.
- Canonical storage: durable project artifacts needed by the system. Graph pickles, SQLite stores, Chroma stores, and parsed raw/enriched document snapshots are not audit logs.

## Current Endpoints

### Discovery

Input: project/document root.

Output: supported file paths.

Current code: `project_paths.list_document_paths()` filters files by extension and skips the project artifact directory.

Desired type: `DocumentRef`.

### Parser

Input: file path.

Output: parsed pages and file metadata.

Current code: `ingestion.parse_to_pages()` calls `FileParser.parse_file()`.

Desired type: `RawDocument`.

### Text Enrichment

Input: raw parsed document text.

Output: normalized/enriched text.

Current code: language normalization and metadata cleanup live mostly in `ingestion.normalize_metadata()` and extractor prompt language resolution. This is not yet a standalone stage.

Desired type: `EnrichedDocument`.

### Chunker

Input: parsed pages plus document identity.

Output: chunk dictionaries with `chunk_uuid`, text, page range, file path, and language metadata.

Current code: `ingestion.build_chunks()` wraps `chunker.chunk_parsed_pages()`.

Desired type: `Chunk`.

### Entity/Relation Extraction

Input: chunks.

Output: entities, relationships, content keywords, per-chunk extraction results, and optional validation results.

Current code: `extractor.extract_from_chunks()` performs LLM extraction and optional second-pass validation.

Desired types: `Entity`, `Relation`, `GraphDelta`.

### Graph Normalization

Input: extracted entities and relations.

Output: normalized/upsertable graph nodes and edges.

Current code: `ingestion.ensure_edge_endpoints()` and `ingestion.merge_graph_data()` normalize nodes/edges before storage.

Desired type: `GraphDelta`.

### Storage And Indexing

Input: raw document text, chunks, graph nodes/edges, vector payloads.

Output: SQLite rows, Chroma collections, and graph pickle snapshots.

Current code: `db_storage.Storage`, `ingestion._write_retrieval_graph_snapshot()`, and `project_paths.resolve_project_paths()`.

Canonical artifacts: document DB, chunk DB, graph DB, Chroma stores, graph pickle, retrieval graph pickle, and parsed raw/enriched document snapshots.

### Query Analysis

Input: user query and conversation history.

Output: retrieval keywords or plan.

Current code: `pathrag.PathRAGRetriever` extracts seed entities directly; `lightrag.extract_keywords()` extracts high/low-level keywords.

Desired type: `QueryPlan`.

### Retrieval

Input: query plan.

Output: context windows, entities, relations, and chunks.

Current code: `pathrag.retrieve_context()` and `lightrag.build_context()`.

Desired type: `RetrievedContext`.

### Answer

Input: query, context window, and conversation history.

Output: answer plus audit payload metadata.

Current code: `PathRAGRetriever.retrieve()` and `LightRAGRetriever.aretrieve()`.

Desired type: `AnswerResult`.

## Completed Task Evidence

- Storage folder per document folder: `project_paths.resolve_project_paths()` resolves local `.appl-kgraph` storage, graph, log, audit, and diagnostics paths per selected document root.
- Log answers and associated input: `query_logging.write_audit_log()` writes chatbot reproducibility JSON under the project audit log directory.
- Logger level as setting: `settings.LoggingSettings.internal_log_level` controls developer logger severity.
- Retrieval answer token setting: `RetrievalSettings.answer_max_tokens` is exposed through `llm_max_tokens`.
- PathRAG conversation history: `PathRAGRetriever.retrieve()` renders history into the answer prompt.
- Extraction completeness check: `extractor.extract_from_chunks()` can run an internal validation pass and writes validation sidecars during ingestion when enabled.

## Implementation Sequence

1. Keep `ingest_paths()` as the compatibility wrapper while extracting stage-level functions around discovery, parse, enrich, chunk, extract, normalize, store, and index.
2. Add shared dataclasses first, then move behavior behind those dataclass boundaries in small PRs.
3. Split retrieval into query analysis, candidate retrieval, context building, reranking, answer generation, and audit logging.
4. Add benchmark entrypoints after stage boundaries exist, so benchmarks can invoke a single module without a full UI or end-to-end run.

## Test Expectations

- Settings tests cover audit, verbosity, internal logging, internal log level, and ignored legacy observability variables.
- Audit JSON tests assert chatbot reproducibility payloads are written under audit paths.
- Extraction validation tests assert diagnostic sidecars are gated by internal logging and named as validation output.
- Retrieval tests assert `top_k_chunk_per_entity` is read from settings.
- Stage tests should be added next to each module as boundaries become explicit.

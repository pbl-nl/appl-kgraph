# Modular Refactor Execution Plan

## Objective

Refactor the ingestion and retrieval pipeline into independently testable and benchmarkable components without changing user-visible behavior. Work through one section at a time, keep the existing `ingest_paths()`, PathRAG, and LightRAG entrypoints as compatibility wrappers until final integration, and do not combine unrelated modules in one commit.

## Working Rules

- Complete and review one numbered section before starting the next.
- Preserve current behavior with characterization tests before moving code.
- Introduce typed inputs and outputs before changing orchestration.
- Keep storage writes, LLM calls, and UI callbacks outside pure transformation functions.
- Pass dependencies explicitly where benchmarking requires substitution; use `Protocol` only when there are multiple real implementations or test doubles.
- Commit after each coherent change within one file or one module.
- Use Conventional Commits with a narrow scope, for example `refactor(parser): return raw document schema`.
- Run the section's focused tests before every commit and the non-storage suite at each section boundary.
- Do not remove compatibility wrappers until the final integration section.

## Target Pipeline

```text
project root
  -> discovery
  -> parser
  -> text enrichment
  -> chunker
  -> entity/relation extraction
  -> graph normalization
  -> canonical storage and vector indexing

query + history
  -> query analysis
  -> candidate retrieval
  -> context construction
  -> optional reranking
  -> answer generation
  -> chatbot audit record
```

## Section 0: Baseline And Test Harness

Goal: establish a trustworthy baseline before moving behavior.

Work:

- Record the current passing test command and known storage-test environment requirement.
- Add characterization tests around `ingest_paths()`, PathRAG retrieval results, and LightRAG retrieval results using fakes at external boundaries.
- Configure test discovery to exclude `_old/` explicitly.
- Decide how storage integration tests obtain a local Chroma implementation in development and CI.

Exit criteria:

- Current wrappers have behavioral tests.
- Unit tests and integration tests are clearly separated.
- Test failures identify code regressions rather than package-mode conflicts.

Suggested commits:

- `test(ingestion): characterize pipeline orchestration`
- `test(retrieval): characterize retriever outputs`
- `chore(tests): separate unit and storage integration suites`

## Section 1: Shared Contracts

Goal: make stage boundaries explicit without moving orchestration.

Endpoints and types:

- `DocumentRef`
- `RawDocument`
- `EnrichedDocument`
- `Chunk`
- `ExtractionResult`
- `Entity`
- `Relation`
- `GraphDelta`
- `QueryPlan`
- `RetrievalCandidates`
- `RetrievedContext`
- `AnswerResult`

Work:

- Review `graph/schemas.py` against actual database and retriever payloads.
- Add missing `ExtractionResult` and `RetrievalCandidates` contracts.
- Add explicit conversion helpers between legacy dictionaries and schemas at module boundaries.
- Define invariants such as stable IDs, page numbering, source references, and metadata ownership.

Exit criteria:

- Every target stage has one documented input and output type.
- Schema tests cover empty values, conversion round trips, duplicate names, and optional metadata.
- Existing callers still receive their current dictionary/result shapes.

Suggested commits:

- `refactor(schemas): add extraction result contract`
- `refactor(schemas): add retrieval candidate contract`
- `test(schemas): cover legacy payload conversion`

## Section 2: Discovery

Goal: isolate project-root scanning from ingestion orchestration.

Endpoint:

```python
discover_documents(project_root: Path) -> list[DocumentRef]
```

Current source: `project_paths.list_document_paths()` and ingestion path filtering.

Work:

- Move discovery behavior into a focused module while retaining `list_document_paths()` as an adapter.
- Make extension filtering, project-artifact exclusion, ordering, and temporary-file rules explicit.
- Keep filesystem access in this stage; do not parse content here.

Exit criteria:

- Discovery can run independently from storage and LLM dependencies.
- Tests cover nested paths, unsupported files, temporary files, missing roots, and `.appl-kgraph` exclusion.

Suggested commits:

- `refactor(discovery): add document discovery endpoint`
- `test(discovery): cover filtering and project artifacts`
- `refactor(project-paths): delegate document listing`

## Section 3: Parser

Goal: convert a `DocumentRef` into a stable raw-document representation.

Endpoint:

```python
parse_document(document: DocumentRef, parser: DocumentParser) -> RawDocument
```

Current source: `FileParser.parse_file()` and `ingestion.parse_to_pages()`.

Work:

- Standardize page numbering, metadata, file errors, and text joining.
- Separate format-specific adapters from the parser endpoint.
- Replace swallowed exceptions with a typed parse result or domain exception handled by orchestration.
- Keep raw snapshot persistence out of the parser.

Exit criteria:

- TXT, Markdown, HTML, PDF, and DOCX parsing share one output contract.
- Parser tests use fixtures and do not initialize storage or an LLM.
- Empty files and parser failures have documented behavior.

Suggested commits:

- `refactor(parser): return raw document contract`
- `refactor(parser): normalize format adapter errors`
- `test(parser): cover supported document formats`

## Section 4: Text Enrichment

Goal: make normalization and optional enrichment composable.

Endpoint:

```python
enrich_document(raw: RawDocument, enrichers: Sequence[TextEnricher]) -> EnrichedDocument
```

Current source: `ingestion.normalize_metadata()`, language handling, and future cleaning/normalization behavior.

Work:

- Start with behavior-preserving metadata and language normalization.
- Model enrichers as ordered transformations with explicit names and settings.
- Keep the identity/no-op enrichment path valid.
- Do not combine extraction prompts or chunking with enrichment.

Exit criteria:

- Each enricher is independently testable.
- The enriched document records which transformations were applied.
- Raw and enriched snapshots remain canonical storage artifacts.

Suggested commits:

- `refactor(enrichment): add enrichment pipeline endpoint`
- `refactor(enrichment): extract metadata normalization`
- `test(enrichment): cover ordered transformations`

## Section 5: Chunker

Goal: isolate deterministic conversion from enriched documents to chunks.

Endpoint:

```python
chunk_document(document: EnrichedDocument, config: ChunkingConfig) -> list[Chunk]
```

Current source: `chunker.chunk_parsed_pages()` and `ingestion.build_chunks()`.

Work:

- Move ID, page-range, filepath, language, and character-count ownership into the chunker boundary.
- Keep chunking deterministic when supplied an ID factory for tests.
- Validate overlap and maximum-size settings before processing.

Exit criteria:

- Chunker has no storage, UI, or LLM dependency.
- Tests cover empty text, oversized sentences, overlap, page boundaries, and stable metadata.

Suggested commits:

- `refactor(chunker): accept enriched document input`
- `refactor(chunker): centralize chunk metadata`
- `test(chunker): cover document boundary cases`

## Section 6: Entity And Relation Extraction

Goal: isolate LLM extraction from graph normalization and persistence.

Endpoint:

```python
extract_graph(chunks: Sequence[Chunk], extractor: GraphExtractor) -> ExtractionResult
```

Current source: `extractor.extract_from_chunks()`.

Work:

- Separate prompt construction, model invocation, response parsing, and optional validation into focused functions.
- Keep cache access behind an injected cache dependency.
- Return extraction findings without creating placeholder nodes or writing diagnostics.
- Treat completeness validation as an optional extraction diagnostic, not chatbot audit.

Exit criteria:

- Model calls can be replaced with deterministic fakes.
- Parser tests cover malformed output and missing fields.
- Validation can be enabled independently and returns typed results.

Suggested commits:

- `refactor(extraction): add extraction result endpoint`
- `refactor(extraction): isolate model response parser`
- `refactor(extraction): inject llm cache dependency`
- `test(extraction): cover validation and malformed output`

## Section 7: Graph Normalization

Goal: make graph merging a pure, independently benchmarkable transformation.

Endpoint:

```python
normalize_graph(extraction: ExtractionResult, existing: GraphView) -> GraphDelta
```

Current source: `ensure_edge_endpoints()`, `merge_graph_data()`, `_resolve_type()`, and description merging in `ingestion.py`.

Work:

- Move entity deduplication, type resolution, relation grouping, placeholder creation, and source-ID merging into a graph-normalization module.
- Define deterministic conflict rules.
- Return additions, updates, and deletions explicitly rather than writing to storage.

Exit criteria:

- Normalization is pure and deterministic.
- Tests cover duplicate names, unknown types, repeated relations, missing endpoints, and source removal.

Suggested commits:

- `refactor(graph): extract entity normalization`
- `refactor(graph): extract relation normalization`
- `refactor(graph): return explicit graph delta`
- `test(graph): cover normalization conflicts`

## Section 8: Canonical Storage And Indexing

Goal: separate persistence responsibilities while retaining one orchestration facade.

Interfaces:

```python
DocumentStore.put_raw(document: RawDocument) -> None
DocumentStore.put_enriched(document: EnrichedDocument) -> None
ChunkStore.put_many(chunks: Sequence[Chunk]) -> None
GraphStore.apply(delta: GraphDelta) -> None
VectorIndex.index_chunks(chunks: Sequence[Chunk]) -> None
VectorIndex.index_graph(delta: GraphDelta) -> None
```

Current source: `db_storage.Storage`, raw snapshot writing, and graph pickle helpers.

Work:

- Keep SQLite, Chroma, snapshots, and graph pickle behavior in separate adapters.
- Preserve `Storage` as a compatibility facade during migration.
- Make transaction and partial-failure behavior explicit.
- Ensure raw/enriched snapshots are not controlled by observability gates.

Exit criteria:

- Each store can be replaced independently in tests and benchmarks.
- Canonical storage and vector indexing failures are distinguishable.
- Integration tests run against an explicitly provisioned local Chroma implementation.

Suggested commits:

- `refactor(storage): extract document store adapter`
- `refactor(storage): extract chunk and graph stores`
- `refactor(indexing): isolate vector index writes`
- `test(storage): cover canonical artifact persistence`

## Section 9: Ingestion Orchestration

Goal: make `ingest_paths()` a thin compatibility wrapper around explicit stages.

Endpoint:

```python
ingest_documents(documents: Sequence[DocumentRef], pipeline: IngestionPipeline) -> IngestionSummary
```

Work:

- Compose discovery, parse, enrich, chunk, extract, normalize, store, and index stages.
- Keep user progress reporting in orchestration, gated by verbosity.
- Keep internal logs and diagnostics outside transformation stages.
- Preserve skip/replace/remove semantics and summary counts.

Exit criteria:

- Every stage can be invoked without the full ingestion pipeline.
- `ingest_paths()` delegates and retains backward-compatible behavior.
- A failure identifies the stage and document involved.

Suggested commits:

- `refactor(ingestion): add pipeline orchestrator`
- `refactor(ingestion): delegate compatibility wrapper`
- `test(ingestion): cover stage failures and summaries`

## Section 10: Query Analysis

Goal: turn query interpretation into a pluggable stage shared by retrieval strategies.

Endpoint:

```python
analyze_query(query: str, history: ConversationHistory) -> QueryPlan
```

Current source: LightRAG keyword extraction and PathRAG seed-entity selection.

Work:

- Preserve strategy-specific analyzers behind one contract.
- Record mode, keywords, seed entities, and relevant settings in `QueryPlan`.
- Keep history rendering and model invocation testable independently.

Exit criteria:

- Query analyzers run without storage access.
- Tests cover empty analysis, history limits, reformulation, and strategy-specific plans.

Suggested commits:

- `refactor(query): add query plan contract`
- `refactor(lightrag): extract query analyzer`
- `refactor(pathrag): extract query analyzer`
- `test(query): cover history-aware analysis`

## Section 11: Retrieval, Context, And Reranking

Goal: separate candidate lookup from context construction and optional reranking.

Endpoints:

```python
retrieve_candidates(plan: QueryPlan, stores: RetrievalStores) -> RetrievalCandidates
build_context(plan: QueryPlan, candidates: RetrievalCandidates) -> RetrievedContext
rerank_context(context: RetrievedContext, reranker: Reranker) -> RetrievedContext
```

Current source: PathRAG context retrieval and LightRAG `build_context()`.

Work:

- Isolate vector queries, graph traversal, evidence expansion, deduplication, truncation, and reranking.
- Preserve PathRAG and LightRAG differences as strategy implementations.
- Keep context construction free of answer generation and audit writes.

Exit criteria:

- Candidate retrieval and context construction can be benchmarked separately.
- Reranking is optional and replaceable.
- Tests cover no-result queries, duplicate evidence, token limits, and deterministic ordering.

Suggested commits:

- `refactor(retrieval): add candidate retrieval contract`
- `refactor(pathrag): extract context builder`
- `refactor(lightrag): extract context builder`
- `refactor(retrieval): isolate reranking stage`
- `test(retrieval): cover context assembly boundaries`

## Section 12: Answer Generation And Audit

Goal: separate answer generation from retrieval and chatbot reproducibility recording.

Endpoints:

```python
generate_answer(
    query: str,
    context: RetrievedContext,
    history: ConversationHistory,
    generator: AnswerGenerator,
) -> AnswerResult

record_chat_audit(result: AnswerResult, project_paths: ProjectPaths) -> Path | None
```

Work:

- Move prompt assembly and answer model invocation behind an answer generator.
- Ensure `AnswerResult` contains model and retrieval metadata needed for reproducibility.
- Keep audit recording as a post-answer side effect controlled only by `AUDIT_ENABLED`.
- Preserve existing PathRAG and LightRAG public result types through adapters.

Exit criteria:

- Answers can be generated from a supplied context without storage access.
- Audit records contain query, history, context, answer, settings, and model metadata.
- Audit failures do not silently alter the answer result.

Suggested commits:

- `refactor(answer): add answer generator endpoint`
- `refactor(audit): accept answer result contract`
- `refactor(pathrag): delegate answer generation`
- `refactor(lightrag): delegate answer generation`
- `test(audit): cover reproducibility payload`

## Section 13: Benchmark Entrypoints

Goal: benchmark individual stages and complete strategies without the UI.

Work:

- Add stage runners for parse, enrich, chunk, extract, normalize, query analysis, retrieval, reranking, and answer generation.
- Use stable JSON input/output fixtures based on shared schemas.
- Record duration, token use, model/settings fingerprint, and quality inputs without mixing benchmark results with chatbot audit logs.
- Support component substitution through explicit configuration.

Exit criteria:

- Every expensive or variable stage can be run independently.
- A complete ingestion/retrieval benchmark can compose the same production stages.
- Benchmark outputs are deterministic where external models are replaced by fixtures.

Suggested commits:

- `feat(benchmark): add stage runner framework`
- `feat(benchmark): add ingestion stage commands`
- `feat(benchmark): add retrieval stage commands`
- `test(benchmark): cover fixture-based runs`

## Section 14: Final Integration

Goal: verify the modular pipeline as one system and retire transitional code deliberately.

Work:

- Run end-to-end ingestion and both retrieval strategies against a small fixture corpus.
- Compare outputs with the Section 0 characterization baseline.
- Test UI progress, CLI behavior, observability gates, canonical artifacts, document replacement/removal, and audit reproduction.
- Remove deprecated aliases only when all callers and migration notes are ready.
- Update README architecture and contributor guidance.

Exit criteria:

- Unit, storage integration, and end-to-end suites pass in the documented environment.
- PathRAG and LightRAG use the shared contracts and stage interfaces.
- No production stage depends on Gradio or CLI code.
- Benchmarks invoke production components rather than duplicate implementations.

Suggested commits:

- `test(integration): cover modular ingestion and retrieval`
- `refactor(compat): remove migrated pipeline aliases`
- `docs(architecture): document modular pipeline`

## Review Rhythm

For each section:

1. Confirm the current endpoint and behavior with tests.
2. Implement one file or one module boundary.
3. Run focused tests and commit.
4. Wire the compatibility wrapper.
5. Run the section test set and commit.
6. Review the diff and update this plan with any contract decisions.
7. Stop for section approval before continuing.

After Sections 5, 9, 12, and 14, run the complete available test suite and record any environment-dependent exclusions in the pull request.

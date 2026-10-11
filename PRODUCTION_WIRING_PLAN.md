# PR 49: production wiring

Base: PR #48 (`66d2e1d`). Local candidate: `codex/modularity-49-production-wiring`. No push or GitHub PR creation.

PRs through #48 delivered the generic contracts and connector interfaces. Their production integration was incomplete; describing the whole refactor as implemented at that point was inaccurate.

## Required scope

1. Bind the real parser, enrichment, chunking, extraction, pure graph normalization, canonical stores and vector indexes to `ingest_documents()`.
2. Route both RAG strategies through query analysis, candidate retrieval, context construction, optional reranking, answer generation and audit.
3. Update UI/CLI consumers for `AnswerResult` and remove replaced ingestion/RAG implementations. Preserve skip, replacement, source removal, partial selection, pending markers, canonical snapshots and resource cleanup.
4. Verify the actual public pipelines, independent strategy stages and UI/CLI callbacks using real local stores and deterministic external model/embedding substitutes.

Small contract fixes remain where required: metadata ownership, endpoint provenance, duplicate chunk IDs, retrieval settings snapshots, native page metadata, distinct reranker IDs, parse error handling and supported HTML dependencies. Keep the DOCX temporary-file/process safety fix: the previous parser forcibly terminated Word and could overwrite a neighboring PDF. The production factory must not import the CLI/orchestration module to remove a document.

## Deferred and preserved

The broader candidate is preserved at `codex/modularity-49-extended-review` (`9ff2589`). New benchmark tooling, no-cache extraction behavior, unused configuration/chunker cleanup and broad README/TESTING rewrites are excluded from this wiring candidate. They are not being presented as completed upstream work.

## Verification

On 2026-10-12, the narrowed candidate passed **189 tests** in 28.48 seconds, with the same five existing SWIG deprecation warnings. The pre-migration PR #48 baseline was 153 passing tests; the broader preserved candidate had 195. The six excluded tests cover the deferred benchmark/no-cache extraction features; production wiring and failure regressions remain.

Static checks pass for all 29 changed Python files, and the whitespace check passes with the repository's existing CRLF convention. The caller trace covers `ingest_paths()`, both RAG public entrypoints, UI/CLI consumers and evaluation callers. Both query paths invoke analysis, retrieval, context, reranking, answer and audit; ingestion invokes the typed pipeline and real storage/index adapters. Pure context/answer adapters are tested independently. Real parser/local-store/model-boundary tests retain actual extraction response parsing and caching, substituting external model/embedding calls. No live provider calls or app server were started.

The new factory no longer imports ingestion orchestration for document removal. Main-checkout edits remain outside this candidate. The original larger version is recoverable from the preserved branch named above.

## Existing limitations for separate review

Shared graph descriptions aggregate contributions across documents; removing/selecting a source cannot subtract its description from a surviving aggregate. Document filtering follows global top-k retrieval, and neither strategy enforces a total prompt-token budget. These are pre-existing retrieval/storage limitations, not missing stage wiring. Live provider behavior, native Chroma mutations on other platforms, Word automation and ranker model quality are outside deterministic local test coverage.

Merge the existing stack oldest first (#37 through #48), verifying each next base/diff. Publish this candidate only when requested, targeting #48 while it is open or `main` once the stack has merged. Preserve ancestry with merge commits, or rebase remaining children after squash merges.

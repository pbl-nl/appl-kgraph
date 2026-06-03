# AGENTS.md

## Project Goal

`appl-kgraph` is a modular graph-based RAG system for querying large document collections by combining vector retrieval with an explicit knowledge graph of extracted entities and relations. It supports pluggable retrieval strategies, including vanilla RAG, LightRAG-style retrieval, and PathRAG-style relational path retrieval.

## Collaboration Guidelines

- Keep changes small, focused, and easy to review.
- Prefer frequent commits over large batches of unrelated work.
- Group related commits into properly scoped pull requests.
- Avoid broad refactors unless they directly support the task at hand.
- Preserve the modular design so retrieval strategies, indexing, prompts, and storage can remain independently inspectable and extensible.

## Commit Style

Use Conventional Commits:

```text
type(scope): short description
```

Common types include:

- `feat`: a new capability or user-visible behavior
- `fix`: a bug fix or behavioral correction
- `chore`: maintenance, tooling, dependency, or repository upkeep
- `docs`: documentation-only changes
- `test`: tests or test fixtures
- `refactor`: code restructuring without intended behavior changes

Examples:

```text
feat(retrieval): add path pruning configuration
fix(ingestion): handle empty parsed documents
chore(deps): update graph storage dependencies
docs(readme): clarify Python environment setup
test(chunker): cover overlapping chunk boundaries
```

Keep the scope short and meaningful, such as `retrieval`, `ingestion`, `graph`, `storage`, `prompts`, `ui`, `tests`, or `docs`.

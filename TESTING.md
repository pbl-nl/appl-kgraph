# Testing

Run tests from the repository root with the project virtual environment.

## Complete Baseline

The current modules use a mix of package imports and top-level imports from
`graph/`. Until those imports are normalized, add both the repository root and
`graph/` to `PYTHONPATH` when running the complete suite in PowerShell:

```powershell
$env:PYTHONPATH = "$(Get-Location)\graph;$(Get-Location)"
.\.venv\Scripts\python.exe -m pytest -q test
```

Baseline on June 22, 2026: 29 tests passed.

## Non-Storage Tests

The tests that do not require Chroma run in normal package mode:

```powershell
.\.venv\Scripts\python.exe -m pytest -q test --ignore=test/test_storage_full.py
```

Baseline on June 22, 2026: 21 tests passed.

## Storage Integration Tests

Storage integration tests require the `chromadb` package from
`requirements.txt`. They use embedded local Chroma stores created under each
test's temporary directory; no external Chroma service or embedding API is
required.

```powershell
$env:PYTHONPATH = "$(Get-Location)\graph;$(Get-Location)"
.\.venv\Scripts\python.exe -m pytest -q test/test_storage_full.py
```

Baseline on June 22, 2026: 8 tests passed.

Running the storage tests without `graph/` on `PYTHONPATH` currently fails
during collection because `graph/db_storage.py` imports `llm` as a top-level
module. This is a known package-mode constraint, not a storage service
requirement.

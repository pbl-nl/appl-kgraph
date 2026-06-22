# Testing

Run tests from the repository root with the project virtual environment.

## Complete Baseline

Run both the unit and storage integration suites:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Baseline on June 22, 2026: 37 tests passed.

## Unit Tests

Exclude tests that exercise the storage integrations:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not storage_integration"
```

Baseline on June 22, 2026: 29 tests passed.

## Storage Integration Tests

Storage integration tests require the `chromadb` package from
`requirements.txt`. They use embedded local Chroma stores created under each
test's temporary directory; no external Chroma service or embedding API is
required.

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m storage_integration
```

Baseline on June 22, 2026: 8 tests passed.

Development and CI use the same embedded Chroma arrangement. CI must install
`requirements.txt`, including `chromadb`, but does not need a Chroma service
container or external embedding credentials.

`pytest.ini` adds both the repository root and `graph/` to the test import path
because production modules currently mix package and top-level imports. This
keeps test collection deterministic until those imports are normalized.

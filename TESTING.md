# Testing

Use the Conda developer environment from the repository root:

```sh
conda env create -f appl-kgraph.yml
conda activate appl-kgraph
```

Run the complete suite for both provider configurations:

```sh
python -m pytest -q --llm-provider=openai
python -m pytest -q --llm-provider=azure
```

Tests use isolated fake settings and mocked model responses. They do not read a
developer's `.env` or require real OpenAI or Azure credentials. Provider tests
exercise the real SDK request construction with a local HTTP mock, checking the
endpoint, authentication, deployment/model, and returned chat/embedding results.

To run only the tests without embedded storage, add `-m "not storage_integration"`.
To run only embedded storage tests, add `-m storage_integration`.

Storage integration tests use SQLite and embedded local Chroma stores in each
test's temporary directory. No external Chroma service or embedding API is
required. CI should create the same Conda environment and run both provider
commands above.

`pytest.ini` adds the repository root and `graph/` to the test import path because
production modules currently mix package and top-level imports.

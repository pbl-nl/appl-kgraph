"""Run tests with deterministic fake credentials, independent of developer settings."""

import os

import dotenv
import pytest


_SETTINGS_PREFIXES = (
    "OPENAI_", "AZURE_OPENAI_", "LLM_", "CHAT_", "EMBEDDING_", "PROMPT_",
    "MERGE_", "DESCRIPTION_", "CHUNK_", "CHUNKS_", "DOCUMENTS_", "GRAPH_",
    "CHROMA_", "PROJECT_", "EXTRACTION_", "AUDIT_", "VERBOSITY_",
    "INTERNAL_LOG", "RETRIEVAL_",
)
_PROVIDER_ENV = {
    "openai": {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_LLM_MODEL": "test-openai-chat",
        "OPENAI_EMBEDDINGS_MODEL": "test-openai-embedding",
    },
    "azure": {
        "AZURE_OPENAI_API_KEY": "test-azure-key",
        "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
        "AZURE_OPENAI_LLM_DEPLOYMENT_NAME": "test-azure-chat",
        "AZURE_OPENAI_EMB_DEPLOYMENT_NAME": "test-azure-embedding",
    },
}


def pytest_addoption(parser):
    parser.addoption(
        "--llm-provider",
        choices=("openai", "azure"),
        default="openai",
        help="Select the fake provider configuration used by the test suite.",
    )


def pytest_configure(config):
    # Settings are imported during collection, before ordinary fixtures run.
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    patch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for name in list(os.environ):
        if name.startswith(_SETTINGS_PREFIXES):
            patch.delenv(name)
    provider = config.getoption("--llm-provider")
    patch.setenv("LLM_PROVIDER", provider)
    for name, value in _PROVIDER_ENV[provider].items():
        patch.setenv(name, value)


@pytest.fixture(scope="session")
def llm_provider(pytestconfig):
    return pytestconfig.getoption("--llm-provider")

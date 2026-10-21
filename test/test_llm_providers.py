"""Exercise provider configuration and SDK requests without external services."""

import json
from functools import partial

import httpx
import pytest
from openai import AzureOpenAI, OpenAI

import llm
import settings as settings_module


_PROVIDER_ENV = {
    "openai": {
        "OPENAI_API_KEY": "openai-test-secret",
        "OPENAI_BASE_URL": "https://openai.test/v1",
        "OPENAI_LLM_MODEL": "chat-model",
        "OPENAI_EMBEDDINGS_MODEL": "embedding-model",
    },
    "azure": {
        "AZURE_OPENAI_API_KEY": "azure-test-secret",
        "AZURE_OPENAI_ENDPOINT": "https://azure.test",
        "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
        "AZURE_OPENAI_LLM_DEPLOYMENT_NAME": "chat-deployment",
        "AZURE_OPENAI_EMB_DEPLOYMENT_NAME": "embedding-deployment",
    },
}


def _set_provider_environment(monkeypatch, provider):
    for name in _PROVIDER_ENV["openai"] | _PROVIDER_ENV["azure"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", provider)
    for name, value in _PROVIDER_ENV[provider].items():
        monkeypatch.setenv(name, value)


@pytest.fixture(params=("openai", "azure"))
def provider_settings(request, monkeypatch):
    _set_provider_environment(monkeypatch, request.param)
    configured = settings_module.load_settings()
    monkeypatch.setattr(llm, "settings", configured)
    return configured


@pytest.fixture
def sdk_requests(monkeypatch):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request, body))
        if request.url.path.endswith("/chat/completions"):
            result = {
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Grounded answer"},
                    "finish_reason": "stop",
                }],
            }
        else:
            result = {
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": index, "embedding": [float(len(text)), 0.5]}
                    for index, text in enumerate(body["input"])
                ],
            }
        return httpx.Response(200, json=result)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(llm, "OpenAI", partial(OpenAI, http_client=client))
        monkeypatch.setattr(llm, "AzureOpenAI", partial(AzureOpenAI, http_client=client))
        yield requests


def _assert_provider_route(request, provider, operation, deployment):
    if provider == "azure":
        assert request.url.host == "azure.test"
        assert request.url.path == f"/openai/deployments/{deployment}/{operation}"
        assert request.url.params["api-version"] == "2024-02-15-preview"
        assert request.headers["api-key"] == "azure-test-secret"
    else:
        assert request.url.host == "openai.test"
        assert request.url.path == f"/v1/{operation}"
        assert "api-version" not in request.url.params
        assert request.headers["authorization"] == "Bearer openai-test-secret"


def test_chat_routes_generation_and_returns_sdk_content(provider_settings, sdk_requests):
    chat = llm.Chat()

    assert chat.generate("What changed?", system="Use the documents.") == "Grounded answer"
    assert chat.generate("Again", temperature=0.3, max_tokens=17) == "Grounded answer"

    provider = provider_settings.provider.provider
    expected_model = "chat-deployment" if provider == "azure" else "chat-model"
    first_request, first_body = sdk_requests[0]
    _assert_provider_route(first_request, provider, "chat/completions", "chat-deployment")
    assert first_body == {
        "model": expected_model,
        "messages": [
            {"role": "system", "content": "Use the documents."},
            {"role": "user", "content": "What changed?"},
        ],
        "temperature": provider_settings.chat.temperature,
        "max_tokens": provider_settings.chat.completion_max_tokens,
    }
    assert sdk_requests[1][1] == {
        "model": expected_model,
        "messages": [{"role": "user", "content": "Again"}],
        "temperature": 0.3,
        "max_tokens": 17,
    }


def test_embedder_routes_batches_and_preserves_input_order(provider_settings, sdk_requests):
    embedder = llm.Embedder()

    assert embedder.embed_texts(iter(["one", "second", "third"]), batch_size=2) == [
        [3.0, 0.5], [6.0, 0.5], [5.0, 0.5],
    ]
    assert embedder.embed_texts([]) == []
    assert len(sdk_requests) == 2
    assert [body["input"] for _, body in sdk_requests] == [["one", "second"], ["third"]]
    provider = provider_settings.provider.provider
    expected_model = "embedding-deployment" if provider == "azure" else "embedding-model"
    for request, body in sdk_requests:
        _assert_provider_route(request, provider, "embeddings", "embedding-deployment")
        assert body["model"] == expected_model

    assert embedder.dimension == 2
    assert embedder.dimension == 2
    assert len(sdk_requests) == 3
    assert sdk_requests[-1][1]["input"] == [" "]


@pytest.mark.parametrize(
    ("provider", "missing", "adapter"),
    [
        ("openai", "OPENAI_API_KEY", llm.Chat),
        ("openai", "OPENAI_LLM_MODEL", llm.Chat),
        ("openai", "OPENAI_API_KEY", llm.Embedder),
        ("azure", "AZURE_OPENAI_API_KEY", llm.Chat),
        ("azure", "AZURE_OPENAI_ENDPOINT", llm.Chat),
        ("azure", "AZURE_OPENAI_LLM_DEPLOYMENT_NAME", llm.Chat),
        ("azure", "AZURE_OPENAI_API_KEY", llm.Embedder),
        ("azure", "AZURE_OPENAI_ENDPOINT", llm.Embedder),
        ("azure", "AZURE_OPENAI_EMB_DEPLOYMENT_NAME", llm.Embedder),
    ],
)
def test_model_adapter_rejects_missing_provider_configuration(
    monkeypatch, provider, missing, adapter
):
    _set_provider_environment(monkeypatch, provider)
    monkeypatch.delenv(missing)
    monkeypatch.setattr(llm, "settings", settings_module.load_settings())

    with pytest.raises(RuntimeError, match=missing):
        adapter()


def test_openai_embedder_uses_default_model_when_not_configured(monkeypatch, sdk_requests):
    _set_provider_environment(monkeypatch, "openai")
    monkeypatch.delenv("OPENAI_EMBEDDINGS_MODEL")
    monkeypatch.setattr(llm, "settings", settings_module.load_settings())

    assert llm.Embedder().embed_texts(["one"]) == [[3.0, 0.5]]
    assert sdk_requests[0][1]["model"] == "text-embedding-3-small"

def test_settings_reject_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")

    with pytest.raises(RuntimeError, match="LLM_PROVIDER must be 'openai' or 'azure'"):
        settings_module.load_settings()

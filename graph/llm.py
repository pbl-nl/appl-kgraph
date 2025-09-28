from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
from prompts import PROMPTS
from settings import settings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, AzureOpenAI

# ---------------------------
# Embeddings
# ---------------------------

class Embedder:
    """
    Provider-agnostic embeddings.
    For OpenAI:
      - OPENAI_API_KEY, OPENAI_EMBEDDINGS_MODEL, [OPENAI_BASE_URL]
    For Azure:
      - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMB_DEPLOYMENT_NAME
    """
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        prov = (provider or settings.provider.provider).strip().lower()
        self.provider = prov
        self._dimension = None  # Dimensionality property for compatibility between LLMs

        if prov == "azure":
            if AzureOpenAI is None:
                raise RuntimeError("openai package not installed. pip install openai")
            self.model = model or settings.provider.azure_embeddings_deployment
            key = api_key or settings.provider.azure_api_key
            endpoint = azure_endpoint or settings.provider.azure_endpoint
            version = azure_api_version or settings.provider.azure_api_version
            if not (self.model and key and endpoint):
                raise RuntimeError("Set AZURE_OPENAI_EMB_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT.")
            self.client = AzureOpenAI(api_key=key, api_version=version, azure_endpoint=endpoint)

        elif prov == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not installed. pip install openai")
            self.model = model or (settings.provider.openai_embeddings_model or "text-embedding-3-small")
            key = api_key or settings.provider.openai_api_key
            base = base_url or settings.provider.openai_base_url
            if not (self.model and key):
                raise RuntimeError("Set OPENAI_API_KEY and OPENAI_EMBEDDINGS_MODEL.")
            self.client = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)

        else:
            raise RuntimeError("LLM_PROVIDER must be 'openai' or 'azure'.")

    @property
    def dimension(self) -> int:
        """
        Lazily fetch the embedding dimensionality once (cached).
        """
        if self._dimension is None:
            # one cheap call
            resp = self.client.embeddings.create(input=[" "], model=self.model)
            self._dimension = len(resp.data[0].embedding)
        return self._dimension
    
    def embed_texts(self, texts: Iterable[str], batch_size: int = settings.embeddings.batch_size) -> List[List[float]]:
        out: List[List[float]] = []
        batch: List[str] = []
        for t in texts:
            batch.append(t)
            if len(batch) >= batch_size:
                resp = self.client.embeddings.create(input=batch, model=self.model)  # both OpenAI & Azure
                out.extend([d.embedding for d in resp.data])
                batch = []
        if batch:
            resp = self.client.embeddings.create(input=batch, model=self.model)
            out.extend([d.embedding for d in resp.data])
        return out

# ---------------------------
# Chat completions
# ---------------------------

_CHAT_SINGLETON = None
_CHAT_LOCK = threading.Lock()

class Chat:
    """
    Unified chat client. Chooses OpenAI or AzureOpenAI based on LLM_PROVIDER in settings.
    """
    def __init__(self):
        prov = settings.provider.provider

        if prov == "azure":
            if AzureOpenAI is None:
                raise RuntimeError("openai package not installed. pip install openai")
            self.client = AzureOpenAI(
                api_key=settings.provider.azure_api_key,
                api_version=settings.provider.azure_api_version,
                azure_endpoint=settings.provider.azure_endpoint,
            )
            self.model = settings.provider.azure_llm_deployment

        elif prov == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not installed. pip install openai")
            base = settings.provider.openai_base_url
            if base:
                self.client = OpenAI(api_key=settings.provider.openai_api_key, base_url=base)
            else:
                self.client = OpenAI(api_key=settings.provider.openai_api_key)
            self.model = settings.provider.openai_llm_model

        else:
            raise RuntimeError("LLM_PROVIDER must be 'openai' or 'azure'.")

    def generate(self, prompt: str,
             system: Optional[str] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        t = settings.chat.temperature if temperature is None else temperature
        mt = settings.chat.max_tokens if max_tokens is None else max_tokens

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=t,
            max_tokens=mt,
        )
        return resp.choices[0].message.content or ""
    
    @classmethod
    def singleton(cls) -> "Chat":
        global _CHAT_SINGLETON
        if _CHAT_SINGLETON is None:
            with _CHAT_LOCK:
                if _CHAT_SINGLETON is None:
                    _CHAT_SINGLETON = Chat()
        return _CHAT_SINGLETON

def llm_summarize_text(to_summarize: str, language: str = "English") -> str:
    """
    Small helper used elsewhere (e.g., during long description merges).
    """
    chat = Chat.singleton()  # provider-agnostic
    prompt = PROMPTS["summarize_text"].format(text=to_summarize, language=language)
    return chat.generate(prompt).strip()

# !!Currently not in use, but could be useful for batch operations. !!
def generate_many(prompts_and_systems: List[Tuple[str, Optional[str]]],
                  max_workers: Optional[int] = None) -> List[str]:
    """
    Run many Chat.generate calls concurrently.
    Each item is (prompt, system). Returns results in the same order.
    """
    chat = Chat.singleton()
    workers = max_workers or int(settings.llmperf.max_concurrency)
    out: List[Optional[str]] = [None] * len(prompts_and_systems)

    def _one(i: int, p: str, s: Optional[str]):
        out[i] = chat.generate(prompt=p, system=s)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, i, p, s) for i, (p, s) in enumerate(prompts_and_systems)]
        for _ in as_completed(futs):
            pass
    return [x or "" for x in out]
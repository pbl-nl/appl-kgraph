from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Any, Dict, Iterable, List, Optional, Tuple
from prompts import PROMPTS
from settings import settings


# ─────────────────────────────────────────────────────────────
# Azure OpenAI chat client (env-driven)
#   Required env vars:
#     - AZURE_OPENAI_ENDPOINT
#     - AZURE_OPENAI_API_KEY
#     - AZURE_OPENAI_CHAT_DEPLOYMENT_NAME   (the chat model deployment name)
#     - AZURE_OPENAI_API_VERSION            (optional, defaults to 2024-02-15-preview)
# ─────────────────────────────────────────────────────────────
try:
    from openai import OpenAI, AzureOpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore
    AzureOpenAI = None  # type: ignore

class Chat:
    """
    Unified chat client. Chooses OpenAI or AzureOpenAI based on LLM_PROVIDER.
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

def llm_summarize_text(to_summarize: str, language: str = "English") -> str:
    """
    Small helper used elsewhere (e.g., during long description merges).
    """
    chat = Chat()  # provider-agnostic
    prompt = PROMPTS["summarize_text"].format(text=to_summarize, language=language)
    return chat.generate(prompt).strip()


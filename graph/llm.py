from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Any, Dict, Iterable, List, Optional, Tuple
from graph.prompts import PROMPTS
load_dotenv()

# ─────────────────────────────────────────────────────────────
# Azure OpenAI chat client (env-driven)
#   Required env vars:
#     - AZURE_OPENAI_ENDPOINT
#     - AZURE_OPENAI_API_KEY
#     - AZURE_OPENAI_CHAT_DEPLOYMENT_NAME   (the chat model deployment name)
#     - AZURE_OPENAI_API_VERSION            (optional, defaults to 2024-02-15-preview)
# ─────────────────────────────────────────────────────────────
try:
    from openai import AzureOpenAI  # type: ignore
except Exception:
    AzureOpenAI = None  # type: ignore


@dataclass
class AzureConfig:
    endpoint: str
    api_key: str
    api_version: str = "2024-02-15-preview"
    chat_deployment: Optional[str] = None  # AZURE_OPENAI_CHAT_DEPLOYMENT_NAME


class AzureChat:
    def __init__(self, cfg: Optional[AzureConfig] = None):
        if AzureOpenAI is None:
            raise RuntimeError("openai package not installed. pip install openai")

        if cfg is None:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            chat_deployment = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME")
            if not endpoint or not api_key or not chat_deployment:
                raise RuntimeError(
                    "Missing Azure env vars. Set AZURE_OPENAI_ENDPOINT, "
                    "AZURE_OPENAI_API_KEY, AZURE_OPENAI_LLM_DEPLOYMENT_NAME"
                )
            cfg = AzureConfig(
                endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                chat_deployment=chat_deployment,
            )
        self.cfg = cfg
        self.client = AzureOpenAI(
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            azure_endpoint=cfg.endpoint,
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.cfg.chat_deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""



def llm_summarize_text(to_summarize: str, language: str = "English") -> str:
    chat = AzureChat()
    prompt = PROMPTS["summarize_text"].format(text=to_summarize, language=language)
    summary = chat.generate(prompt)
    return summary.strip()
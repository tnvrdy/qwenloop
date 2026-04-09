"""
Chat endpoint wrapper supporting Qwen (DashScope) and OpenAI.

DashScope is OpenAI-compatible, so we use the OpenAI SDK for both providers.

Env vars:
  QWEN_API_KEY (required for provider="qwen")
  OPENAI_API_KEY (required for provider="openai")
"""

from __future__ import annotations

import os
import random
import time

import openai
from openai import OpenAI

_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError,
              openai.APITimeoutError, openai.InternalServerError)


DASHSCOPE_BASE_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

QWEN_DEFAULT_MODEL = "qwen-plus"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

_client_cache: dict[tuple[str, str], OpenAI] = {}


def _get_client(base_url: str, api_key: str) -> OpenAI:
    key = (base_url, api_key)
    if key not in _client_cache:
        _client_cache[key] = OpenAI(api_key=api_key, base_url=base_url)
    return _client_cache[key]


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    provider: str = "qwen",
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Send messages and return the assistant's reply as a plain string.

    Args:
        messages: List of chat messages in OpenAI format.
        model: Override the default model for the chosen provider.
        temperature: Sampling temperature (0.7 default for exploration diversity).
        max_tokens: Budget for the reply.
        provider: "qwen" or "openai".
        base_url: Override the default API endpoint.
        api_key: Override the env-var API key.
    """
    if provider == "qwen":
        resolved_key = api_key or os.environ["QWEN_API_KEY"]
        resolved_url = base_url or DASHSCOPE_BASE_URL
        resolved_model = model or QWEN_DEFAULT_MODEL
    else:
        resolved_key = api_key or os.environ["OPENAI_API_KEY"]
        resolved_url = base_url or OPENAI_BASE_URL
        resolved_model = model or OPENAI_DEFAULT_MODEL

    client = _get_client(resolved_url, resolved_key)

    delay = 2.0
    MAX_ATTEMPTS = 4
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except _RETRYABLE as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            jitter = random.uniform(0, 0.5)
            print(f"[llm] {type(e).__name__}, retry {attempt+1}/3 in {delay+jitter:.1f}s")
            time.sleep(delay + jitter)
            delay *= 2
    return ""


if __name__ == "__main__":
    reply = chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello to me, Tanvi."},
        ],
        max_tokens=32,
    )
    print(repr(reply))

"""
Chat endpoint wrapper supporting Qwen (DashScope) and OpenAI.

DashScope is OpenAI-compatible, so we use the OpenAI SDK for both providers.

Env vars:
  QWEN_API_KEY (required for provider="qwen")
  OPENAI_API_KEY (required for provider="openai")
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


DASHSCOPE_BASE_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

QWEN_DEFAULT_MODEL = "qwen-plus"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


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
        messages: List of chat messages in OpenAI format, e.g. [{"role": "system", "content": "..."},
                     {"role": "user", "content": "..."}]
        model: Model name. Pass to override [QWEN, OPENAI]_DEFAULT_MODEL constant
        temperature: 0.7 to create variety for an exploration policy, can raise?
        max_tokens: Budget for the reply, pass or default to 256
        provider: "qwen" or "openai". Pass or default to "qwen"
        base_url: Pass to override [OPENAI, DASHSCOPE]_BASE_URL constant API endpoint 
                  (e.g. if you want to use dashscope's intl endpoint instead of US)
        api_key: Pass to override [OPENAI, QWEN]_API_KEY env var
    """
    if provider == "qwen":
        resolved_key = api_key or os.environ["QWEN_API_KEY"]
        resolved_url = base_url or DASHSCOPE_BASE_URL
        resolved_model = model or QWEN_DEFAULT_MODEL
    else:
        resolved_key = api_key or os.environ["OPENAI_API_KEY"]
        resolved_url = base_url or OPENAI_BASE_URL
        resolved_model = model or OPENAI_DEFAULT_MODEL

    client = OpenAI(api_key=resolved_key, base_url=resolved_url)

    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    reply = chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello to me, Tanvi."},
        ],
        max_tokens=32,
    )
    print(repr(reply))

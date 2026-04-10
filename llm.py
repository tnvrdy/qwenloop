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
from datetime import datetime, timezone
from pathlib import Path

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

    _acquire_rate_limit_slot()

    delay = 2.0
    MAX_ATTEMPTS = int(os.environ.get("LLM_MAX_ATTEMPTS", "4"))
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
                _emit_retry_telemetry(
                    event="retry_exhausted",
                    model=resolved_model,
                    provider=provider,
                    attempt=attempt + 1,
                    error_type=type(e).__name__,
                    sleep_seconds=0.0,
                )
                raise
            jitter = random.uniform(0, 0.5)
            sleep_s = delay + jitter
            _emit_retry_telemetry(
                event="retry",
                model=resolved_model,
                provider=provider,
                attempt=attempt + 1,
                error_type=type(e).__name__,
                sleep_seconds=sleep_s,
            )
            print(f"[llm] {type(e).__name__}, retry {attempt + 1}/{MAX_ATTEMPTS} in {sleep_s:.1f}s")
            time.sleep(sleep_s)
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


def _acquire_rate_limit_slot() -> None:
    qps_raw = os.environ.get("LLM_RATE_LIMIT_QPS", "").strip()
    if not qps_raw:
        return
    try:
        qps = float(qps_raw)
    except ValueError:
        return
    if qps <= 0:
        return

    state_path = Path(os.environ.get("LLM_RATE_LIMIT_STATE_FILE", "/tmp/qwenloop_llm_rate_limit.state"))
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    now = time.time()
    min_interval = 1.0 / qps

    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        try:
            import fcntl
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass

        last_ts = 0.0
        if state_path.exists():
            try:
                last_ts = float(state_path.read_text(encoding="utf-8").strip())
            except Exception:
                last_ts = 0.0

        wait_s = max(0.0, (last_ts + min_interval) - now)
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.time()
        state_path.write_text(f"{now:.6f}", encoding="utf-8")

        try:
            import fcntl
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass


def _emit_retry_telemetry(
    *,
    event: str,
    model: str,
    provider: str,
    attempt: int,
    error_type: str,
    sleep_seconds: float,
) -> None:
    path_raw = os.environ.get("LLM_RETRY_TELEMETRY_FILE", "").strip()
    if not path_raw:
        return
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "provider": provider,
        "model": model,
        "attempt": attempt,
        "error_type": error_type,
        "sleep_seconds": round(sleep_seconds, 3),
    }
    path = Path(path_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json_dumps(row) + "\n")


def json_dumps(obj: dict) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)

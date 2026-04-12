"""
Chat endpoint wrapper supporting Qwen (DashScope) and OpenAI.

DashScope is OpenAI-compatible, so we use the OpenAI SDK for both providers.

Env vars:
  QWEN_API_KEY (required for provider="qwen")
  OPENAI_API_KEY (required for provider="openai")
  GEMINI_API_KEY (required for provider="gemini")
  LLM_PROVIDER (optional default provider: qwen|openai|gemini, default=qwen)
  LLM_RATE_LIMIT_QPS (optional)
  LLM_RATE_LIMIT_MODE (optional: file_lock|process|none, default=file_lock)
  LLM_MAX_ATTEMPTS (optional, default=4)
  LLM_RETRY_TELEMETRY_FILE (optional)
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from openai import OpenAI

_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError,
              openai.APITimeoutError, openai.InternalServerError)


DASHSCOPE_BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

QWEN_DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")
OPENAI_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_client_cache: dict[tuple[str, str], OpenAI] = {}
_process_rate_lock = threading.Lock()
_process_next_allowed_ts = 0.0


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
    provider: str | None = None,
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
        provider: "qwen", "openai", or "gemini". If omitted, uses LLM_PROVIDER or "qwen".
        base_url: Override the default API endpoint.
        api_key: Override the env-var API key.
    """
    resolved_provider = (provider or os.environ.get("LLM_PROVIDER", "qwen")).strip().lower()
    if resolved_provider == "qwen":
        resolved_key = api_key or os.environ["QWEN_API_KEY"]
        resolved_url = base_url or DASHSCOPE_BASE_URL
        resolved_model = model or QWEN_DEFAULT_MODEL
    elif resolved_provider == "openai":
        resolved_key = api_key or os.environ["OPENAI_API_KEY"]
        resolved_url = base_url or OPENAI_BASE_URL
        resolved_model = model or OPENAI_DEFAULT_MODEL
    elif resolved_provider == "gemini":
        resolved_key = api_key or os.environ["GEMINI_API_KEY"]
        resolved_url = base_url or GEMINI_BASE_URL
        resolved_model = model or GEMINI_DEFAULT_MODEL
    else:
        raise ValueError(
            f"unknown provider {resolved_provider!r}; expected one of: qwen, openai, gemini"
        )

    client = _get_client(resolved_url, resolved_key)

    delay = 2.0
    MAX_ATTEMPTS = int(os.environ.get("LLM_MAX_ATTEMPTS", "4"))
    for attempt in range(MAX_ATTEMPTS):
        _acquire_rate_limit_slot()
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
                    provider=resolved_provider,
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
                provider=resolved_provider,
                attempt=attempt + 1,
                error_type=type(e).__name__,
                sleep_seconds=sleep_s,
            )
            print(f"[llm] {type(e).__name__}, retry {attempt + 1}/{MAX_ATTEMPTS} in {sleep_s:.1f}s")
            time.sleep(sleep_s)
            delay *= 2
    return ""


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
    mode = os.environ.get("LLM_RATE_LIMIT_MODE", "file_lock").strip().lower()
    if mode in {"none", "off"}:
        return
    if mode in {"process", "local"}:
        _acquire_process_local_slot(qps)
        return
    _acquire_file_lock_slot(qps)


def _acquire_process_local_slot(qps: float) -> None:
    global _process_next_allowed_ts
    min_interval = 1.0 / qps
    with _process_rate_lock:
        now = time.time()
        wait_s = max(0.0, _process_next_allowed_ts - now)
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.time()
        _process_next_allowed_ts = now + min_interval


def _acquire_file_lock_slot(qps: float) -> None:
    state_path = Path(os.environ.get("LLM_RATE_LIMIT_STATE_FILE", "/tmp/qwenloop_llm_rate_limit.state"))
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    now = time.time()
    min_interval = 1.0 / qps

    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        _flock(lock_f, lock=True)

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

        _flock(lock_f, lock=False)


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
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _flock(file_obj, *, lock: bool) -> None:
    try:
        import fcntl

        mode = fcntl.LOCK_EX if lock else fcntl.LOCK_UN
        fcntl.flock(file_obj.fileno(), mode)
    except Exception:
        return

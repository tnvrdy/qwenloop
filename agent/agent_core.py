"""
Shared step-loop logic used by both goal-directed and freeform exploration agents.
"""

from __future__ import annotations

import time

from actions import ACTION_VOCABULARY, ActionParseError, parse_action
from browser_env import BrowserEnv, Observation
from llm import chat
from trajectory_store import TrajectoryWriter


_GOAL_DIRECTED_SYSTEM_PROMPT = f"""You are an autonomous browser agent. You control a real web browser.

Each turn you will receive:
  - GOAL: the task you must complete
  - URL: the current page URL
  - PAGE: a numbered list of interactive elements on the current page
  - HISTORY: the actions you have already taken (empty on the first turn)

Your job is to decide the single next action that best makes progress toward the GOAL.

{ACTION_VOCABULARY}

Rules:
1. Output EXACTLY ONE action per reply, on a single line. Nothing else — no explanation, no preamble.
2. Use only the actions listed above. Any other output will be treated as a parse error.
3. When the GOAL is achieved, output: stop
4. If the page does not help and you cannot make progress, output: stop""".strip()


_FREEFORM_SYSTEM_PROMPT = f"""You are a curious web user browsing the internet naturally. You control a real web browser.

Each turn you will receive:
  - URL: the current page URL
  - PAGE: a numbered list of interactive elements on the current page
  - HISTORY: the actions you have already taken (empty on the first turn)

Browse as a real person would — follow links that look interesting, search for things, \
interact with the page, explore content in depth. Act naturally and purposefully. \
Do NOT just scroll repeatedly or click randomly. Each action should reflect genuine \
human curiosity or intent.

IMPORTANT: Stay on the current website. Do NOT use goto to navigate to a different \
domain. Explore the site you are already on by clicking links, using search features, \
interacting with page elements, and navigating its internal pages.

{ACTION_VOCABULARY}

Rules:
1. Output EXACTLY ONE action per reply, on a single line. Nothing else — no explanation, no preamble.
2. Use only the actions listed above. Any other output will be treated as a parse error.
3. Do NOT output stop — keep exploring for the full session.
4. Do NOT use goto to leave the current site's domain.""".strip()


def _build_user_message(
    obs_text: str,
    action_history: list[str],
    goal: str | None = None,
) -> str:
    history_block = (
        "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(action_history))
        if action_history
        else "  (none)"
    )
    parts = []
    if goal:
        parts.append(f"GOAL: {goal}\n")
    parts.append(f"HISTORY:\n{history_block}\n")
    parts.append(f"PAGE:\n{obs_text}")
    return "\n".join(parts)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def _annotate_action(parsed, raw: str, obs: Observation, env: BrowserEnv) -> str:
    if parsed.action_type == "click" and parsed.index is not None:
        label = (
            obs.element_descs[parsed.index]
            if parsed.index < len(obs.element_descs)
            else "?"
        )
        return f'click {parsed.index}  →  "{label}"  →  {env.page.url}'
    if parsed.action_type == "type" and parsed.submit:
        return f"{raw}  →  {env.page.url}"
    if parsed.action_type == "goto":
        return f"{raw}  →  {env.page.url}"
    return raw


def run_steps(
    env: BrowserEnv,
    tw: TrajectoryWriter,
    goal: str | None = None,
    model: str | None = None,
    max_steps: int = 4,
    include_raw_model_output: bool = False,
) -> str:
    """
    Core observe -> act -> record loop.

    If goal is provided, the agent works toward it (goal-directed mode).
    If goal is None, the agent browses freely (freeform mode) and stop
    actions are ignored.

    Assumes env is already navigated to a page and tw is open.
    Returns the termination reason string.
    """
    freeform = goal is None
    system_prompt = _FREEFORM_SYSTEM_PROMPT if freeform else _GOAL_DIRECTED_SYSTEM_PROMPT
    system_msg = {"role": "system", "content": system_prompt}
    action_history: list[str] = []
    consecutive_failures = 0
    consecutive_repeats = 0
    last_action: str | None = None
    step_latencies_ms: list[float] = []

    for step_num in range(max_steps):
        step_start = time.perf_counter()
        obs = env.get_text_observation()

        user_msg = {
            "role": "user",
            "content": _build_user_message(obs.text, action_history, goal=goal),
        }
        raw_full = chat([system_msg, user_msg], model=model)
        raw = _first_line(raw_full)
        print(f"  [step {step_num}] model: {raw}")

        if raw == last_action:
            consecutive_repeats += 1
        else:
            consecutive_repeats = 0
        last_action = raw

        parsed = None
        parse_error: str | None = None
        try:
            parsed = parse_action(raw)
        except ActionParseError as e:
            parse_error = str(e)
            print(f"  [step {step_num}] parse error: {parse_error!r}")

        exec_result: dict | None = None
        if parsed is not None:
            if freeform and parsed.action_type == "stop":
                print(f"  [step {step_num}] ignoring stop (freeform mode)")
                parsed = None
            else:
                exec_result = env.execute_action(parsed)
                print(f"  [step {step_num}] exec: {exec_result!r}")

        screenshot_path = tw.screenshot_path_for(step_num)
        state = env.capture_full_state(screenshot_path)

        exec_ok = bool(exec_result and exec_result.get("ok"))
        extra = {
            "parse_error": parse_error,
            "exec_error": (exec_result or {}).get("error"),
        }
        if include_raw_model_output:
            extra["raw_model_output"] = raw_full

        tw.write_step(
            step=step_num,
            state=state,
            action=raw,
            action_ok=exec_ok,
            extra=extra,
        )
        step_latencies_ms.append((time.perf_counter() - step_start) * 1000.0)

        if parsed is not None:
            if not exec_ok:
                err = (exec_result or {}).get("error", "unknown")
                history_entry = f"{raw}  [failed: {err.splitlines()[0]}]"
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                history_entry = _annotate_action(parsed, raw, obs, env)
            action_history.append(history_entry)

        if not freeform and parsed is not None and parsed.action_type == "stop":
            _record_runtime_metrics(tw, step_latencies_ms)
            return "stop"
        if consecutive_failures >= 3:
            _record_runtime_metrics(tw, step_latencies_ms)
            return "consecutive_failures"
        if consecutive_repeats >= 2:
            print(f"  [step {step_num}] stuck: repeated {raw!r} 3 times")
            _record_runtime_metrics(tw, step_latencies_ms)
            return "stuck"

    _record_runtime_metrics(tw, step_latencies_ms)
    return "max_steps"


def _record_runtime_metrics(tw: TrajectoryWriter, step_latencies_ms: list[float]) -> None:
    if not step_latencies_ms:
        return
    sorted_lats = sorted(step_latencies_ms)
    p95_idx = max(0, min(len(sorted_lats) - 1, int(len(sorted_lats) * 0.95) - 1))
    tw.add_metadata(
        {
            "runtime_metrics": {
                "steps_recorded": len(step_latencies_ms),
                "avg_step_latency_ms": round(sum(step_latencies_ms) / len(step_latencies_ms), 2),
                "p95_step_latency_ms": round(sorted_lats[p95_idx], 2),
                "max_step_latency_ms": round(sorted_lats[-1], 2),
            }
        }
    )

"""
Agent loop for autonomous exploration!!

Runs short episodes (default 4 steps) against a live browser, using Qwen to pick actions
from a constrained action vocabulary. Every step captures rich observations (AX tree, 
HTML, viewport screenshot) and writes them to a trajectory directory.

Usage:
    from agent import run_exploration_episode
    traj_dir = run_exploration_episode("https://en.wikipedia.org", "Find the article on black holes") 5

    # or from cli:
    python agent.py https://www.stanford.edu "Find the page for Ian Jones' upcoming event" 4
"""

from __future__ import annotations

import sys
from pathlib import Path

from actions import ACTION_VOCABULARY, ActionParseError, parse_action
from browser_env import BrowserEnv
from llm import chat
from trajectory_store import TrajectoryWriter


_SYSTEM_PROMPT = f"""You are an autonomous browser agent. You control a real web browser.

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


def build_user_message(
    goal: str,
    obs_text: str,
    action_history: list[str],
) -> str:
    history_block = (
        "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(action_history))
        if action_history
        else "  (none)"
    )
    return (
        f"GOAL: {goal}\n\n"
        f"HISTORY:\n{history_block}\n\n"
        f"PAGE:\n{obs_text}"
    )


def run_exploration_episode(
    url: str,
    goal: str,
    trajectories_dir: str | Path = "trajectories",
    model: str | None = None,
    max_steps: int = 4,
    headless: bool = True,
) -> Path:
    """
    Run one short exploration episode and persist the full trajectory.

    Returns the path to the trajectory directory
    (contains metadata.json, steps.jsonl, screenshots/).
    """
    system_msg = {"role": "system", "content": _SYSTEM_PROMPT}
    action_history: list[str] = []
    consecutive_failures: int = 0

    with (
        BrowserEnv(headless=headless) as env,
        TrajectoryWriter(trajectories_dir, goal=goal, start_url=url) as tw,
    ):
        env.goto(url)

        for step_num in range(max_steps):
            # 1. observe (light text obs for the prompt)
            obs = env.get_text_observation()

            # 2. build prompt and call LLM
            user_msg = {
                "role": "user",
                "content": build_user_message(goal, obs.text, action_history),
            }
            raw_full = chat([system_msg, user_msg], model=model)
            raw = _first_line(raw_full)
            print(f"[step {step_num}] model: {raw!r}")

            # 3. parse action
            parsed = None
            parse_error: str | None = None
            try:
                parsed = parse_action(raw)
            except ActionParseError as e:
                parse_error = str(e)
                print(f"[step {step_num}] parse error: {parse_error}")

            # 4. execute action (skip if parse failed)
            exec_result: dict | None = None
            if parsed is not None:
                exec_result = env.execute_action(parsed)
                print(f"[step {step_num}] exec: {exec_result}")

            # 5. capture rich state and write step
            screenshot_path = tw.screenshot_path_for(step_num)
            state = env.capture_full_state(screenshot_path)

            exec_ok = bool(exec_result and exec_result.get("ok"))
            tw.write_step(
                step=step_num,
                state=state,
                action=raw,
                action_ok=exec_ok,
                extra={
                    "parse_error": parse_error,
                    "exec_error": (exec_result or {}).get("error"),
                    "raw_model_output": raw_full,
                },
            )

            # 6. update action history with semantic context
            if parsed is not None:
                if not exec_ok:
                    err = (exec_result or {}).get("error", "unknown")
                    history_entry = f"{raw}  [failed: {err.splitlines()[0]}]"
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                    history_entry = _annotate_action(parsed, raw, obs, env)
                action_history.append(history_entry)

            # 7. termination checks
            if parsed is not None and parsed.action_type == "stop":
                tw.set_termination_reason("stop")
                print(f"[episode] agent stopped at step {step_num}")
                break
            if consecutive_failures >= 3:
                tw.set_termination_reason("consecutive_failures")
                print(f"[episode] bailing: {consecutive_failures} consecutive failures")
                break
        else:
            tw.set_termination_reason("max_steps")
            print(f"[episode] reached max_steps ({max_steps})")

        return tw.traj_dir


def _first_line(text: str) -> str:
    """Extract the first non-empty line from model output."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def _annotate_action(parsed, raw: str, obs, env) -> str:
    """
    Build a history entry with semantic context so the LLM
    can reason about what each past action actually did.
    """
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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python agent.py <url> <goal> [max_steps]")
        sys.exit(1)

    _url = sys.argv[1]
    _goal = sys.argv[2]
    _max = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    traj = run_exploration_episode(_url, _goal, max_steps=_max, headless=False)
    print(f"\nTrajectory saved: {traj}")

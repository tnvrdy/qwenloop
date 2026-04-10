"""
Freeform exploration agent — explore first, label after.

The agent browses WITHOUT any pre-specified goal, acting as a curious human
user would. After each exploration episode, a separate LLM pass reviews
what happened and decides whether anything "meaningful" occurred. If so,
it labels the trajectory with a retroactive goal description. If not,
the trajectory is marked as discarded.

Each micro-episode is saved as its own trajectory directory (same format as
goal-directed runs), so downstream consumers (judge, decomposer) work
identically on the labeled ones.

Usage:
    python agent/agent_freeform.py https://github.com/explore --episodes 5
    python agent/agent_freeform.py https://en.wikipedia.org --episodes 10 --max-steps 8
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from agent.agent_core import run_steps
from browser_env import BrowserEnv
from collection_config import CollectionIOConfig, resolve_io_config
from llm import chat
from trajectory_store import TrajectoryWriter, load_trajectory, update_metadata
from agent.agent_goaldirected import _validate_max_steps

# Retroactive labeling prompt

_LABEL_PROMPT = """\
You are reviewing a short browsing session to decide if the user accomplished \
anything meaningful.

Here is the sequence of actions the user took:

{steps_summary}

A trajectory is "meaningful" if the user accomplished something a real person \
would describe as a task or activity, for example:
  - Searched for a specific topic and found relevant results
  - Navigated to a specific page or section of a website
  - Compared products, articles, or options
  - Used a site feature (filters, sorting, settings, forms)
  - Found specific information (a price, a date, a fact, a recipe)
  - Explored a category or topic in depth across multiple pages

A trajectory is NOT meaningful if:
  - The user just scrolled around aimlessly without finding anything
  - The user clicked one or two links without any coherent purpose
  - All actions failed or resulted in errors
  - The user stayed on the same page doing nothing substantive

If the trajectory IS meaningful, respond with a JSON object:
  {{"meaningful": true, "goal": "<concise goal describing what to accomplish>"}}

The goal should be phrased as an INSTRUCTION — what a user would set out to do, not \
a past-tense description of what happened. Use imperative/infinitive form.
  GOOD: "Find the Wikipedia article on quantum entanglement and read the experimental evidence section"
  GOOD: "Search for wireless headphones under $50 and compare the top two results"
  BAD:  "Found the Wikipedia article on quantum entanglement" (past tense)
  BAD:  "Clicked on search, typed quantum, clicked first result" (narrating steps)

If the trajectory is NOT meaningful, respond with:
  {{"meaningful": false, "goal": null}}

Return ONLY the JSON object, no other text."""


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _summarize_step_for_label(step: dict) -> str:
    action = step.get("action", "(no action)")
    ok = step.get("action_ok", False)
    url = step.get("url", "")
    title = step.get("title", "")
    status = "" if ok else " [FAILED]"
    return f"  Action: {action}{status}\n    → URL: {url}\n    → Title: {title}"


def label_trajectory(traj_dir: str | Path, model: str | None = None) -> dict:
    """
    Retroactively label a trajectory by reviewing what happened.

    Returns {"meaningful": bool, "goal": str | None}.
    Also writes the label into the trajectory's metadata.json.
    """
    traj = load_trajectory(traj_dir)
    steps = traj["steps"]

    if not steps:
        result = {"meaningful": False, "goal": None}
        update_metadata(traj_dir, {"goal": "(empty)", "label_result": result})
        return result

    steps_summary = "\n\n".join(
        f"Step {i}:\n{_summarize_step_for_label(s)}"
        for i, s in enumerate(steps)
    )

    prompt = _LABEL_PROMPT.format(steps_summary=steps_summary)
    raw = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=256,
    )

    cleaned = _strip_markdown_fences(raw)
    try:
        obj = json.loads(cleaned)
        meaningful = bool(obj.get("meaningful", False))
        goal = obj.get("goal", "").strip() if meaningful else None
    except (json.JSONDecodeError, ValueError, TypeError):
        meaningful = False
        goal = None

    result = {"meaningful": meaningful, "goal": goal}

    if meaningful and goal:
        update_metadata(traj_dir, {"goal": goal, "label_result": result})
    else:
        update_metadata(traj_dir, {"goal": "(unlabeled)", "label_result": result})

    return result


# freeform session: N micro-episodes on one browser

def run_freeform_session(
    seed_url: str,
    num_episodes: int = 5,
    trajectories_dir: str | Path = "trajectories",
    model: str | None = None,
    max_steps: int = 30,
    headless: bool = True,
    label_mode: str = "deferred",
    writer_flush_every: int = 1,
    writer_async: bool = False,
    writer_queue_size: int = 256,
    compress_heavy: bool = False,
    include_raw_model_output: bool = False,
    io_config: CollectionIOConfig | None = None,
) -> list[Path]:
    """
    Run a continuous freeform exploration session.

    Opens one browser, navigates to seed_url, then loops:
      1. Explore goallessly for max_steps actions
      2. Save the raw trajectory
      3. Retroactively label: did anything meaningful happen?
      4. Stay on whatever page the agent ended up on
      5. Repeat

    Returns a list of trajectory directory paths (one per micro-episode).
    Only meaningful trajectories get a real goal label; others are marked
    "(unlabeled)" but still saved for potential future use.
    """
    traj_dirs: list[Path] = []
    meaningful_count = 0
    max_steps = _validate_max_steps(max_steps)
    cfg = resolve_io_config(
        io_config,
        writer_flush_every=writer_flush_every,
        writer_async=writer_async,
        writer_queue_size=writer_queue_size,
        compress_heavy=compress_heavy,
        include_raw_model_output=include_raw_model_output,
    )

    with BrowserEnv(headless=headless) as env:
        env.goto(seed_url)
        print(f"[freeform] seed: {seed_url}")
        print(f"[freeform] planning {num_episodes} episodes, up to {max_steps} steps each\n")

        for ep in range(num_episodes):
            current_url = env.page.url
            print(f"[freeform ep {ep + 1}/{num_episodes}] exploring from {current_url} ...")

            with TrajectoryWriter(
                trajectories_dir,
                goal="(unlabeled)",
                start_url=current_url,
                flush_every=cfg.flush_every,
                async_writer=cfg.async_writer,
                queue_size=cfg.queue_size,
                compress_heavy=cfg.compress_heavy,
            ) as tw:
                ep_start = time.perf_counter()
                reason = run_steps(
                    env,
                    tw,
                    goal=None,
                    model=model,
                    max_steps=max_steps,
                    include_raw_model_output=cfg.include_raw_model_output,
                )
                tw.add_metadata({"collection_elapsed_seconds": round(time.perf_counter() - ep_start, 3)})
                tw.set_termination_reason(reason)
                traj_dirs.append(tw.traj_dir)
                traj_path = tw.traj_dir

            if label_mode == "inline":
                print(f"[freeform ep {ep + 1}/{num_episodes}] exploration done ({reason}), labeling ...")
                label = label_trajectory(traj_path, model=model)
                if label["meaningful"]:
                    meaningful_count += 1
                    print(f"[freeform ep {ep + 1}/{num_episodes}] MEANINGFUL: {label['goal']!r}")
                else:
                    print(f"[freeform ep {ep + 1}/{num_episodes}] not meaningful, kept as (unlabeled)")
            else:
                print(f"[freeform ep {ep + 1}/{num_episodes}] exploration done ({reason}), labeling deferred")

            print(f"[freeform ep {ep + 1}/{num_episodes}] now at {env.page.url}\n")

    if label_mode == "inline":
        print(f"[freeform] session complete: {len(traj_dirs)} trajectories, "
              f"{meaningful_count} meaningful")
    else:
        print(f"[freeform] session complete: {len(traj_dirs)} trajectories, labeling not run inline")
    return traj_dirs


def label_trajectories_batch(
    traj_dirs: list[str | Path],
    model: str | None = None,
) -> dict:
    meaningful = 0
    errors = 0
    for i, td in enumerate(traj_dirs):
        try:
            result = label_trajectory(td, model=model)
            if result.get("meaningful"):
                meaningful += 1
            print(f"[label {i + 1}/{len(traj_dirs)}] {Path(td).name}: {'meaningful' if result.get('meaningful') else 'unlabeled'}")
        except Exception as e:
            errors += 1
            print(f"[label {i + 1}/{len(traj_dirs)}] {Path(td).name}: ERROR {e}")
    return {
        "total": len(traj_dirs),
        "meaningful": meaningful,
        "errors": errors,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Freeform exploration (explore first, label after)")
    parser.add_argument("seed_url", help="Starting URL")
    parser.add_argument("--episodes", type=int, default=5, help="Number of micro-episodes")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per episode (hard cap: 50)")
    parser.add_argument("-o", "--output", default="trajectories", help="Output directory")
    parser.add_argument("--model", default=None, help="LLM model override")
    parser.add_argument("--headed", action="store_true", help="Run in headed mode (visible browser)")
    parser.add_argument(
        "--label-mode",
        choices=["inline", "deferred"],
        default="deferred",
        help="inline = label right after each episode; deferred = collect only",
    )
    args = parser.parse_args()

    dirs = run_freeform_session(
        seed_url=args.seed_url,
        num_episodes=args.episodes,
        trajectories_dir=args.output,
        model=args.model,
        max_steps=args.max_steps,
        headless=not args.headed,
        label_mode=args.label_mode,
    )

    if args.label_mode == "deferred":
        print("\nLabeling deferred; run label_trajectories_batch(...) or orchestrator --label-freeform.")
    else:
        print(f"\nTrajectories saved:")
        for d in dirs:
            meta = json.loads((d / "metadata.json").read_text())
            tag = "✓" if meta.get("label_result", {}).get("meaningful") else "·"
            print(f"  {tag} {d.name}  goal={meta.get('goal', '?')!r}")

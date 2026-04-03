"""
LLM judge for trajectory quality filtering

Reads collected trajectories, calls Qwen to score each one for coherence and goal-directedness 
on a 1-5 scale, and writes the result back into the trajectory directory as a judge_result.json

Usage:
    from judge import judge_trajectory, judge_all_trajectories

    result = judge_trajectory("trajectories/traj_20260402_123456_ab12cd")
    summary = judge_all_trajectories("trajectories/", threshold=3)

    # or CLI:
    python judge.py trajectories/ --threshold 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm import chat
from trajectory_store import iter_trajectories, load_trajectory


JUDGE_RESULT_FILE = "judge_result.json"

_AX_TREE_PREVIEW_LINES = 40

_JUDGE_PROMPT = """\
You are evaluating the quality of a short browser exploration trajectory.

The agent was given the following goal:
  "{goal}"

Starting URL: {start_url}

Here is the sequence of actions the agent took, along with a summary of what it observed at each step:

{steps_text}

Rate this trajectory on a scale of 1-5:
  1 = Incoherent or random actions with no relation to the goal or each other
  2 = Mostly random, maybe one relevant action
  3 = Partially coherent — some actions make sense but the sequence is clumsy or incomplete
  4 = Good — actions are logical and make reasonable progress toward the goal
  5 = Excellent — a clean, efficient sequence that clearly advances the goal

Output ONLY a single integer (1-5). Nothing else."""


def _summarize_step(step: dict) -> str:
    """
    Build a concise text summary of one step for the judge prompt.
    Includes the action, resulting URL/title, and a short ax tree preview
    """
    action = step.get("action", "(no action)")
    ok = step.get("action_ok", False)
    url = step.get("url", "")
    title = step.get("title", "")

    status = "" if ok else " [FAILED]"
    header = f"Action: {action}{status}\n  → URL: {url}\n  → Title: {title}"

    ax_tree = step.get("ax_tree", "")
    if ax_tree:
        lines = ax_tree.splitlines()[:_AX_TREE_PREVIEW_LINES]
        preview = "\n".join(f"    {l}" for l in lines)
        if len(ax_tree.splitlines()) > _AX_TREE_PREVIEW_LINES:
            preview += "\n    ..."
        header += f"\n  Page structure:\n{preview}"

    return header


def _build_steps_text(steps: list[dict]) -> str:
    parts = []
    for i, step in enumerate(steps):
        parts.append(f"Step {i}:\n{_summarize_step(step)}")
    return "\n\n".join(parts)


def _parse_score(response: str) -> int:
    """
    Extract an integer score from the LLM response:
    pull all digits, parse as int, clamp to [1, 5]
    """
    digits = "".join(filter(str.isdigit, response))
    if not digits:
        return 1
    score = int(digits[0])
    return max(1, min(5, score))


def judge_trajectory(
    traj_dir: str | Path,
    threshold: int = 3,
    model: str | None = None,
) -> dict:
    """
    Judge a single trajectory and write the result to judge_result.json inside trajectory's dir

    Returns dict w keys:
      - trajectory_id: str (traj dir name)
      - score: int (1-5)
      - pass: bool (whether score >= threshold)
      - threshold: int (minimum score to pass, default 3)
      - response: str (LLM response, stripped)
      - prompt_length: int
    """
    traj_dir = Path(traj_dir)
    traj = load_trajectory(traj_dir)
    meta = traj["metadata"]
    steps = traj["steps"]

    goal = meta.get("goal", "(unknown goal)")
    start_url = meta.get("start_url", "(unknown)")
    traj_id = meta.get("trajectory_id", traj_dir.name)

    if not steps:
        result = {
            "trajectory_id": traj_id,
            "score": 1,
            "pass": False,
            "response": "(no steps to judge)",
        }
        _write_result(traj_dir, result)
        return result

    steps_text = _build_steps_text(steps)
    prompt = _JUDGE_PROMPT.format(
        goal=goal,
        start_url=start_url,
        steps_text=steps_text,
    )

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=16,
    )

    score = _parse_score(response)
    passed = score >= threshold

    result = {
        "trajectory_id": traj_id,
        "score": score,
        "pass": passed,
        "threshold": threshold,
        "response": response.strip(),
        "prompt_length": len(prompt),
    }
    _write_result(traj_dir, result)
    return result

def judge_all_trajectories(
    base_dir: str | Path,
    threshold: int = 3,
    model: str | None = None,
    force: bool = False,
) -> dict:
    """
    Judge all trajectories under base_dir. Skips already-judged trajectories unless force=True.

    Returns dict w keys:
      - total_judged: int (total trajectories judged)
      - passed: int (total trajectories passed)
      - failed: int (total trajectories failed)
      - skipped: int (total trajectories skipped)
      - errors: int (total errors encountered)
    """
    base_dir = Path(base_dir)
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    errors = 0

    trajs = list(iter_trajectories(base_dir))
    print(f"Judging {len(trajs)} trajectories in {base_dir}")

    for i, (traj_id, traj_path) in enumerate(trajs):
        result_path = traj_path / JUDGE_RESULT_FILE

        if result_path.exists() and not force:
            skipped += 1
            continue

        total += 1
        try:
            result = judge_trajectory(traj_path, threshold=threshold, model=model)
            if result["pass"]:
                passed += 1
            else:
                failed += 1
            print(f"  [{i + 1}/{len(trajs)}] {traj_id}: score={result['score']} {'PASS' if result['pass'] else 'FAIL'}")
        except Exception as e:
            errors += 1
            print(f"  [{i + 1}/{len(trajs)}] {traj_id}: ERROR {e}")

    summary = {
        "total_judged": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }
    print(f"\nSummary: {summary}")
    return summary


def _write_result(traj_dir: Path, result: dict) -> None:
    path = traj_dir / JUDGE_RESULT_FILE
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge collected trajectories")
    parser.add_argument("trajectories_dir", help="Base directory of collected trajectories")
    parser.add_argument("--threshold", type=int, default=3, help="Minimum score to pass (1-5)")
    parser.add_argument("--model", default=None, help="LLM model override")
    parser.add_argument("--force", action="store_true", help="Re-judge already-judged trajectories")
    args = parser.parse_args()

    judge_all_trajectories(
        args.trajectories_dir,
        threshold=args.threshold,
        model=args.model,
        force=args.force,
    )

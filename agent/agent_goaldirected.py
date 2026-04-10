"""
Goal-directed exploration agent.

Runs episodes against a live browser, using Qwen to pick actions from a
constrained action vocabulary. Supports both single-episode and batching.

Usage:
    python agent/agent_goaldirected.py https://www.stanford.edu "Find the next poetry reading event" 6
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from agent.agent_core import run_steps
from browser_env import BrowserEnv
from collection_config import CollectionIOConfig, resolve_io_config
from io_utils import dir_size_bytes
from trajectory_store import TrajectoryWriter, load_trajectory


def _validate_max_steps(max_steps: int) -> int:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if max_steps > 50:
        raise ValueError("max_steps must be <= 50 (hard cap)")
    return max_steps


def run_exploration_episode(
    url: str,
    goal: str,
    trajectories_dir: str | Path = "trajectories",
    model: str | None = None,
    max_steps: int = 25,
    headless: bool = True,
    writer_flush_every: int = 1,
    writer_async: bool = False,
    writer_queue_size: int = 256,
    compress_heavy: bool = False,
    include_raw_model_output: bool = False,
    io_config: CollectionIOConfig | None = None,
) -> Path:
    """Run one episode with its own browser. Fine for testing, not for scale."""
    max_steps = _validate_max_steps(max_steps)
    cfg = resolve_io_config(
        io_config,
        writer_flush_every=writer_flush_every,
        writer_async=writer_async,
        writer_queue_size=writer_queue_size,
        compress_heavy=compress_heavy,
        include_raw_model_output=include_raw_model_output,
    )
    with (
        BrowserEnv(headless=headless) as env,
        TrajectoryWriter(
            trajectories_dir,
            goal=goal,
            start_url=url,
            flush_every=cfg.flush_every,
            async_writer=cfg.async_writer,
            queue_size=cfg.queue_size,
            compress_heavy=cfg.compress_heavy,
        ) as tw,
    ):
        env.goto(url)
        print(f"[episode] goal={goal!r}  url={url}")
        reason = run_steps(
            env,
            tw,
            goal=goal,
            model=model,
            max_steps=max_steps,
            include_raw_model_output=cfg.include_raw_model_output,
        )
        tw.set_termination_reason(reason)
        print(f"[episode] done ({reason})")
        return tw.traj_dir


def run_task_batch(
    tasks: list[dict],
    trajectories_dir: str | Path = "trajectories",
    model: str | None = None,
    max_steps: int = 20,
    headless: bool = True,
    writer_flush_every: int = 1,
    writer_async: bool = False,
    writer_queue_size: int = 256,
    compress_heavy: bool = False,
    include_raw_model_output: bool = False,
    io_config: CollectionIOConfig | None = None,
) -> list[dict]:
    """
    Run a batch of (url, goal) tasks on a single persistent browser.

    Each task produces its own trajectory directory. The browser
    navigates to the task URL at the start of each episode.

    Returns a list of result dicts (one per task)
    """
    results: list[dict] = []
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
        for i, task in enumerate(tasks):
            url, goal = task["url"], task["goal"]
            seed_source = task.get("seed_source")
            try:
                env.goto(url)
                print(f"[batch {i + 1}/{len(tasks)}] goal={goal!r}  url={url}")

                ep_start = time.perf_counter()
                with TrajectoryWriter(
                    trajectories_dir,
                    goal=goal,
                    start_url=url,
                    flush_every=cfg.flush_every,
                    async_writer=cfg.async_writer,
                    queue_size=cfg.queue_size,
                    compress_heavy=cfg.compress_heavy,
                ) as tw:
                    if seed_source:
                        tw.add_metadata({"seed_source": seed_source})
                    reason = run_steps(
                        env,
                        tw,
                        goal=goal,
                        model=model,
                        max_steps=max_steps,
                        include_raw_model_output=cfg.include_raw_model_output,
                    )
                    tw.set_termination_reason(reason)
                    traj_dir = tw.traj_dir

                elapsed_s = time.perf_counter() - ep_start
                meta = load_trajectory(traj_dir, include_heavy=False)["metadata"]

                results.append({
                    "status": "ok",
                    "trajectory_id": meta.get("trajectory_id", traj_dir.name),
                    "num_steps": meta.get("num_steps", 0),
                    "termination_reason": meta.get("termination_reason", "unknown"),
                    "elapsed_seconds": round(elapsed_s, 3),
                    "bytes_written": dir_size_bytes(traj_dir),
                    "runtime_metrics": meta.get("runtime_metrics", {}),
                })
                print(f"[batch {i + 1}/{len(tasks)}] done ({reason})")

            except Exception as e:
                results.append({
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "task": task,
                })
                print(f"[batch {i + 1}/{len(tasks)}] ERROR: {e}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python agent/agent_goaldirected.py <url> <goal> [max_steps]")
        sys.exit(1)

    _url = sys.argv[1]
    _goal = sys.argv[2]
    _max = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    traj = run_exploration_episode(_url, _goal, max_steps=_max, headless=False)
    print(f"\nTrajectory saved: {traj}")

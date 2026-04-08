"""
Parallel orchestrator for exploration episodes.

Two subcommands:
    python orchestrator.py tasks tasks.jsonl -o trajectories/ -w 8
    python orchestrator.py freeform -o trajectories/ -w 4 --episodes 5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _run_task_batch(tasks, trajectories_dir, model, max_steps):
    try:
        from agent_goaldirected import run_task_batch
        results = run_task_batch(tasks, trajectories_dir=trajectories_dir,
                                 model=model, max_steps=max_steps, headless=True)
        ok = [r for r in results if r["status"] == "ok"]
        return {
            "status": "ok",
            "completed": len(ok),
            "errors": len(results) - len(ok),
            "total_steps": sum(r.get("num_steps", 0) for r in ok),
        }
    except Exception as e:
        return {"status": "error", "error_type": type(e).__name__, "error": str(e)}


def _run_freeform(seed_url, num_episodes, trajectories_dir, model, max_steps):
    try:
        from agent_freeform import run_freeform_session
        traj_dirs = run_freeform_session(seed_url=seed_url, num_episodes=num_episodes,
                                          trajectories_dir=trajectories_dir, model=model,
                                          max_steps=max_steps, headless=True)
        from trajectory_store import load_trajectory
        meaningful = sum(
            1 for td in traj_dirs
            if load_trajectory(td)["metadata"].get("label_result", {}).get("meaningful")
        )
        total_steps = sum(
            load_trajectory(td)["metadata"].get("num_steps", 0) for td in traj_dirs
        )
        return {
            "status": "ok",
            "seed_url": seed_url,
            "num_trajectories": len(traj_dirs),
            "meaningful": meaningful,
            "total_steps": total_steps,
        }
    except Exception as e:
        return {"status": "error", "error_type": type(e).__name__, "error": str(e)}


def _chunk(lst, n):
    k = math.ceil(len(lst) / n) if n > 0 else len(lst)
    return [lst[i:i + k] for i in range(0, len(lst), k)] if lst else []


def _load_tasks(path, limit=None):
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
                if limit and len(tasks) >= limit:
                    break
    return tasks


def _write_summary(summary, trajectories_dir):
    p = Path(trajectories_dir) / "run_summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2) + "\n")
    return p


def run_tasks(
    tasks_path: str | Path,
    trajectories_dir: str | Path = "trajectories",
    max_workers: int = 4,
    max_steps: int = 4,
    model: str | None = None,
    limit: int | None = None,
) -> dict:
    """
    Run exploration episodes in parallel.
    Supports task-driven, freeform, or mixed mode.
    """
    trajectories_dir = str(Path(trajectories_dir).resolve())
    tasks = _load_tasks(tasks_path, limit=limit)
    n_workers = min(max_workers, len(tasks))
    chunks = _chunk(tasks, n_workers)

    print(f"Orchestrator: {len(tasks)} tasks, {n_workers} workers (batch size ~{len(chunks[0]) if chunks else 0})")
    print(f"Output: {trajectories_dir}\n")

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_task_batch, ch, trajectories_dir, model, max_steps): i
            for i, ch in enumerate(chunks)
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            done = len(results)
            if r["status"] == "ok":
                print(f"  [{done}/{len(chunks)}] {r['completed']} done, {r['errors']} errors, {r['total_steps']} steps")
            else:
                print(f"  [{done}/{len(chunks)}] ERROR: {r['error'][:80]}")

    elapsed = time.time() - t0
    completed = sum(r.get("completed", 0) for r in results if r["status"] == "ok")
    errors = sum(r.get("errors", 0) for r in results if r["status"] == "ok")
    total_steps = sum(r.get("total_steps", 0) for r in results if r["status"] == "ok")

    summary = {
        "mode": "tasks",
        "total_tasks": len(tasks),
        "completed": completed,
        "errors": errors,
        "avg_steps": round(total_steps / completed, 1) if completed else 0,
        "elapsed_seconds": round(elapsed, 1),
        "trajectories_per_minute": round(completed / (elapsed / 60), 1) if elapsed > 0 else 0,
    }
    sp = _write_summary(summary, trajectories_dir)

    print(f"\n{'=' * 50}")
    print(f"  Completed:  {completed}/{len(tasks)} ({errors} errors)")
    print(f"  Avg steps:  {summary['avg_steps']}")
    print(f"  Time:       {summary['elapsed_seconds']}s ({summary['trajectories_per_minute']} traj/min)")
    print(f"  Summary:    {sp}")
    print(f"{'=' * 50}")
    return summary


def run_freeform(
    trajectories_dir: str | Path = "trajectories",
    max_workers: int = 4,
    episodes_per_worker: int = 5,
    max_steps: int = 8,
    model: str | None = None,
    seeds: list[str] | None = None,
) -> dict:
    trajectories_dir = str(Path(trajectories_dir).resolve())
    if seeds is None:
        from task_generator import DEFAULT_SEEDS
        seeds = [s["url"] for s in DEFAULT_SEEDS]

    print(f"Orchestrator: {max_workers} freeform workers, {episodes_per_worker} episodes each")
    print(f"Output: {trajectories_dir}\n")

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_freeform, seeds[i % len(seeds)], episodes_per_worker,
                        trajectories_dir, model, max_steps): i
            for i in range(max_workers)
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            done = len(results)
            if r["status"] == "ok":
                print(f"  [{done}/{max_workers}] {r['meaningful']}/{r['num_trajectories']} meaningful, "
                      f"{r['total_steps']} steps from {r['seed_url']}")
            else:
                print(f"  [{done}/{max_workers}] ERROR: {r['error'][:80]}")

    elapsed = time.time() - t0
    ok = [r for r in results if r["status"] == "ok"]
    total_trajs = sum(r["num_trajectories"] for r in ok)
    total_meaningful = sum(r["meaningful"] for r in ok)
    total_steps = sum(r["total_steps"] for r in ok)

    summary = {
        "mode": "freeform",
        "workers": max_workers,
        "total_trajectories": total_trajs,
        "meaningful": total_meaningful,
        "avg_steps": round(total_steps / total_trajs, 1) if total_trajs else 0,
        "elapsed_seconds": round(elapsed, 1),
        "trajectories_per_minute": round(total_trajs / (elapsed / 60), 1) if elapsed > 0 else 0,
    }
    sp = _write_summary(summary, trajectories_dir)

    print(f"\n{'=' * 50}")
    print(f"  Trajectories: {total_trajs} ({total_meaningful} meaningful)")
    print(f"  Avg steps:    {summary['avg_steps']}")
    print(f"  Time:         {summary['elapsed_seconds']}s ({summary['trajectories_per_minute']} traj/min)")
    print(f"  Summary:      {sp}")
    print(f"{'=' * 50}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel exploration orchestrator")
    sub = parser.add_subparsers(dest="mode", required=True)

    tp = sub.add_parser("tasks", help="Run goal-directed episodes from a tasks JSONL")
    tp.add_argument("tasks_file", help="Path to tasks JSONL")
    tp.add_argument("-o", "--output", default="trajectories")
    tp.add_argument("-w", "--workers", type=int, default=4)
    tp.add_argument("--max-steps", type=int, default=4)
    tp.add_argument("--model", default=None)
    tp.add_argument("--limit", type=int, default=None)
    tp.add_argument("--judge", action="store_true")
    tp.add_argument("--threshold", type=int, default=3)

    fp = sub.add_parser("freeform", help="Run goalless freeform exploration")
    fp.add_argument("-o", "--output", default="trajectories")
    fp.add_argument("-w", "--workers", type=int, default=4)
    fp.add_argument("--episodes", type=int, default=5, help="Episodes per worker")
    fp.add_argument("--max-steps", type=int, default=8)
    fp.add_argument("--model", default=None)
    fp.add_argument("--judge", action="store_true")
    fp.add_argument("--threshold", type=int, default=3)

    args = parser.parse_args()

    if args.mode == "tasks":
        run_tasks(
            tasks_path=args.tasks_file,
            trajectories_dir=args.output,
            max_workers=args.workers,
            max_steps=args.max_steps,
            model=args.model,
            limit=args.limit,
        )
    else:
        run_freeform(
            trajectories_dir=args.output,
            max_workers=args.workers,
            episodes_per_worker=args.episodes,
            max_steps=args.max_steps,
            model=args.model,
        )

    if args.judge:
        print("\nRunning judge on collected trajectories...\n")
        from judge import judge_all_trajectories
        judge_all_trajectories(args.output, threshold=args.threshold, model=args.model)

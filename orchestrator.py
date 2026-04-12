"""
Parallel orchestrator for exploration episodes.

Two subcommands:
    python orchestrator.py tasks tasks.jsonl -o trajectories/ -w 8
    python orchestrator.py freeform -o trajectories/ -w 4 --episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from utils.collection_config import CollectionIOConfig, resolve_io_config
from utils.io_utils import dir_size_bytes
from agent.agent_goaldirected import _validate_max_steps
from trajectory_store import load_trajectory_metadata


def _run_task_batch(
    tasks,
    trajectories_dir,
    model,
    max_steps,
    headless,
    collect_size_metrics,
    io_config: CollectionIOConfig,
):
    try:
        from agent.agent_goaldirected import run_task_batch
        results = run_task_batch(
            tasks,
            trajectories_dir=trajectories_dir,
            model=model,
            max_steps=max_steps,
            headless=headless,
            collect_size_metrics=collect_size_metrics,
            io_config=io_config,
        )
        ok = [r for r in results if r["status"] == "ok"]
        step_latencies = []
        for row in ok:
            rm = row.get("runtime_metrics") or {}
            if rm.get("avg_step_latency_ms") is not None:
                step_latencies.append(rm["avg_step_latency_ms"])
        return {
            "status": "ok",
            "completed": len(ok),
            "errors": len(results) - len(ok),
            "total_steps": sum(r.get("num_steps", 0) for r in ok),
            "bytes_written": sum(r.get("bytes_written", 0) for r in ok),
            "step_latency_samples_ms": step_latencies,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _run_freeform(
    seed_url,
    num_episodes,
    trajectories_dir,
    model,
    max_steps,
    headless,
    collect_size_metrics,
    return_trajectory_dirs,
    io_config: CollectionIOConfig,
):
    try:
        from agent.agent_freeform import run_freeform_session
        traj_dirs = run_freeform_session(
            seed_url=seed_url,
            num_episodes=num_episodes,
            trajectories_dir=trajectories_dir,
            model=model,
            max_steps=max_steps,
            headless=headless,
            label_mode="deferred",
            io_config=io_config,
        )
        meaningful = 0
        total_steps = 0
        for td in traj_dirs:
            meta = load_trajectory_metadata(td)
            if meta.get("label_result", {}).get("meaningful"):
                meaningful += 1
            total_steps += int(meta.get("num_steps", 0))
        total_bytes = sum(dir_size_bytes(Path(td)) for td in traj_dirs) if collect_size_metrics else 0
        return {
            "status": "ok",
            "seed_url": seed_url,
            "num_trajectories": len(traj_dirs),
            "meaningful": meaningful,
            "total_steps": total_steps,
            "bytes_written": total_bytes,
            "trajectory_dirs": [str(td) for td in traj_dirs] if return_trajectory_dirs else [],
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _chunk(lst, n):
    if not lst:
        return []
    if n <= 0:
        return [lst]
    k = (len(lst) + n - 1) // n
    return [lst[i:i + k] for i in range(0, len(lst), k)]


def _validate_workers(max_workers: int) -> int:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    return max_workers


def _validate_worker_backend(worker_backend: str) -> str:
    backend = (worker_backend or "").strip().lower()
    if backend not in {"process", "thread"}:
        raise ValueError("worker_backend must be one of: process, thread")
    return backend


def _load_tasks(path, limit=None):
    tasks = []
    skipped_invalid = 0
    normalized = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                task = json.loads(line)
                raw_url = str(task.get("url", "")).strip()
                fixed_url = _normalize_task_url(raw_url)
                if fixed_url is None:
                    skipped_invalid += 1
                    continue
                if fixed_url != raw_url:
                    normalized += 1
                task["url"] = fixed_url
                tasks.append(task)
                if limit and len(tasks) >= limit:
                    break
    if normalized:
        print(f"Orchestrator: normalized {normalized} task URL(s)")
    if skipped_invalid:
        print(f"Orchestrator: skipped {skipped_invalid} invalid task URL(s)")
    return tasks


def _normalize_task_url(url: str) -> str | None:
    if not url:
        return None
    raw = url.strip()
    candidate = raw if "://" in raw else f"https://{raw}"
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        return None

    host = (parsed.hostname or "").strip().lower()
    if not host:
        return None

    # common task artifact: bare host without TLD, e.g. "https://apple/"
    # reject short/noisy placeholders!!!!
    if "." not in host and host not in {"localhost"}:
        if len(host) < 4:
            return None
        host = f"{host}.com"

    if "." not in host and host != "localhost":
        return None

    netloc = host
    if parsed.port is not None:
        netloc = f"{host}:{parsed.port}"
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        netloc = f"{auth}@{netloc}"

    normalized = parsed._replace(scheme=parsed.scheme.lower(), netloc=netloc)
    return urlunparse(normalized)


def _write_summary(summary, trajectories_dir):
    p = Path(trajectories_dir) / "run_summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2) + "\n")
    return p


def _configure_llm_limits(trajectories_dir: str | Path, llm_qps: float | None) -> None:
    if llm_qps is None:
        return
    os.environ["LLM_RATE_LIMIT_QPS"] = str(llm_qps)
    os.environ.setdefault("LLM_RETRY_TELEMETRY_FILE", str(Path(trajectories_dir) / "llm_retry_telemetry.jsonl"))


def _build_io_config(
    *,
    trajectories_dir: str | Path,
    writer_flush_every: int,
    writer_async: bool,
    writer_queue_size: int,
    compress_heavy: bool,
    include_raw_model_output: bool,
    screenshot_every_n_steps: int,
    scale_mode: bool,
    llm_qps: float | None,
) -> CollectionIOConfig:
    _configure_llm_limits(trajectories_dir, llm_qps)
    if scale_mode:
        writer_flush_every = max(writer_flush_every, 16)
        writer_async = True
        compress_heavy = True
    return resolve_io_config(
        None,
        writer_flush_every=writer_flush_every,
        writer_async=writer_async,
        writer_queue_size=writer_queue_size,
        compress_heavy=compress_heavy,
        include_raw_model_output=include_raw_model_output,
        screenshot_every_n_steps=screenshot_every_n_steps,
    )


def _maybe_quality_report(trajectories_dir: str | Path) -> dict | None:
    try:
        from judge import summarize_collection_quality

        return summarize_collection_quality(trajectories_dir)
    except Exception:
        return None


def _retry_telemetry_summary(trajectories_dir: str | Path) -> dict | None:
    telemetry_path = Path(trajectories_dir) / "llm_retry_telemetry.jsonl"
    if not telemetry_path.exists():
        return None
    retries = 0
    exhausted = 0
    error_types: dict[str, int] = {}
    with open(telemetry_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = row.get("event")
            err = row.get("error_type", "unknown")
            if event == "retry":
                retries += 1
                error_types[err] = error_types.get(err, 0) + 1
            elif event == "retry_exhausted":
                exhausted += 1
                error_types[err] = error_types.get(err, 0) + 1
    return {
        "telemetry_path": str(telemetry_path),
        "retry_events": retries,
        "retry_exhausted_events": exhausted,
        "error_types": error_types,
    }


def run_tasks(
    tasks_path: str | Path,
    trajectories_dir: str | Path = "trajectories",
    max_workers: int = 4,
    max_steps: int = 25,
    model: str | None = None,
    limit: int | None = None,
    headless: bool = True,
    writer_flush_every: int = 8,
    writer_async: bool = True,
    writer_queue_size: int = 256,
    compress_heavy: bool = True,
    include_raw_model_output: bool = False,
    llm_qps: float | None = None,
    screenshot_every_n_steps: int = 1,
    scale_mode: bool = False,
    collect_size_metrics: bool | None = None,
    worker_backend: str = "process",
) -> dict:
    """
    Run goal-directed exploration episodes in parallel.
    """
    trajectories_dir = str(Path(trajectories_dir).resolve())
    max_workers = _validate_workers(max_workers)
    worker_backend = _validate_worker_backend(worker_backend)
    max_steps = _validate_max_steps(max_steps)
    if collect_size_metrics is None:
        collect_size_metrics = not scale_mode
    io_config = _build_io_config(
        trajectories_dir=trajectories_dir,
        writer_flush_every=writer_flush_every,
        writer_async=writer_async,
        writer_queue_size=writer_queue_size,
        compress_heavy=compress_heavy,
        include_raw_model_output=include_raw_model_output,
        screenshot_every_n_steps=screenshot_every_n_steps,
        scale_mode=scale_mode,
        llm_qps=llm_qps,
    )
    tasks = _load_tasks(tasks_path, limit=limit)
    if not tasks:
        print("Orchestrator: 0 tasks loaded; nothing to run.")
        summary = {
            "mode": "tasks",
            "total_tasks": 0,
            "completed": 0,
            "errors": 0,
            "avg_steps": 0,
            "elapsed_seconds": 0.0,
            "trajectories_per_minute": 0,
            "steps_per_second": 0,
            "bytes_written_mb": 0.0,
            "write_mb_per_second": 0,
            "avg_step_latency_ms": None,
            "p95_step_latency_ms": None,
            "quality_report": None,
        }
        _write_summary(summary, trajectories_dir)
        return summary
    n_workers = min(max_workers, len(tasks))
    chunks = _chunk(tasks, n_workers)

    print(f"Orchestrator: {len(tasks)} tasks, {n_workers} workers (batch size ~{len(chunks[0]) if chunks else 0})")
    print(f"Worker backend: {worker_backend}")
    print(f"Output: {trajectories_dir}\n")

    results = []
    worker_failures: list[dict] = []
    t0 = time.time()

    executor_cls = ProcessPoolExecutor if worker_backend == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _run_task_batch,
                ch,
                trajectories_dir,
                model,
                max_steps,
                headless,
                collect_size_metrics,
                io_config,
            ): i
            for i, ch in enumerate(chunks)
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                r = future.result()
            except Exception as e:
                r = {
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "worker_id": worker_id,
                }
            results.append(r)
            done = len(results)
            if r["status"] == "ok":
                print(f"  [{done}/{len(chunks)}] {r['completed']} done, {r['errors']} errors, {r['total_steps']} steps")
            else:
                worker_failures.append(r)
                print(f"  [{done}/{len(chunks)}] ERROR(worker={worker_id}): {r['error'][:120]}")

    elapsed = time.time() - t0
    completed = sum(r.get("completed", 0) for r in results if r["status"] == "ok")
    errors = sum(r.get("errors", 0) for r in results if r["status"] == "ok")
    total_steps = sum(r.get("total_steps", 0) for r in results if r["status"] == "ok")
    total_bytes = sum(r.get("bytes_written", 0) for r in results if r["status"] == "ok")
    lat_samples = []
    for r in results:
        if r["status"] == "ok":
            lat_samples.extend(r.get("step_latency_samples_ms", []))
    lat_samples.sort()

    summary = {
        "mode": "tasks",
        "worker_backend": worker_backend,
        "total_tasks": len(tasks),
        "completed": completed,
        "errors": errors,
        "avg_steps": round(total_steps / completed, 1) if completed else 0,
        "elapsed_seconds": round(elapsed, 1),
        "trajectories_per_minute": round(completed / (elapsed / 60), 1) if elapsed > 0 else 0,
        "steps_per_second": round(total_steps / elapsed, 2) if elapsed > 0 else 0,
        "bytes_written_mb": round(total_bytes / (1024 * 1024), 2),
        "write_mb_per_second": round((total_bytes / (1024 * 1024)) / elapsed, 2) if elapsed > 0 else 0,
        "avg_step_latency_ms": round(sum(lat_samples) / len(lat_samples), 2) if lat_samples else None,
        "p95_step_latency_ms": _p95(lat_samples),
        "worker_failures": worker_failures,
    }
    summary["llm_retry_telemetry"] = _retry_telemetry_summary(trajectories_dir)
    summary["quality_report"] = None if scale_mode else _maybe_quality_report(trajectories_dir)
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
    max_steps: int = 30,
    model: str | None = None,
    seeds: list[str] | None = None,
    headless: bool = True,
    label_freeform: bool = False,
    writer_flush_every: int = 8,
    writer_async: bool = True,
    writer_queue_size: int = 256,
    compress_heavy: bool = True,
    include_raw_model_output: bool = False,
    llm_qps: float | None = None,
    screenshot_every_n_steps: int = 1,
    scale_mode: bool = False,
    collect_size_metrics: bool | None = None,
    worker_backend: str = "process",
) -> dict:
    trajectories_dir = str(Path(trajectories_dir).resolve())
    max_workers = _validate_workers(max_workers)
    worker_backend = _validate_worker_backend(worker_backend)
    max_steps = _validate_max_steps(max_steps)
    if collect_size_metrics is None:
        collect_size_metrics = not scale_mode
    io_config = _build_io_config(
        trajectories_dir=trajectories_dir,
        writer_flush_every=writer_flush_every,
        writer_async=writer_async,
        writer_queue_size=writer_queue_size,
        compress_heavy=compress_heavy,
        include_raw_model_output=include_raw_model_output,
        screenshot_every_n_steps=screenshot_every_n_steps,
        scale_mode=scale_mode,
        llm_qps=llm_qps,
    )
    if seeds is None:
        from task_generation.task_generator import DEFAULT_SEEDS
        seeds = [s["url"] for s in DEFAULT_SEEDS]
    if not seeds:
        raise ValueError("freeform seeds list cannot be empty")
    if episodes_per_worker < 1:
        raise ValueError("episodes_per_worker must be >= 1")

    print(f"Orchestrator: {max_workers} freeform workers, {episodes_per_worker} episodes each")
    print(f"Worker backend: {worker_backend}")
    print(f"Output: {trajectories_dir}\n")

    results = []
    worker_failures: list[dict] = []
    t0 = time.time()

    executor_cls = ProcessPoolExecutor if worker_backend == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_freeform,
                seeds[i % len(seeds)],
                episodes_per_worker,
                trajectories_dir,
                model,
                max_steps,
                headless,
                collect_size_metrics,
                label_freeform,
                io_config,
            ): i
            for i in range(max_workers)
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                r = future.result()
            except Exception as e:
                r = {
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "worker_id": worker_id,
                }
            results.append(r)
            done = len(results)
            if r["status"] == "ok":
                print(f"  [{done}/{max_workers}] {r['meaningful']}/{r['num_trajectories']} meaningful, "
                      f"{r['total_steps']} steps from {r['seed_url']}")
            else:
                worker_failures.append(r)
                print(f"  [{done}/{max_workers}] ERROR(worker={worker_id}): {r['error'][:120]}")

    elapsed = time.time() - t0
    ok = [r for r in results if r["status"] == "ok"]
    total_trajs = sum(r["num_trajectories"] for r in ok)
    total_meaningful = sum(r["meaningful"] for r in ok)
    total_steps = sum(r["total_steps"] for r in ok)
    total_bytes = sum(r.get("bytes_written", 0) for r in ok)
    all_traj_dirs = []
    for r in ok:
        all_traj_dirs.extend(r.get("trajectory_dirs", []))

    label_summary = None
    if label_freeform and all_traj_dirs:
        from agent.agent_freeform import label_trajectories_batch
        print("\nRunning deferred freeform labeling...")
        label_summary = label_trajectories_batch(all_traj_dirs, model=model)
        total_meaningful = label_summary["meaningful"]

    summary = {
        "mode": "freeform",
        "worker_backend": worker_backend,
        "workers": max_workers,
        "total_trajectories": total_trajs,
        "meaningful": total_meaningful,
        "avg_steps": round(total_steps / total_trajs, 1) if total_trajs else 0,
        "elapsed_seconds": round(elapsed, 1),
        "trajectories_per_minute": round(total_trajs / (elapsed / 60), 1) if elapsed > 0 else 0,
        "steps_per_second": round(total_steps / elapsed, 2) if elapsed > 0 else 0,
        "bytes_written_mb": round(total_bytes / (1024 * 1024), 2),
        "write_mb_per_second": round((total_bytes / (1024 * 1024)) / elapsed, 2) if elapsed > 0 else 0,
        "labeling": label_summary,
        "worker_failures": worker_failures,
    }
    summary["llm_retry_telemetry"] = _retry_telemetry_summary(trajectories_dir)
    summary["quality_report"] = None if scale_mode else _maybe_quality_report(trajectories_dir)
    sp = _write_summary(summary, trajectories_dir)

    print(f"\n{'=' * 50}")
    print(f"  Trajectories: {total_trajs} ({total_meaningful} meaningful)")
    print(f"  Avg steps:    {summary['avg_steps']}")
    print(f"  Time:         {summary['elapsed_seconds']}s ({summary['trajectories_per_minute']} traj/min)")
    print(f"  Summary:      {sp}")
    print(f"{'=' * 50}")
    return summary


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    idx = max(0, min(len(values) - 1, int(len(values) * 0.95) - 1))
    return round(values[idx], 2)


def _add_common_collection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--worker-backend",
        choices=["process", "thread"],
        default="process",
        help="Parallel worker backend. Use thread for lightweight local runs, process for stronger isolation.",
    )
    p.add_argument("--headed", action="store_true")
    p.add_argument("--writer-flush-interval", type=int, default=8)
    p.add_argument("--llm-qps", type=float, default=None)
    p.add_argument("--writer-queue-size", type=int, default=256)
    p.add_argument("--sync-writer", action="store_true")
    p.add_argument("--no-compress-heavy", action="store_true")
    p.add_argument("--screenshot-every", type=int, default=1, help="Capture screenshot every N steps (0 disables)")
    p.add_argument("--scale-mode", action="store_true", help="Apply high-throughput profile for large shard runs")
    size_group = p.add_mutually_exclusive_group()
    size_group.add_argument(
        "--collect-size-metrics",
        dest="collect_size_metrics",
        action="store_true",
        help="Compute per-trajectory disk size metrics (adds filesystem overhead)",
    )
    size_group.add_argument(
        "--skip-size-metrics",
        dest="collect_size_metrics",
        action="store_false",
        help="Skip per-trajectory disk size scans",
    )
    p.set_defaults(collect_size_metrics=None)
    p.add_argument("--include-raw-model-output", action="store_true")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--threshold", type=int, default=3)


def _common_run_kwargs(args) -> dict:
    return {
        "headless": not args.headed,
        "writer_flush_every": args.writer_flush_interval,
        "writer_async": not args.sync_writer,
        "writer_queue_size": args.writer_queue_size,
        "compress_heavy": not args.no_compress_heavy,
        "screenshot_every_n_steps": args.screenshot_every,
        "scale_mode": args.scale_mode,
        "collect_size_metrics": args.collect_size_metrics,
        "include_raw_model_output": args.include_raw_model_output,
        "llm_qps": args.llm_qps,
        "worker_backend": args.worker_backend,
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Parallel exploration orchestrator. For cluster runs, always use an "
            "isolated output directory per shard/job to avoid shared-write hotspots."
        )
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    tp = sub.add_parser("tasks", help="Run goal-directed episodes from a tasks JSONL")
    tp.add_argument("tasks_file", help="Path to tasks JSONL")
    tp.add_argument("-o", "--output", default="trajectories")
    tp.add_argument("-w", "--workers", type=int, default=4)
    tp.add_argument("--max-steps", type=int, default=25)
    tp.add_argument("--model", default=None)
    tp.add_argument("--limit", type=int, default=None)
    _add_common_collection_args(tp)

    fp = sub.add_parser("freeform", help="Run goalless freeform exploration")
    fp.add_argument("-o", "--output", default="trajectories")
    fp.add_argument("-w", "--workers", type=int, default=4)
    fp.add_argument("--episodes", type=int, default=5, help="Episodes per worker")
    fp.add_argument("--max-steps", type=int, default=30)
    fp.add_argument("--model", default=None)
    fp.add_argument("--label-freeform", action="store_true", help="Run freeform labeling after collection")
    _add_common_collection_args(fp)

    args = parser.parse_args()
    common_kwargs = _common_run_kwargs(args)

    if args.mode == "tasks":
        run_tasks(
            tasks_path=args.tasks_file,
            trajectories_dir=args.output,
            max_workers=args.workers,
            max_steps=args.max_steps,
            model=args.model,
            limit=args.limit,
            **common_kwargs,
        )
    else:
        run_freeform(
            trajectories_dir=args.output,
            max_workers=args.workers,
            episodes_per_worker=args.episodes,
            max_steps=args.max_steps,
            model=args.model,
            label_freeform=args.label_freeform,
            **common_kwargs,
        )

    if args.judge:
        print("\nRunning judge on collected trajectories...\n")
        from judge import judge_all_trajectories
        judge_all_trajectories(args.output, threshold=args.threshold, model=args.model, max_workers=args.workers)

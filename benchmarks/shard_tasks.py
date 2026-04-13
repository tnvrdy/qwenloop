#!/usr/bin/env python3
"""
Deterministically split a tasks.jsonl into shard files.

Example:
  python benchmarks/shard_tasks.py \
    --tasks-file tasks.jsonl \
    --num-shards 32 \
    --output-dir runs/2026-04-02/shards
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_tasks(tasks_file: Path, num_shards: int, output_dir: Path, prefix: str = "tasks_shard") -> dict:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [output_dir / f"{prefix}_{i:03d}.jsonl" for i in range(num_shards)]
    shard_files = [p.open("w", encoding="utf-8") for p in shard_paths]
    counts = [0 for _ in range(num_shards)]
    total = 0
    try:
        with tasks_file.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                _ = json.loads(line)
                shard_id = line_no % num_shards
                shard_files[shard_id].write(line + "\n")
                counts[shard_id] += 1
                total += 1
    finally:
        for handle in shard_files:
            handle.close()

    manifest = {
        "tasks_file": str(tasks_file.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_shards": num_shards,
        "total_tasks": total,
        "shards": [
            {"shard_id": i, "tasks_file": str(shard_paths[i].resolve()), "num_tasks": counts[i]}
            for i in range(num_shards)
        ],
    }
    manifest_path = output_dir / "shard_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Split tasks JSONL into deterministic shard files")
    parser.add_argument("--tasks-file", required=True, help="Source tasks JSONL")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of deterministic output shards")
    parser.add_argument("--output-dir", required=True, help="Destination directory for shard files")
    parser.add_argument("--prefix", default="tasks_shard", help="Shard filename prefix")
    args = parser.parse_args()

    manifest = split_tasks(
        tasks_file=Path(args.tasks_file),
        num_shards=args.num_shards,
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
    )
    print(
        f"Wrote {manifest['num_shards']} shards with {manifest['total_tasks']} total tasks to "
        f"{manifest['output_dir']}"
    )


if __name__ == "__main__":
    main()

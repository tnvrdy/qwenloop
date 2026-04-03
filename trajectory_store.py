"""
Raw trajectory storage for exploration episodes.

Layout on disk:
    <base_dir>/
      traj_<id>/
        metadata.json
        steps.jsonl # one JSON object per line, one per step
        screenshots/
          step_000.png
          step_001.png
          ...

Purely raw data capture, wrangling into chatml/training formats will happen downstream.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class TrajectoryWriter:
    """
    Accumulates steps for a single trajectory and writes them to disk.

    Usage:
        with TrajectoryWriter(base_dir, goal=goal, start_url=url) as tw:
            tw.write_step(step=0, state=state, action="click 3", action_ok=True)
            tw.write_step(step=1, ...)
        # metadata.json is finalized on __exit__
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        goal: str,
        start_url: str,
        trajectory_id: str | None = None,
    ) -> None:
        self.trajectory_id = trajectory_id or _make_id()
        self.goal = goal
        self.start_url = start_url

        self.traj_dir = Path(base_dir) / self.trajectory_id
        self.screenshots_dir = self.traj_dir / "screenshots"
        self.steps_path = self.traj_dir / "steps.jsonl"
        self.metadata_path = self.traj_dir / "metadata.json"

        self._step_count = 0
        self._start_time = _now_iso()
        self._termination_reason: str = "unknown"
        self._steps_file = None

    def __enter__(self) -> TrajectoryWriter:
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)
        self._steps_file = open(self.steps_path, "a")
        return self

    def __exit__(self, *_: Any) -> None:
        if self._steps_file:
            self._steps_file.close()
        self._write_metadata()


    def screenshot_path_for(self, step: int) -> Path:
        """Return the screenshot path for a given step number."""
        return self.screenshots_dir / f"step_{step:03d}.png"

    def write_step(
        self,
        *,
        step: int,
        state: dict,
        action: str,
        action_ok: bool,
        extra: dict | None = None,
    ) -> None:
        """
        Persist one step.

        Args:
            step: 0-based step index
            state: dict from BrowserEnv.capture_full_state(). must contain
                   "url", "title", "ax_tree", "html", "screenshot_path"
            action: raw action string the agent produced
            action_ok: whether the action executed successfully
            extra: any additional fields to include in the step record
        """
        record: dict[str, Any] = {
            "step": step,
            "timestamp": _now_iso(),
            "url": state.get("url", ""),
            "title": state.get("title", ""),
            "ax_tree": state.get("ax_tree", ""),
            "html": state.get("html", ""),
            "screenshot_path": state.get("screenshot_path", ""),
            "action": action,
            "action_ok": action_ok,
        }
        if extra:
            record.update(extra)

        self._steps_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._steps_file.flush()
        self._step_count += 1

    def set_termination_reason(self, reason: str) -> None:
        self._termination_reason = reason


    def _write_metadata(self) -> None:
        meta = {
            "trajectory_id": self.trajectory_id,
            "goal": self.goal,
            "start_url": self.start_url,
            "num_steps": self._step_count,
            "start_time": self._start_time,
            "end_time": _now_iso(),
            "termination_reason": self._termination_reason,
        }
        self.metadata_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
        )


def _make_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid4().hex[:6]
    return f"traj_{ts}_{short}"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

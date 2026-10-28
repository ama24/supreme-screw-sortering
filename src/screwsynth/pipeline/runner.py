"""Run a plan: each command in its own process, output streamed line by line, stoppable.

State goes to output/<name>/pipeline_status.json and the full output to output/<name>/pipeline.log,
so the GUI and the CLI can show what happened in an earlier session.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from screwsynth.common.io import write_json, write_text, write_yaml
from screwsynth.common.paths import repo_root

from .steps import Plan, Step, status_path

log = logging.getLogger("screwsynth.pipeline")

LineSink = Callable[[str], None]


class Runner:
    """Executes a Plan. run() blocks; call it from a thread to keep a GUI responsive."""

    def __init__(self, plan: Plan, on_line: LineSink | None = None) -> None:
        self.plan = plan
        self.on_line = on_line or (lambda line: None)
        self.status: dict[str, Any] = {
            "recipe": plan.recipe.name,
            "state": "idle",
            "steps": {s.key: {"state": "pending"} for s in plan.steps},
        }
        self._proc: subprocess.Popen[str] | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._log_file: Any = None

    # -------------------------------------------------------------- control

    @property
    def running(self) -> bool:
        return self.status["state"] == "running"

    def cancel(self) -> None:
        """Stop after killing the current command (and its children, e.g. Blender)."""
        self._cancel.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)

    def progress(self) -> dict[str, tuple[int, int]]:
        """(done, total) per step: finished commands, or the step's own counter while running."""
        out = {}
        for s in self.plan.steps:
            st = self.status["steps"][s.key]
            if st["state"] == "running" and s.progress is not None:
                out[s.key] = s.progress()
            else:
                out[s.key] = (st.get("commands_done", 0), st.get("commands_total", 0))
        return out

    # ------------------------------------------------------------------ run

    def run(self, fresh: bool = False) -> bool:
        """All steps in order; stops at the first failure. True if every step succeeded."""
        plan = self.plan
        plan.run_dir.mkdir(parents=True, exist_ok=True)
        if fresh:
            self._fresh()
        write_yaml(plan.config_path, plan.config)  # every step reads this merged config
        self._log_file = (plan.run_dir / "pipeline.log").open("a", encoding="utf-8")
        self.status.update(state="running", started=_now(), finished=None)
        self._emit(f"=== {plan.recipe.name}: {len(plan.steps)} step(s), config {plan.config_path}")
        self._save()
        write_environment(plan.run_dir / "environment.txt")
        ok = True
        try:
            for step in plan.steps:
                if self._cancel.is_set():
                    self.status["steps"][step.key]["state"] = "cancelled"
                    continue
                if not self._run_step(step):
                    ok = False
                    break
        finally:
            for s in plan.steps:  # steps after a failure never started
                if self.status["steps"][s.key]["state"] == "pending":
                    self.status["steps"][s.key]["state"] = "skipped"
            self.status["state"] = (
                "cancelled" if self._cancel.is_set() else ("done" if ok else "failed")
            )
            self.status["finished"] = _now()
            self._emit(f"=== {self.status['state']}")
            self._save()
            self._log_file.close()
            self._log_file = None
        return ok and not self._cancel.is_set()

    def _run_step(self, step: Step) -> bool:
        st = self.status["steps"][step.key]
        st.update(state="running", started=_now())
        for path in step.clean:
            if path.exists():
                self._emit(f"--- removing previous output {path}")
                shutil.rmtree(path, ignore_errors=True)
        cmds = step.commands()
        st.update(commands_total=len(cmds), commands_done=0)
        self._emit(f"\n=== [{step.key}] {step.title}")
        self._save()
        t0 = time.perf_counter()
        for cmd in cmds:
            code = self._run_command(cmd)
            if self._cancel.is_set():
                st.update(state="cancelled", seconds=round(time.perf_counter() - t0, 1))
                return False
            if code != 0:
                st.update(
                    state="failed",
                    returncode=code,
                    failed_command=cmd,
                    seconds=round(time.perf_counter() - t0, 1),
                )
                self._emit(f"!!! [{step.key}] failed with exit code {code}: {' '.join(cmd)}")
                return False
            st["commands_done"] += 1
            self._save()
        st.update(state="done", finished=_now(), seconds=round(time.perf_counter() - t0, 1))
        self._emit(f"=== [{step.key}] done in {st['seconds']:.0f} s")
        return True

    def _run_command(self, cmd: Sequence[str]) -> int:
        self._emit("$ " + " ".join(cmd))
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(
                list(cmd),
                cwd=repo_root(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **kwargs,
            )
        except OSError as exc:
            self._emit(f"!!! cannot start {cmd[0]}: {exc}")
            return 127
        with self._lock:
            self._proc = proc
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._emit(line.rstrip("\r\n"))
            return proc.wait()
        finally:
            with self._lock:
                self._proc = None

    # -------------------------------------------------------------- helpers

    def _fresh(self) -> None:
        """Delete this dataset's previous render/postprocess/label outputs (not the shared assets)."""
        run_dir = self.plan.run_dir
        for sub in ("scenes", "processed", "previews"):
            shutil.rmtree(run_dir / sub, ignore_errors=True)
        for f in ("balance_state.json", "run_config.yaml", "failed.jsonl", "log.txt"):
            (run_dir / f).unlink(missing_ok=True)
        for f in run_dir.glob("show_*.png"):
            f.unlink()
        shutil.rmtree(self.plan.dataset_dir, ignore_errors=True)
        self._emit(
            f"--- start fresh: removed previous outputs of {run_dir} and {self.plan.dataset_dir}"
        )

    def _emit(self, line: str) -> None:
        if self._log_file is not None:
            self._log_file.write(line + "\n")
            self._log_file.flush()
        self.on_line(line)

    def _save(self) -> None:
        write_json(status_path(self.plan.run_dir), self.status)


def write_environment(path: Path) -> None:
    """Git commit, Python and every installed package version, so a dataset can be reproduced."""
    root = repo_root()

    def run(*cmd: str) -> str:
        try:
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=60)
            return r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "?"

    commit = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    lines = [
        f"date: {_now()}",
        f"git commit: {commit}{' (with uncommitted changes)' if dirty else ''}",
        f"python: {sys.version.split()[0]} ({sys.executable})",
        f"platform: {platform.platform()}",
        "",
        "# pip freeze",
        run(sys.executable, "-m", "pip", "freeze", "--exclude-editable"),
    ]
    write_text(path, "\n".join(lines) + "\n")


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def tail(path: Path, n: int = 400) -> list[str]:
    """Last n lines of a text file (earlier pipeline.log in the GUI)."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()[-n:]

"""Generic JSONL adapter.

One trajectory per file. Each line is a JSON object describing one step.
Recognised keys (all optional except ``action`` or ``command``):

``thought``       free text the agent wrote before acting
``action``        the tool or action name, for example ``edit``, ``bash``
``command``       the command string, for example ``pytest tests/``
``observation``   what came back from the environment
``exit_code``     integer exit status of the command
``timestamp``     seconds since the epoch, float
``tokens_in``     prompt tokens for the step
``tokens_out``    completion tokens for the step
``tool``          optional tool class (edit, test, read, search, shell,
                  submit, other); when given it is used as is, otherwise the
                  class is derived from ``action`` and ``command``

A line with ``"type": "meta"`` carries run level fields: ``run_id``,
``task_id``, ``model``, ``patch`` (the final diff as a string),
``exit_status`` (the agent framework's own end state, a string) and
``resolved`` (true or false: whether the task's tests passed on the patch). A line with
``"type": "patch"`` and a ``patch`` key is also accepted.

The ``action`` string is classified into a tool class by :func:`classify_tool`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trajlens.model import (
    EDIT,
    OTHER,
    READ,
    SEARCH,
    SHELL,
    SUBMIT,
    TEST,
    TOOL_CLASSES,
    Step,
    Trajectory,
)

_TEST_RE = re.compile(
    r"(^|[\s/;&|])(pytest|py\.test|unittest|tox|nox|jest|vitest|mocha|cargo\s+test|go\s+test|"
    r"npm\s+(run\s+)?test|yarn\s+test|pnpm\s+test|make\s+test|phpunit|rspec|gradle\s+test|"
    r"mvn\s+test|ctest)\b"
)
# Running a test or reproduction script directly, for example
# ``python reproduce.py`` or ``python3 -u tests/test_io.py``. The
# SWE-agent default prompt tells the agent to write a reproduction script and
# rerun it after the fix, so for that family this is the usual check.
_SCRIPT_TEST_RE = re.compile(
    r"(^|[\s/;&|])python[0-9.]*\s+(-\S+\s+)*(\S*/)?"
    r"(test_\w*|\w*_test|tests?|reproduce\w*|repro\w*)\.py\b"
)
_EDIT_RE = re.compile(
    r"^\s*(edit|str_replace_editor|str_replace_based_edit_tool|create|insert|"
    r"write|write_file|apply_patch|sed\s+-i|patch\b|git\s+apply)"
)
_READ_RE = re.compile(r"^\s*(open|view|cat|head|tail|less|read|read_file|goto|scroll_(up|down))\b")
_SEARCH_RE = re.compile(r"^\s*(grep|rg|ag|find|search|search_dir|search_file|find_file|ls|fd)\b")
_SUBMIT_RE = re.compile(r"^\s*(submit|finish|exit|done)\b")

_ACTION_CLASS = {
    "edit": EDIT,
    "str_replace_editor": EDIT,
    "str_replace_based_edit_tool": EDIT,
    "write": EDIT,
    "write_file": EDIT,
    "create": EDIT,
    "insert": EDIT,
    "test": TEST,
    "run_tests": TEST,
    "read": READ,
    "read_file": READ,
    "open": READ,
    "view": READ,
    "search": SEARCH,
    "find": SEARCH,
    "grep": SEARCH,
    "bash": SHELL,
    "shell": SHELL,
    "run": SHELL,
    "execute": SHELL,
    "submit": SUBMIT,
    "finish": SUBMIT,
}


def classify_tool(action: str, command: str) -> str:
    """Map an action name and its command to one tool class.

    The action name wins when it is a known editor, reader or submit action.
    A shell action is looked at by its command: a test runner, or a python
    test or reproduction script (``test_*.py``, ``*_test.py``,
    ``reproduce*.py``, ``repro*.py``), is ``test``, an
    in-place edit is ``edit``, a pager or cat is ``read``, grep/find is
    ``search``, anything else is ``shell``.
    """
    name = (action or "").strip().lower()
    cmd = (command or "").strip()
    known = _ACTION_CLASS.get(name)
    if known is not None and known != SHELL:
        return known
    text = cmd if cmd else name
    if _TEST_RE.search(text) or _SCRIPT_TEST_RE.search(text):
        return TEST
    if _EDIT_RE.match(text):
        return EDIT
    if _SUBMIT_RE.match(text):
        return SUBMIT
    if _READ_RE.match(text):
        return READ
    if _SEARCH_RE.match(text):
        return SEARCH
    if known == SHELL or cmd:
        return SHELL
    return OTHER


def _opt_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def step_from_record(index: int, rec: dict) -> Step:
    """Build one Step from a generic JSONL record."""
    action = str(rec.get("action", "") or "")
    command = str(rec.get("command", "") or "")
    tool = rec.get("tool")
    if tool not in TOOL_CLASSES:
        tool = classify_tool(action, command)
    return Step(
        index=index,
        tool=tool,
        command=command,
        thought=str(rec.get("thought", "") or ""),
        observation=str(rec.get("observation", "") or ""),
        exit_code=_opt_int(rec.get("exit_code")),
        timestamp=_opt_float(rec.get("timestamp")),
        tokens_in=_opt_int(rec.get("tokens_in")),
        tokens_out=_opt_int(rec.get("tokens_out")),
        raw_action=action,
    )


def load_generic(path: str | Path) -> Trajectory:
    """Read one generic JSONL trajectory file."""
    path = Path(path)
    traj = Trajectory(run_id=path.stem, source_format="generic")
    index = 0
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: not valid JSON: {exc}") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            kind = rec.get("type", "step")
            if kind == "meta":
                traj.run_id = str(rec.get("run_id", traj.run_id))
                traj.task_id = str(rec.get("task_id", ""))
                traj.model = str(rec.get("model", ""))
                if "patch" in rec:
                    traj.patch = str(rec["patch"] or "")
                if rec.get("exit_status") is not None:
                    traj.extra["exit_status"] = str(rec["exit_status"])
                if isinstance(rec.get("resolved"), bool):
                    traj.resolved = rec["resolved"]
                continue
            if kind == "patch":
                traj.patch = str(rec.get("patch", "") or "")
                continue
            traj.steps.append(step_from_record(index, rec))
            index += 1
    return traj


def step_record(step: Step) -> dict:
    """The generic JSONL record of one step; empty and unknown fields are left out."""
    rec: dict = {"action": step.raw_action, "tool": step.tool, "command": step.command}
    for key in ("thought", "observation"):
        if getattr(step, key):
            rec[key] = getattr(step, key)
    for key in ("exit_code", "timestamp", "tokens_in", "tokens_out"):
        if getattr(step, key) is not None:
            rec[key] = getattr(step, key)
    return rec


def dump_generic(traj: Trajectory, path: str | Path) -> None:
    """Write one trajectory as a generic JSONL file that :func:`load_generic` reads back."""
    meta: dict = {"type": "meta", "run_id": traj.run_id}
    if traj.task_id:
        meta["task_id"] = traj.task_id
    if traj.model:
        meta["model"] = traj.model
    if traj.extra.get("exit_status"):
        meta["exit_status"] = traj.extra["exit_status"]
    if traj.resolved is not None:
        meta["resolved"] = traj.resolved
    meta["patch"] = traj.patch
    with Path(path).open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta) + "\n")
        for step in traj.steps:
            fh.write(json.dumps(step_record(step)) + "\n")

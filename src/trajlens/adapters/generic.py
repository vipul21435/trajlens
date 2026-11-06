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
``resolved`` (true/false or 1/0: whether the task's tests passed on the patch). A
missing or null field keeps its default (the file name for ``run_id``,
empty otherwise). A line with
``"type": "patch"`` and a ``patch`` key is also accepted.

The ``action`` string is classified into a tool class by :func:`classify_tool`.
"""

from __future__ import annotations

import json
import math
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

# A test runner as the command word of one simple command. Leading variable
# assignments and wrappers (env, time, timeout N, uv run, npx, python -m,
# coverage run -m, ...) are skipped first, so ``PYTHONPATH=. python -m pytest``
# is a test run while ``pip install pytest`` or ``grep pytest`` is not.
_RUNNER_PREFIX_RE = re.compile(
    r"^(?:(?:[A-Za-z_]\w*=\S*|env|time|nice|xvfb-run|timeout\s+\S+)\s+)*"
    r"(?:(?:uv|poetry|pipenv|pdm|hatch)\s+run\s+(?:-\S+\s+)*|npx\s+|bundle\s+exec\s+)?"
    r"(?:(?:\S*/)?python[0-9.]*\s+(?:-[A-Za-z]+\s+)*-m\s+|coverage\s+run\s+(?:-\S+\s+)*-m\s+)?"
)
_RUNNER_RE = re.compile(
    r"^(?:\S*/)?(pytest|py\.test|unittest|tox|nox|jest|vitest|mocha|cargo\s+test|go\s+test|"
    r"npm\s+(run\s+)?test|yarn\s+test|pnpm\s+test|make\s+test|phpunit|rspec|gradle\s+test|"
    r"\./gradlew\s+test|mvn\s+test|ctest)\b"
    r"|^(?:(?:\S*/)?python[0-9.]*\s+(?:-\S+\s+)*)?(?:\S*/)?(?:manage\.py\s+test|runtests\.py)\b"
)
_SEGMENT_RE = re.compile(r"&&|\|\||[;|\n]")


def runs_test_runner(command: str) -> bool:
    """True when some simple command in ``command`` starts with a test runner."""
    for part in _SEGMENT_RE.split(command or ""):
        part = part.strip().lstrip("\"'(").strip()
        if part.startswith(("bash -c ", "sh -c ")):
            part = part.split(None, 2)[2].lstrip("\"'").strip() if len(part.split()) > 2 else ""
        rest = _RUNNER_PREFIX_RE.sub("", part, count=1)
        if _RUNNER_RE.match(rest):
            return True
    return False


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
_CD_PREFIX_RE = re.compile(r"^\s*cd\s+(\"[^\"]*\"|'[^']*'|\S+)\s*(&&|;)\s*")
_SUBMIT_RE = re.compile(r"^\s*(submit|finish|exit|done)\b")
# A shell write: cat, echo, printf or tee whose first line sends output to a
# file, for example ``cat > check.py << 'EOF'`` or ``echo x >> notes.txt``.
# ``2>&1`` and writes to /dev/null do not count.
_WRITER_RE = re.compile(r"^\s*(cat|echo|printf|tee)\b")
_OUT_REDIRECT_RE = re.compile(r"(?<![0-9&<>])(?:1)?>>?\s*(?!&|/dev/null\b)[\"']?[^\s&|;<>\"']")
_TEE_TARGET_RE = re.compile(r"^\s*tee\s+(?:-\S+\s+)*(?!/dev/null\b)[\"']?[^\s&|;<>\"'-]")

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
    "search_dir": SEARCH,
    "search_file": SEARCH,
    "find_file": SEARCH,
    "bash": SHELL,
    "shell": SHELL,
    "run": SHELL,
    "execute": SHELL,
    "submit": SUBMIT,
    "finish": SUBMIT,
}


# The file editor tool of SWE-agent 1.x and of tool-calling agents takes a
# sub-command: ``view`` only reads, the others (create, str_replace, insert,
# undo_edit) change files.
EDITOR_ACTIONS = ("str_replace_editor", "str_replace_based_edit_tool")


def _editor_class(name: str, cmd: str) -> str | None:
    """READ or EDIT for a file-editor call, None when the step is not one.

    The sub-command is the token after the editor name in ``cmd``, or the
    first token of ``cmd`` when the action name alone names the editor.
    """
    tokens = cmd.split()
    if tokens and tokens[0].lower() in EDITOR_ACTIONS:
        tokens = tokens[1:]
    elif name not in EDITOR_ACTIONS:
        return None
    sub = tokens[0].lower() if tokens else ""
    return READ if sub == "view" else EDIT


_HEREDOC_RE = re.compile(r"<<-?\s*[\"']?(\w+)[\"']?")


def strip_heredocs(command: str) -> str:
    """``command`` without the bodies of its here-documents (the lines up to each delimiter)."""
    out: list[str] = []
    ends: list[str] = []
    for line in (command or "").splitlines():
        if ends:
            if line.strip() == ends[0]:
                ends.pop(0)
            continue
        out.append(line)
        ends.extend(_HEREDOC_RE.findall(line))
    return "\n".join(out)


_QUOTED_RE = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'[^']*'")
_CHAIN_RE = re.compile(r"&&|\|\||;|\|")


def is_shell_write(command: str) -> bool:
    """True when the first line of ``command`` only writes files with cat, echo, printf or tee.

    Every command chained on that line (``&&``, ``;``, ``|``) must be one of
    those four and at least one must send output to a file, so
    ``cat << EOF | tee f.py`` is a write while ``echo "" > cfg && python app.py``
    and ``printf x | tool > out`` are not.
    """
    lines = (command or "").strip().splitlines()
    first = lines[0] if lines else ""
    while _CD_PREFIX_RE.match(first):
        first = _CD_PREFIX_RE.sub("", first, count=1)
    parts = _CHAIN_RE.split(_QUOTED_RE.sub("Q", first))
    if not parts or not all(_WRITER_RE.match(part) for part in parts):
        return False
    return any(_OUT_REDIRECT_RE.search(part) or _TEE_TARGET_RE.search(part) for part in parts)


def classify_tool(action: str, command: str) -> str:
    """Map an action name and its command to one tool class.

    A file-editor call (``str_replace_editor`` or
    ``str_replace_based_edit_tool``, as the action name or the command's first
    word) is ``read`` for its ``view`` sub-command and ``edit`` otherwise.
    Otherwise the action name wins when it is a known editor, reader or submit
    action.
    A shell action is looked at by its command: a test runner, or a python
    test or reproduction script (``test_*.py``, ``*_test.py``,
    ``reproduce*.py``, ``repro*.py``), is ``test``, an
    in-place edit or a shell write (``cat > f << EOF``, ``echo x > f``,
    ``tee f``) is ``edit``, a pager or cat is ``read``, grep/find is
    ``search``, anything else is ``shell``. A test run is checked first,
    but the test runner is looked for outside here-document bodies only, so
    ``cat > f.py << EOF`` whose body imports pytest is an edit, while
    ``echo x > cfg && pytest`` is a test run.
    """
    name = (action or "").strip().lower()
    cmd = (command or "").strip()
    editor = _editor_class(name, cmd)
    if editor is not None:
        return editor
    known = _ACTION_CLASS.get(name)
    if known is not None and known != SHELL:
        return known
    text = cmd if cmd else name
    # "cd /repo && grep ..." is a grep: drop leading directory changes.
    while _CD_PREFIX_RE.match(text):
        text = _CD_PREFIX_RE.sub("", text, count=1)
    shell = strip_heredocs(text)
    if runs_test_runner(shell) or _SCRIPT_TEST_RE.search(shell):
        return TEST
    if is_shell_write(text):
        return EDIT
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


def _opt_int(value: object) -> int | None:
    """``int(value)``, or None when it is missing or not a finite number."""
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError, OverflowError):
        return None


def _opt_float(value: object) -> float | None:
    """``float(value)``, or None when it is missing, not a number, NaN or infinite."""
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


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


def _text(value: object, default: str) -> str:
    return default if value is None else str(value)


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
                traj.run_id = _text(rec.get("run_id"), traj.run_id)
                traj.task_id = _text(rec.get("task_id"), "")
                traj.model = _text(rec.get("model"), "")
                if "patch" in rec:
                    traj.patch = str(rec["patch"] or "")
                if rec.get("exit_status") is not None:
                    traj.extra["exit_status"] = str(rec["exit_status"])
                resolved = rec.get("resolved")
                if isinstance(resolved, bool) or (isinstance(resolved, int) and resolved in (0, 1)):
                    traj.resolved = bool(resolved)
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

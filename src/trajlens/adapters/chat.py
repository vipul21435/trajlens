"""Chat-transcript adapter.

Many published trajectory datasets store a run as the message list sent to
the model rather than in an agent framework's own file format. This adapter
reads such a list when each agent turn is a discussion followed by one
fenced code block holding the command, the convention of SWE-agent's
thought-action parser and of mini-swe-agent.

Accepted input: a JSON list of messages, or an object with the list under
``messages``, ``trajectory`` or ``history``. A message is an object with a
role (``role``) and a text (``content`` or ``text``; a list of
``{"type": "text", "text": ...}`` parts is joined).

* Messages with role ``assistant`` or ``ai`` are agent turns. The command is
  the last fenced block, as in SWE-agent's ``ThoughtActionParser``; the text
  before it is the thought.
* The next message with role ``user``, ``tool`` or ``observation`` after an
  agent turn is its observation. Messages before the first agent turn (the
  system prompt and the task statement) are skipped.
* Object inputs may carry ``patch`` (or ``generated_patch`` or
  ``submission``), ``exit_status``, ``resolved`` (or ``target``),
  ``instance_id`` (or ``task_id``) and ``model`` (or ``model_name``).

No exit codes or timestamps are present in this shape; failures are inferred
from observation text by the metrics (see ``--infer-errors``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trajlens.adapters.generic import classify_tool
from trajlens.adapters.minisweagent import SUBMIT_MARKERS
from trajlens.model import OTHER, SUBMIT, Step, Trajectory

_FENCE_RE = re.compile(r"^```[^\n`]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
AGENT_ROLES = ("assistant", "ai")
OBSERVATION_ROLES = ("user", "tool", "observation")


def message_text(msg: dict) -> str:
    """The text of one message, from ``content`` or ``text``."""
    value = msg.get("content", msg.get("text"))
    if isinstance(value, list):
        parts = [p.get("text", "") for p in value if isinstance(p, dict)]
        return "\n".join(str(p) for p in parts if p)
    return "" if value is None else str(value)


def split_last_block(text: str) -> tuple[str, str]:
    """Return (thought, command): the command is the last fenced block."""
    blocks = list(_FENCE_RE.finditer(text or ""))
    if not blocks:
        return (text or "").strip(), ""
    last = blocks[-1]
    return text[: last.start()].strip(), last.group(1).strip()


def _tool(command: str) -> tuple[str, str]:
    if not command:
        return OTHER, ""
    if any(marker in command for marker in SUBMIT_MARKERS):
        return SUBMIT, "submit"
    name = command.split(None, 1)[0]
    return classify_tool(name, command), name


def _first(data: dict, *keys: str):
    for k in keys:
        if data.get(k) is not None:
            return data[k]
    return None


def parse_chat(data, run_id: str) -> Trajectory:
    """Build a Trajectory from a parsed chat transcript."""
    meta: dict = {}
    messages = data
    if isinstance(data, dict):
        meta = data
        messages = _first(data, "messages", "trajectory", "history")
    if not isinstance(messages, list):
        raise ValueError("chat transcript must be a message list or hold one under 'messages'")
    traj = Trajectory(run_id=str(_first(meta, "run_id") or run_id), source_format="chat")
    pending: tuple[str, str] | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        if role in AGENT_ROLES:
            if pending is not None:
                _append(traj, pending, "")
            pending = split_last_block(message_text(msg))
        elif role in OBSERVATION_ROLES and pending is not None:
            _append(traj, pending, message_text(msg))
            pending = None
    if pending is not None:
        _append(traj, pending, "")
    traj.patch = str(_first(meta, "patch", "generated_patch", "submission") or "")
    traj.task_id = str(_first(meta, "task_id", "instance_id") or "")
    traj.model = str(_first(meta, "model", "model_name") or "")
    if meta.get("exit_status") is not None:
        traj.extra["exit_status"] = str(meta["exit_status"])
    resolved = _first(meta, "resolved", "target")
    if isinstance(resolved, bool):
        traj.resolved = resolved
    return traj


def _append(traj: Trajectory, pending: tuple[str, str], observation: str) -> None:
    thought, command = pending
    tool, name = _tool(command)
    traj.steps.append(
        Step(
            index=len(traj.steps),
            tool=tool,
            command=command,
            thought=thought,
            observation=observation,
            raw_action=name,
        )
    )


def load_chat(path: str | Path) -> Trajectory:
    """Read one chat transcript JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return parse_chat(data, path.stem)

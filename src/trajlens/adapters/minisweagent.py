"""mini-swe-agent trajectory adapter.

mini-swe-agent saves a run as JSON in one of two shapes, both read here:

* a list of chat messages (``role``, ``content``), as in the sample files
  under ``tests/test_data`` in the mini-swe-agent repository;
* an object with ``messages`` (the same list), ``info`` (``exit_status``,
  ``submission``, ``model_stats``, ``config``) and ``trajectory_format``
  (``mini-swe-agent-1`` or later).

Each assistant message carries exactly one fenced code block with the
command; the text before the fence is the thought. The user message that
follows is the observation, formatted as
``<returncode>N</returncode>\\n<output>...</output>``. The exit code is read
from the ``returncode`` tag; the observation is the text inside ``output``
(the whole message when the tags are absent).

A command whose text contains ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` or
``MINI_SWE_AGENT_FINAL_OUTPUT`` is the framework's submit signal and is
classified as ``submit``. Every other command is classified by its text as in
the generic adapter (test runner, in-place edit, pager, search, shell).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trajlens.adapters.generic import _opt_int, classify_tool
from trajlens.model import OTHER, SUBMIT, Step, Trajectory

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_RC_RE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>")
_OUT_RE = re.compile(r"<output>\n?(.*?)\n?</output>", re.DOTALL)
SUBMIT_MARKERS = ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "MINI_SWE_AGENT_FINAL_OUTPUT")


def split_assistant(content: str) -> tuple[str, str]:
    """Return (thought, command) from an assistant message."""
    m = _FENCE_RE.search(content or "")
    if not m:
        return (content or "").strip(), ""
    return content[: m.start()].strip(), m.group(1).strip()


def parse_observation(content: str) -> tuple[int | None, str]:
    """Return (exit_code, output) from a user observation message."""
    text = content or ""
    rc = _RC_RE.search(text)
    out = _OUT_RE.search(text)
    code = int(rc.group(1)) if rc else None
    return code, (out.group(1) if out else text)


def _tool_for(command: str) -> tuple[str, str]:
    if any(marker in command for marker in SUBMIT_MARKERS):
        return SUBMIT, "submit"
    if not command:
        return OTHER, ""
    return classify_tool("bash", command), "bash"


def parse_minisweagent(data, run_id: str) -> Trajectory:
    """Build a Trajectory from a parsed mini-swe-agent JSON document."""
    info: dict = {}
    if isinstance(data, dict):
        messages = data.get("messages")
        info = data.get("info") if isinstance(data.get("info"), dict) else {}
    else:
        messages = data
    if not isinstance(messages, list):
        raise ValueError("mini-swe-agent file must be a message list or an object with 'messages'")
    traj = Trajectory(run_id=run_id, source_format="minisweagent")
    pending: tuple[str, str] | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str):
            content = json.dumps(content) if content is not None else ""
        if role == "assistant":
            if pending is not None:
                _append(traj, pending, None, "")
            pending = split_assistant(content)
        elif role == "user" and pending is not None:
            code, out = parse_observation(content)
            _append(traj, pending, code, out)
            pending = None
    if pending is not None:
        _append(traj, pending, None, "")
    traj.patch = str(info.get("submission") or "")
    traj.extra["exit_status"] = str(info.get("exit_status") or "")
    stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}
    traj.extra["api_calls"] = _opt_int(stats.get("api_calls"))
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    traj.model = str(model_cfg.get("model_name") or "")
    return traj


def _append(traj: Trajectory, pending: tuple[str, str], code: int | None, out: str) -> None:
    thought, command = pending
    tool, name = _tool_for(command)
    traj.steps.append(
        Step(
            index=len(traj.steps),
            tool=tool,
            command=command,
            thought=thought,
            observation=out,
            exit_code=code,
            raw_action=name,
        )
    )


def load_minisweagent(path: str | Path) -> Trajectory:
    """Read one mini-swe-agent ``.traj.json`` file."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    stem = path.name[: -len(".traj.json")] if path.name.endswith(".traj.json") else path.stem
    return parse_minisweagent(data, stem)

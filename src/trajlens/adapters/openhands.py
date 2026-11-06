"""OpenHands event JSON adapter.

OpenHands records a session as an event stream. The shape read here is the
one written by ``event_to_dict`` in the OpenHands repository at release
0.9.8 (``openhands/events/serialization/event.py``): a JSON list of event
objects, each with ``id``, ``timestamp`` (ISO 8601), ``source`` (``agent``
or ``user``), ``message`` and either

* ``action`` with ``args`` (``run`` carries ``args.command``; ``run_ipython``
  carries ``args.code``; ``edit``, ``write`` and ``read`` carry ``args.path``;
  ``finish``, ``message`` and ``think`` carry text), or
* ``observation`` with ``content``, ``extras`` and ``cause`` (the id of the
  action it answers). A ``run`` observation carries ``extras.exit_code``.

A file holding an object with an ``events`` or ``history`` list is accepted
too. Only agent actions become steps; user messages and system events are
skipped. The observation for a step is the event whose ``cause`` equals the
action id, otherwise the first observation without a ``cause`` that comes
after the action and before the next action; an action with neither has an
empty observation and no exit code.

Token counts are read when present from ``llm_metrics.accumulated_token_usage``
on the action (cumulative; per step values are the differences) or from
``tool_call_metadata.model_response.usage``. The final patch is not part of
the event stream; it is read from a top level ``patch`` or ``git_patch`` key
when the file is an object, otherwise it stays empty.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from trajlens.adapters.generic import _opt_int, classify_tool
from trajlens.model import EDIT, OTHER, READ, SHELL, SUBMIT, Step, Trajectory

_ACTION_TOOL = {
    "run": SHELL,
    "run_ipython": SHELL,
    "edit": EDIT,
    "write": EDIT,
    "read": READ,
    "finish": SUBMIT,
    "message": OTHER,
    "think": OTHER,
    "browse": OTHER,
    "browse_interactive": OTHER,
    "delegate": OTHER,
    "reject": OTHER,
}


def _stamp(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _usage(event: dict) -> tuple[int | None, int | None, bool]:
    """Token counts of one action event and whether they are cumulative."""
    meta = event.get("tool_call_metadata")
    if isinstance(meta, dict):
        resp = meta.get("model_response")
        usage = resp.get("usage") if isinstance(resp, dict) else None
        if isinstance(usage, dict):
            tin = _opt_int(usage.get("prompt_tokens"))
            return tin, _opt_int(usage.get("completion_tokens")), False
    metrics = event.get("llm_metrics")
    if isinstance(metrics, dict):
        acc = metrics.get("accumulated_token_usage")
        if isinstance(acc, dict):
            tin = _opt_int(acc.get("prompt_tokens"))
            return tin, _opt_int(acc.get("completion_tokens")), True
    return None, None, False


def _command(name: str, args: dict) -> str:
    if name == "run":
        return str(args.get("command") or "")
    if name == "run_ipython":
        return str(args.get("code") or "")
    if name in ("edit", "write", "read"):
        return f"{name} {args.get('path') or ''}".strip()
    if name == "finish":
        return "finish"
    return name


def _thought(name: str, args: dict, event: dict) -> str:
    for key in ("thought", "final_thought", "content"):
        if isinstance(args.get(key), str) and args[key]:
            return args[key]
    return str(event.get("message") or "")


def _is_key(value: object) -> bool:
    """Event ids and causes are ints or strings; anything else links nothing."""
    return isinstance(value, int | str) and not isinstance(value, bool)


def _next_free_observation(events: list[dict], pos: int) -> dict | None:
    """The first observation after ``events[pos]`` that names no other action as its cause.

    The scan stops at the next action, so an action never takes the output of
    a later one.
    """
    own = events[pos].get("id")
    for x in events[pos + 1 :]:
        if isinstance(x.get("action"), str):
            return None
        if "observation" in x and (x.get("cause") is None or x.get("cause") == own):
            return x
    return None


def parse_openhands(data, run_id: str) -> Trajectory:
    """Build a Trajectory from a parsed OpenHands event list."""
    patch = ""
    events = data
    if isinstance(data, dict):
        events = data.get("events", data.get("history"))
        patch = str(data.get("patch") or data.get("git_patch") or "")
    if not isinstance(events, list):
        raise ValueError("OpenHands file must be an event list or an object with 'events'")
    events = [e for e in events if isinstance(e, dict)]
    by_cause: dict = {}
    for e in events:
        cause = e.get("cause")
        if "observation" in e and _is_key(cause) and cause not in by_cause:
            by_cause[cause] = e
    traj = Trajectory(run_id=run_id, source_format="openhands")
    traj.patch = patch
    prev_in = prev_out = None
    for pos, e in enumerate(events):
        name = e.get("action")
        if not isinstance(name, str) or e.get("source") == "user":
            continue
        args = e.get("args") if isinstance(e.get("args"), dict) else {}
        own = e.get("id")
        obs = (by_cause.get(own) if _is_key(own) else None) or _next_free_observation(events, pos)
        obs = obs or {}
        extras = obs.get("extras") if isinstance(obs.get("extras"), dict) else {}
        tin, tout, cumulative = _usage(e)
        if cumulative and tin is not None:
            tin, prev_in = (tin - prev_in if prev_in is not None else tin), tin
        if cumulative and tout is not None:
            tout, prev_out = (tout - prev_out if prev_out is not None else tout), tout
        command = _command(name, args)
        tool = _ACTION_TOOL.get(name)
        if tool is None or tool == SHELL:
            tool = classify_tool("bash" if tool == SHELL else name, command)
        if name in ("edit", "write"):
            tool = EDIT
        if obs.get("observation") == "error" and extras.get("exit_code") is None:
            exit_code = 1
        else:
            exit_code = _opt_int(extras.get("exit_code"))
        traj.steps.append(
            Step(
                index=len(traj.steps),
                tool=tool,
                command=command,
                thought=_thought(name, args, e),
                observation=str(obs.get("content") or ""),
                exit_code=exit_code,
                timestamp=_stamp(e.get("timestamp")),
                tokens_in=tin,
                tokens_out=tout,
                raw_action=name,
            )
        )
        if name == "finish":
            traj.extra["final_thought"] = _thought(name, args, e)
    return traj


def load_openhands(path: str | Path) -> Trajectory:
    """Read one OpenHands event JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return parse_openhands(data, path.stem)

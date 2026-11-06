"""SWE-agent ``.traj`` adapter.

A ``.traj`` file is one JSON object. The shapes read here were taken from the
demonstration trajectories published in the SWE-agent repository:

* ``trajectory``: a list of steps, each with ``action`` (the command string
  the agent issued, for example ``open "src/app.py" 10`` or ``submit``),
  ``observation``, ``thought``, ``response`` and optionally
  ``execution_time`` (seconds) and ``state``.
* ``info``: run level fields. ``submission`` is the final patch,
  ``exit_status`` the framework's own verdict, ``model_stats`` the token and
  cost totals (``tokens_sent``, ``tokens_received``, ``api_calls``,
  ``instance_cost``).
* ``history``: the message list sent to the model. Used only when there is
  no ``trajectory`` key: every ``message_type == "action"`` message becomes a
  step and the next ``observation`` message its observation.

SWE-agent does not record exit codes per step, so ``exit_code`` is ``None``
and the error metrics stay at zero for this format. Timestamps are the
cumulative sum of ``execution_time`` when present, so durations are the
environment's time, not wall clock.

The first word of ``action`` is the tool name (``open``, ``edit``, ``create``,
``find_file``, ``submit``, ``python`` ...); the whole string is the command.
"""

from __future__ import annotations

import json
from pathlib import Path

from trajlens.adapters.generic import _opt_int, classify_tool
from trajlens.model import Step, Trajectory


def _split_action(action: object) -> tuple[str, str]:
    text = "" if action is None else str(action).strip()
    name = text.split(None, 1)[0] if text else ""
    return name, text


def _step(index: int, action: object, observation: str, thought: str, stamp: float | None) -> Step:
    name, command = _split_action(action)
    return Step(
        index=index,
        tool=classify_tool(name, command),
        command=command,
        thought=str(thought or ""),
        observation=str(observation or ""),
        exit_code=None,
        timestamp=stamp,
        raw_action=name,
    )


def _steps_from_trajectory(items: list) -> list[Step]:
    steps: list[Step] = []
    clock = 0.0
    have_clock = False
    for rec in items:
        if not isinstance(rec, dict):
            continue
        t = rec.get("execution_time")
        stamp = None
        if t is not None:
            try:
                clock += float(t)
                have_clock = True
            except (TypeError, ValueError):
                pass
        if have_clock:
            stamp = clock
        action, obs = rec.get("action", ""), rec.get("observation", "")
        steps.append(_step(len(steps), action, obs, rec.get("thought"), stamp))
    return steps


def _steps_from_history(history: list) -> list[Step]:
    steps: list[Step] = []
    pending: dict | None = None
    for msg in history:
        if not isinstance(msg, dict):
            continue
        kind = msg.get("message_type")
        untyped_action = kind is None and msg.get("role") == "assistant" and "action" in msg
        if kind == "action" or untyped_action:
            if pending is not None:
                action = pending.get("action", "")
                steps.append(_step(len(steps), action, "", pending.get("thought"), None))
            pending = msg
        elif kind == "observation" and pending is not None:
            content = msg.get("content", "")
            thought = pending.get("thought")
            steps.append(_step(len(steps), pending.get("action", ""), content, thought, None))
            pending = None
    if pending is not None:
        steps.append(_step(len(steps), pending.get("action", ""), "", pending.get("thought"), None))
    return steps


def parse_sweagent(data: dict, run_id: str) -> Trajectory:
    """Build a Trajectory from a parsed ``.traj`` object."""
    if not isinstance(data, dict):
        raise ValueError("SWE-agent .traj must be a JSON object")
    traj = Trajectory(run_id=run_id, source_format="sweagent")
    if isinstance(data.get("trajectory"), list):
        items = [x for x in data["trajectory"] if isinstance(x, dict)]
        if items and not any("action" in x for x in items):
            raise ValueError(
                "no trajectory step has an 'action' key; is this a chat or tool-call "
                "transcript? pass --format chat or --format toolcalls"
            )
        traj.steps = _steps_from_trajectory(data["trajectory"])
    elif isinstance(data.get("history"), list):
        traj.steps = _steps_from_history(data["history"])
    else:
        raise ValueError("SWE-agent .traj has neither a 'trajectory' nor a 'history' list")
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    traj.patch = str(info.get("submission") or "")
    traj.extra["exit_status"] = str(info.get("exit_status") or "")
    stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}
    traj.extra["tokens_in"] = _opt_int(stats.get("tokens_sent"))
    traj.extra["tokens_out"] = _opt_int(stats.get("tokens_received"))
    traj.extra["api_calls"] = _opt_int(stats.get("api_calls"))
    env = data.get("environment")
    if isinstance(env, str):
        traj.task_id = env
    return traj


def load_sweagent(path: str | Path) -> Trajectory:
    """Read one SWE-agent ``.traj`` file."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return parse_sweagent(data, path.stem)

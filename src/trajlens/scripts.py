"""Runs of scripts the agent wrote itself count as test runs.

Agents often check a fix with a throwaway script whose name says nothing
(``python convert.py``, ``python check_wheel.py``) rather than with the
project's test runner. The command alone cannot tell such a check from any
other program, but the trajectory can: if the agent created the file earlier
in the same run, running it is the agent testing its own work.

A file counts as created by the agent when an edit step's command is
``create <path>`` or ``write <path>`` (SWE-agent, the OpenHands editor and
event stream), or any step writes it with a redirect or tee
(``cat > check.py``, ``cat << EOF > check.py``, ``echo ... > check.py``,
``tee check.py``), whatever class the step was given.
A later shell step running ``python <path>`` (any flags, any directory
prefix) on a file with the same base name is then reclassified as ``test``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import PurePosixPath

from trajlens.adapters.generic import strip_heredocs
from trajlens.model import EDIT, SHELL, TEST, Step, Trajectory

_CREATE_RE = re.compile(r"^\s*(?:create|write)\s+[\"']?([^\s\"']+)")
_REDIRECT_RE = re.compile(r">\s*[\"']?([^\s\"'|;&]+\.py)\b")
_TEE_RE = re.compile(r"(?:^|[|;&\s])tee\s+(?:-\S+\s+)*[\"']?([^\s\"'|;&]+\.py)\b")
_RUN_RE = re.compile(r"(?:^|[\s;&|])python[0-9.]*\s+(?:-\S+\s+)*[\"']?([^\s\"';&|]+\.py)\b")


def _base(path: str) -> str:
    return PurePosixPath(path).name


def created_files(step: Step) -> list[str]:
    """Base names of files this step creates."""
    out: list[str] = []
    if step.tool == EDIT:
        m = _CREATE_RE.match(step.command)
        if m:
            out.append(_base(m.group(1)))
    shell = strip_heredocs(step.command)
    for p in _REDIRECT_RE.findall(shell) + _TEE_RE.findall(shell):
        if _base(p) not in out:
            out.append(_base(p))
    return out


def run_script(step: Step) -> str | None:
    """Base name of the python script a shell step runs, if any."""
    if step.tool != SHELL:
        return None
    m = _RUN_RE.search(step.command)
    return _base(m.group(1)) if m else None


def promote_script_runs(traj: Trajectory) -> Trajectory:
    """A copy of ``traj`` whose runs of agent-created scripts are ``test`` steps.

    The input is not changed; steps that are not promoted are shared.
    """
    created: set[str] = set()
    steps: list[Step] = []
    changed = False
    for step in traj.steps:
        script = run_script(step)
        if script is not None and script in created:
            steps.append(replace(step, tool=TEST))
            changed = True
        else:
            steps.append(step)
        created.update(created_files(step))
    return replace(traj, steps=steps) if changed else traj

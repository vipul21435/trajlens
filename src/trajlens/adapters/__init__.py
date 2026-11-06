"""Adapters from trajectory file formats to the step model.

``load(path, fmt)`` reads one file. With ``fmt="auto"`` the format is
guessed from the file name and, for ``.json``, from the document shape:

* ``*.jsonl``                       generic JSONL
* ``*.traj``                        SWE-agent
* ``*.traj.json``                   mini-swe-agent
* other ``*.json``: a list of ``role``/``content`` messages or an object
  with ``messages`` is mini-swe-agent; a list of events with ``action`` or
  ``observation`` keys, or an object with ``events``/``history`` holding such
  events, is OpenHands; an object with ``trajectory`` is SWE-agent.
"""

from __future__ import annotations

import json
from pathlib import Path

from trajlens.adapters.generic import load_generic
from trajlens.adapters.minisweagent import load_minisweagent, parse_minisweagent
from trajlens.adapters.openhands import load_openhands, parse_openhands
from trajlens.adapters.sweagent import load_sweagent, parse_sweagent
from trajlens.model import Trajectory

FORMATS = ("generic", "sweagent", "minisweagent", "openhands")

_LOADERS = {
    "generic": load_generic,
    "sweagent": load_sweagent,
    "minisweagent": load_minisweagent,
    "openhands": load_openhands,
}


def _looks_like_events(items: list) -> bool:
    dicts = [x for x in items if isinstance(x, dict)]
    return bool(dicts) and all("action" in x or "observation" in x for x in dicts)


def _looks_like_messages(items: list) -> bool:
    dicts = [x for x in items if isinstance(x, dict)]
    return bool(dicts) and all("role" in x for x in dicts)


def detect_json(data) -> str:
    """Name the format of a parsed ``.json`` document, or raise ValueError."""
    if isinstance(data, list):
        if _looks_like_events(data):
            return "openhands"
        if _looks_like_messages(data):
            return "minisweagent"
    elif isinstance(data, dict):
        plain_history = "history" in data and "messages" not in data and "events" not in data
        if "trajectory" in data or plain_history:
            return "sweagent"
        fmt = str(data.get("trajectory_format", ""))
        if "messages" in data or fmt.startswith("mini-swe-agent"):
            return "minisweagent"
        if isinstance(data.get("events"), list):
            return "openhands"
    raise ValueError("cannot tell the trajectory format from the document; pass --format")


def detect_format(path: str | Path) -> str:
    """Guess the format of one file from its name, then from its contents."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".jsonl"):
        return "generic"
    if name.endswith(".traj"):
        return "sweagent"
    if name.endswith(".traj.json"):
        return "minisweagent"
    with path.open(encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON: {exc}") from exc
    return detect_json(data)


def load(path: str | Path, fmt: str = "auto") -> Trajectory:
    """Load one trajectory file in the named format, or detect the format."""
    if fmt == "auto":
        fmt = detect_format(path)
    loader = _LOADERS.get(fmt)
    if loader is None:
        raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")
    return loader(path)


__all__ = [
    "FORMATS",
    "detect_format",
    "detect_json",
    "load",
    "load_generic",
    "load_minisweagent",
    "load_openhands",
    "load_sweagent",
    "parse_minisweagent",
    "parse_openhands",
    "parse_sweagent",
]

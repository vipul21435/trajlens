"""Adapters from trajectory file formats to the step model.

``load(path, fmt)`` reads one file. With ``fmt="auto"`` the format is
guessed from the file name and, for ``.json``, from the document shape:

* ``*.jsonl``                       generic JSONL
* ``*.traj``                        SWE-agent
* ``*.traj.json``                   mini-swe-agent
* other ``*.json``: a list of ``role``/``content`` messages or an object
  with ``messages`` is mini-swe-agent, unless a message carries
  ``tool_calls`` (then ``toolcalls``); a list of events with ``action`` or
  ``observation`` keys, or an object with ``events``/``history`` holding such
  events, is OpenHands; an object whose ``trajectory`` is a list of
  ``role`` messages with no ``action`` is ``toolcalls`` when a message has
  ``tool_calls`` and ``chat`` otherwise; any other object with
  ``trajectory`` is SWE-agent.

A ``chat`` transcript held as a bare list looks like a mini-swe-agent file
and is read as one; pass ``--format chat`` for it.
"""

from __future__ import annotations

import json
from pathlib import Path

from trajlens.adapters.chat import load_chat, parse_chat
from trajlens.adapters.generic import load_generic
from trajlens.adapters.minisweagent import load_minisweagent, parse_minisweagent
from trajlens.adapters.openhands import load_openhands, parse_openhands
from trajlens.adapters.sweagent import load_sweagent, parse_sweagent
from trajlens.adapters.toolcalls import load_toolcalls, parse_toolcalls
from trajlens.model import Trajectory

FORMATS = ("generic", "sweagent", "minisweagent", "openhands", "chat", "toolcalls")

_LOADERS = {
    "generic": load_generic,
    "sweagent": load_sweagent,
    "minisweagent": load_minisweagent,
    "openhands": load_openhands,
    "chat": load_chat,
    "toolcalls": load_toolcalls,
}


def _looks_like_events(items: list) -> bool:
    dicts = [x for x in items if isinstance(x, dict)]
    return bool(dicts) and all("action" in x or "observation" in x for x in dicts)


def _looks_like_messages(items: list) -> bool:
    dicts = [x for x in items if isinstance(x, dict)]
    return bool(dicts) and all("role" in x for x in dicts)


def _has_tool_calls(items: list) -> bool:
    return any(isinstance(x, dict) and x.get("tool_calls") for x in items)


def _transcript_format(items: object) -> str | None:
    """``toolcalls`` or ``chat`` when ``items`` is a message list rather than SWE-agent steps."""
    if not isinstance(items, list):
        return None
    dicts = [x for x in items if isinstance(x, dict)]
    if not dicts or any("action" in x for x in dicts) or not all("role" in x for x in dicts):
        return None
    return "toolcalls" if _has_tool_calls(dicts) else "chat"


def detect_json(data: object) -> str:
    """Name the format of a parsed ``.json`` document, or raise ValueError."""
    if isinstance(data, list):
        if _looks_like_events(data):
            return "openhands"
        if _looks_like_messages(data):
            return "toolcalls" if _has_tool_calls(data) else "minisweagent"
    elif isinstance(data, dict):
        transcript = _transcript_format(data.get("trajectory"))
        if transcript is not None:
            return transcript
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
    try:
        return detect_json(data)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load(path: str | Path, fmt: str = "auto") -> Trajectory:
    """Load one trajectory file in the named format, or detect the format.

    Any problem with the file's contents is raised as ValueError whose message
    starts with the path; a missing or unreadable file raises OSError.
    """
    if fmt == "auto":
        try:
            fmt = detect_format(path)
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path}: not UTF-8 text: {exc}") from exc
    loader = _LOADERS.get(fmt)
    if loader is None:
        raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")
    try:
        return loader(path)
    except OSError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
        text = str(exc)
        if text.startswith(str(path)):
            raise
        kind = "" if type(exc) in (ValueError, json.JSONDecodeError) else f"{type(exc).__name__}: "
        raise ValueError(f"{path}: {kind}{text}") from exc


__all__ = [
    "FORMATS",
    "detect_format",
    "load_chat",
    "parse_chat",
    "detect_json",
    "load",
    "load_generic",
    "load_minisweagent",
    "load_openhands",
    "load_sweagent",
    "parse_minisweagent",
    "parse_openhands",
    "parse_sweagent",
    "load_toolcalls",
    "parse_toolcalls",
]

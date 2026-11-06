"""Function-calling transcript adapter.

Agents built on a model's native tool calling (OpenHands from its
function-calling agent on, and many others) are often published as the
message list in the widely used chat-completions shape: an ``assistant``
message holds ``content`` (the thought) and ``tool_calls``, each with an
``id`` and ``function: {name, arguments}`` (``arguments`` a JSON string or an
object); a ``tool`` message answers one call through ``tool_call_id``.

Accepted input: a JSON list of such messages, or an object holding the list
under ``messages``, ``trajectory`` or ``history``, with optional run fields
``model_patch`` (or ``patch``, ``generated_patch``, ``submission``),
``exit_status``, ``resolved`` (true/false or 1/0), ``instance_id`` (or
``task_id``) and ``model`` (or ``model_name``).

Every tool call is one step; an assistant message with no tool call is one
step of class ``other`` whose observation is the next ``user`` message.
Tool names map to classes as follows:

* shell tools (``execute_bash``, ``bash``, ``run``, ``shell``, ``terminal``,
  ``execute_ipython_cell``): the command is ``arguments.command`` (or
  ``code``) and is classified by its text, as in the generic adapter;
* editor tools (``str_replace_editor``, ``str_replace_based_edit_tool``,
  ``edit_file``): ``view`` is ``read``, every other sub-command is ``edit``;
  the command is ``"<sub-command> <path>"``;
* ``finish``, ``submit``: ``submit``; ``think`` and anything unknown:
  classified by name, usually ``other``.

Exit codes: a shell observation ending in OpenHands' line
``[Command finished with exit code N]`` gives ``N``; one ending in its
timeout notice gives -1 and one ending in
``[Error occurred in processing last action]`` gives 1; any other shell
observation has none. Any other tool's observation is a failure (exit code 1)
when it starts with ``ERROR`` and a success (0) otherwise.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trajlens.adapters.chat import message_text
from trajlens.adapters.generic import classify_tool
from trajlens.model import EDIT, OTHER, READ, SUBMIT, Step, Trajectory

SHELL_TOOLS = ("execute_bash", "bash", "run", "shell", "terminal", "execute_ipython_cell")
EDITOR_TOOLS = ("str_replace_editor", "str_replace_based_edit_tool", "edit_file")
SUBMIT_TOOLS = ("finish", "submit")
_EXIT_RE = re.compile(r"\[Command finished with exit code (-?\d+)\]\s*$")
_TIMEOUT_RE = re.compile(r"^\[The command (?:has no new output after|timed out after) ")


def _args(call: dict) -> dict:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    raw = fn.get("arguments", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
    return raw if isinstance(raw, dict) else {"raw": raw}


def _name(call: dict) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(fn.get("name") or call.get("name") or "")


def describe_call(name: str, args: dict) -> tuple[str, str]:
    """Return (tool class, command string) for one tool call."""
    if name in SHELL_TOOLS:
        command = str(args.get("command") or args.get("code") or "")
        return classify_tool("bash", command), command
    if name in EDITOR_TOOLS:
        sub = str(args.get("command") or "edit")
        command = f"{sub} {args.get('path') or ''}".strip()
        return (READ if sub == "view" else EDIT), command
    if name in SUBMIT_TOOLS:
        return SUBMIT, name
    tool = classify_tool(name, "")
    return tool, name


def exit_code_for(name: str, observation: str) -> int | None:
    """Exit code of one tool call from its observation text.

    A shell reply ending in OpenHands' timeout notice (``[The command has no
    new output after N seconds ...]`` or ``[The command timed out after ...]``)
    gets -1, and one ending in ``[Error occurred in processing last action]``
    gets 1, so both count as failed steps.
    """
    if name in SHELL_TOOLS:
        text = observation.rstrip()
        m = _EXIT_RE.search(text)
        if m:
            return int(m.group(1))
        lines = text.splitlines()
        last = lines[-1].strip() if lines else ""
        if _TIMEOUT_RE.match(last):
            return -1
        if last == "[Error occurred in processing last action]":
            return 1
        return None
    return 1 if observation.lstrip().startswith("ERROR") else 0


def _first(data: dict, *keys: str):
    for k in keys:
        if data.get(k) is not None:
            return data[k]
    return None


def parse_toolcalls(data: object, run_id: str) -> Trajectory:
    """Build a Trajectory from a parsed function-calling transcript."""
    meta: dict = {}
    messages = data
    if isinstance(data, dict):
        meta = data
        messages = _first(data, "messages", "trajectory", "history")
    if not isinstance(messages, list):
        raise ValueError("tool-call transcript must be a message list or hold one under 'messages'")
    messages = [m for m in messages if isinstance(m, dict)]
    replies: dict[str, str] = {}
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") is not None:
            replies.setdefault(str(m["tool_call_id"]), message_text(m))
    traj = Trajectory(run_id=str(_first(meta, "run_id") or run_id), source_format="toolcalls")
    for pos, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        thought = message_text(m)
        calls = m.get("tool_calls") if isinstance(m.get("tool_calls"), list) else []
        calls = [c for c in calls if isinstance(c, dict)]
        if not calls:
            nxt = next((x for x in messages[pos + 1 :] if x.get("role") in ("user", "tool")), {})
            obs = message_text(nxt) if nxt.get("role") == "user" else ""
            traj.steps.append(
                Step(index=len(traj.steps), tool=OTHER, thought=thought, observation=obs,
                     raw_action="message")
            )  # fmt: skip
            continue
        for k, call in enumerate(calls):
            name = _name(call)
            tool, command = describe_call(name, _args(call))
            obs = replies.get(str(call.get("id")), "")
            traj.steps.append(
                Step(
                    index=len(traj.steps),
                    tool=tool,
                    command=command,
                    thought=thought if k == 0 else "",
                    observation=obs,
                    exit_code=exit_code_for(name, obs),
                    raw_action=name,
                )
            )
    traj.patch = str(_first(meta, "model_patch", "patch", "generated_patch", "submission") or "")
    traj.task_id = str(_first(meta, "task_id", "instance_id") or "")
    traj.model = str(_first(meta, "model", "model_name") or "")
    if meta.get("exit_status") is not None:
        traj.extra["exit_status"] = str(meta["exit_status"])
    resolved = _first(meta, "resolved", "target")
    if isinstance(resolved, bool) or resolved in (0, 1):
        traj.resolved = bool(resolved)
    return traj


def load_toolcalls(path: str | Path) -> Trajectory:
    """Read one function-calling transcript JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return parse_toolcalls(data, path.stem)

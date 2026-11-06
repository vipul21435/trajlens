import json

import pytest

from trajlens.adapters import load
from trajlens.adapters.toolcalls import describe_call, exit_code_for, parse_toolcalls


def _call(cid, name, args, as_string=True):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args) if as_string else args},
    }


ROW = {
    "instance_id": "demo__lib-2",
    "resolved": 1,
    "exit_status": "submit",
    "model_patch": "diff --git a/lib/x.py b/lib/x.py\n",
    "trajectory": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": "Look first.",
            "tool_calls": [
                _call("c1", "str_replace_editor", {"command": "view", "path": "/w/lib/x.py"}),
                _call("c2", "execute_bash", {"command": "python -m pytest -q"}, as_string=False),
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "str_replace_editor", "content": "1 x=1"},
        {
            "role": "tool",
            "tool_call_id": "c2",
            "content": "1 failed\n[Command finished with exit code 1]",
        },
        {"role": "assistant", "content": "I should edit."},
        {"role": "user", "content": "Please continue with a tool call."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _call("c3", "str_replace_editor", {"command": "str_replace", "path": "/w/lib/x.py"})
            ],
        },
        {"role": "tool", "tool_call_id": "c3", "content": "ERROR:\nNo replacement was performed"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c4", "think", {"t": 1})]},
        {"role": "tool", "tool_call_id": "c4", "content": "Your thought has been logged."},
        {"role": "assistant", "content": "", "tool_calls": [_call("c5", "finish", {})]},
    ],
}


def test_parse_row():
    t = parse_toolcalls(ROW, "r")
    assert [s.tool for s in t.steps] == ["read", "test", "other", "edit", "other", "submit"]
    assert [s.exit_code for s in t.steps] == [0, 1, None, 1, 0, 0]
    assert t.steps[0].thought == "Look first." and t.steps[1].thought == ""
    assert t.steps[1].observation.startswith("1 failed")
    assert t.steps[2].observation == "Please continue with a tool call."
    assert t.steps[3].command == "str_replace /w/lib/x.py"
    assert t.resolved is True and t.task_id == "demo__lib-2"
    assert t.extra["exit_status"] == "submit" and t.patch.startswith("diff")


def test_helpers():
    assert describe_call("execute_bash", {"command": "pytest"}) == ("test", "pytest")
    assert describe_call("execute_ipython_cell", {"code": "print(1)"})[0] == "shell"
    assert describe_call("str_replace_editor", {"command": "create", "path": "a.py"}) == (
        "edit",
        "create a.py",
    )
    assert describe_call("task_tracker", {}) == ("other", "task_tracker")
    assert exit_code_for("execute_bash", "out\n[Command finished with exit code -1]\n") == -1
    assert exit_code_for("execute_bash", "no marker") is None
    assert exit_code_for("think", "ok") == 0


def test_load_and_errors(tmp_path):
    p = tmp_path / "r.json"
    bad_args = {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "run",
                "arguments": "{not json"}}]}  # fmt: skip
    p.write_text(json.dumps([bad_args]))
    t = load(p, "toolcalls")
    assert t.run_id == "r" and t.steps[0].command == "" and t.resolved is None
    with pytest.raises(ValueError):
        parse_toolcalls({"x": 1}, "r")

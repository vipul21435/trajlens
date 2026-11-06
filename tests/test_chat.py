import json

import pytest

from trajlens.adapters import load
from trajlens.adapters.chat import message_text, parse_chat, split_last_block
from trajlens.analysis import analyze_trajectories

FENCE = "```"


def _ai(thought, cmd):
    return {"role": "ai", "text": f"{thought}\n\n{FENCE}\n{cmd}\n{FENCE}"}


def _obs(text):
    return {"role": "user", "text": text}


ROW = {
    "instance_id": "demo__lib-1",
    "model_name": "m",
    "target": False,
    "exit_status": "submitted",
    "generated_patch": "diff --git a/lib/x.py b/lib/x.py\n",
    "trajectory": [
        {"role": "system", "text": ""},
        _obs("ISSUE: x is wrong"),
        _ai("Make a script.", "create reproduce.py"),
        _obs("[File: reproduce.py (1 lines total)]"),
        _ai("Run it.", "python reproduce.py"),
        _obs('Traceback (most recent call last):\n  File "reproduce.py"'),
        _ai("Fix.", "edit 1:1\nx = 2\nend_of_edit"),
        _obs("File updated."),
        _ai("Done.", "submit"),
    ],
}


def test_split_last_block_takes_last():
    text = f"See:\n{FENCE}python\nprint(1)\n{FENCE}\nNow run:\n{FENCE}\nls -la\n{FENCE}\n"
    thought, cmd = split_last_block(text)
    assert cmd == "ls -la" and thought.startswith("See:")
    assert split_last_block("no block here") == ("no block here", "")


def test_message_text_forms():
    assert message_text({"content": "a"}) == "a"
    assert message_text({"text": "b"}) == "b"
    assert message_text({"content": [{"type": "text", "text": "c"}, {"type": "image"}]}) == "c"
    assert message_text({}) == ""


def test_parse_chat_row():
    t = parse_chat(ROW, "row")
    assert [s.tool for s in t.steps] == ["edit", "test", "edit", "submit"]
    assert t.steps[1].observation.startswith("Traceback")
    assert t.steps[3].observation == ""
    assert t.task_id == "demo__lib-1" and t.model == "m"
    assert t.resolved is False and t.extra["exit_status"] == "submitted"
    assert t.patch.startswith("diff --git")
    rep = analyze_trajectories([t])
    m = rep["runs"][0]["metrics"]
    assert m["error_steps"] == 1 and m["errors_inferred"]
    assert {f["label"] for f in rep["runs"][0]["findings"]} == {"untested_final_edit"}


def test_parse_chat_list_and_errors(tmp_path):
    msgs = [{"role": "assistant", "content": "thinking only"}, _ai("x", "ls")]
    t = parse_chat(msgs, "r")
    assert [s.command for s in t.steps] == ["", "ls"] and t.steps[0].tool == "other"
    assert t.resolved is None
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"messages": msgs, "run_id": "named"}))
    assert load(p, "chat").run_id == "named"
    with pytest.raises(ValueError):
        parse_chat({"nothing": 1}, "r")

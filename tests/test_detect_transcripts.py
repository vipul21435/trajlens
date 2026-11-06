"""Chat and tool-call transcripts under a ``trajectory`` key are not SWE-agent files."""

import json

import pytest

from trajlens.adapters import detect_json, load
from trajlens.adapters.sweagent import parse_sweagent

CHAT_ROW = {
    "instance_id": "demo__demo-1",
    "model_name": "m",
    "trajectory": [
        {"role": "system", "text": "You are an agent."},
        {"role": "ai", "text": "Look first.\n```\nls\n```"},
        {"role": "user", "text": "a.py"},
    ],
    "exit_status": "submitted",
    "generated_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
}

TOOL_ROW = {
    "instance_id": "demo__demo-2",
    "trajectory": [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "run it",
            "tool_calls": [
                {"id": "c1", "function": {"name": "execute_bash", "arguments": {"command": "ls"}}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "a.py"},
    ],
    "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
}


def test_detect_chat_and_toolcalls_rows():
    assert detect_json(CHAT_ROW) == "chat"
    assert detect_json(TOOL_ROW) == "toolcalls"
    assert detect_json({"trajectory": [{"action": "ls", "observation": ""}]}) == "sweagent"
    assert detect_json({"trajectory": []}) == "sweagent"
    assert detect_json(TOOL_ROW["trajectory"]) == "toolcalls"


def test_auto_load_reads_the_patch(tmp_path):
    path = tmp_path / "row.json"
    path.write_text(json.dumps(TOOL_ROW), encoding="utf-8")
    traj = load(path)
    assert traj.source_format == "toolcalls"
    assert traj.patch.startswith("diff --git")


def test_sweagent_rejects_transcripts():
    with pytest.raises(ValueError, match="pass --format"):
        parse_sweagent(CHAT_ROW, "x")

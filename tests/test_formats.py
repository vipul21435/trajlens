import json
from pathlib import Path

import pytest

from trajlens.adapters import detect_format, detect_json, load
from trajlens.adapters.minisweagent import parse_minisweagent, parse_observation, split_assistant
from trajlens.adapters.openhands import parse_openhands
from trajlens.adapters.sweagent import parse_sweagent
from trajlens.classify import classify_run
from trajlens.metrics import patch_paths, run_metrics

FX = Path(__file__).parent / "fixtures"


def test_sweagent_trajectory_file():
    t = load(FX / "sample.traj")
    assert t.source_format == "sweagent"
    assert t.task_id == "demo-env"
    assert [s.tool for s in t.steps] == ["search", "read", "edit", "test", "submit"]
    assert t.steps[0].raw_action == "ls"
    assert t.steps[2].command.startswith("edit 2:2")
    assert t.steps[0].exit_code is None
    assert [s.timestamp for s in t.steps] == [0.5, 0.75, 1.75, 3.75, 4.0]
    assert t.patch.startswith("diff --git")
    assert t.extra["tokens_in"] == 1200 and t.extra["api_calls"] == 5
    m = run_metrics(t)
    assert m.tokens_in == 1200 and m.tokens_out == 80
    assert m.duration_seconds == 3.5
    assert m.patch_files == ["src/calc.py"] and m.tests_after_last_edit
    assert classify_run(t, m) == []


def test_sweagent_history_only():
    t = load(FX / "history_only.traj")
    assert [s.tool for s in t.steps] == ["search", "read", "submit"]
    assert t.steps[0].observation.startswith("Found 1")
    assert t.steps[1].observation == ""
    assert t.steps[2].observation == ""
    assert t.patch == ""


def test_sweagent_bad_execution_time_and_errors():
    t = parse_sweagent({"trajectory": [{"action": "ls", "execution_time": "x"}, 3]}, "r")
    assert len(t.steps) == 1 and t.steps[0].timestamp is None
    with pytest.raises(ValueError):
        parse_sweagent({"nothing": 1}, "r")
    with pytest.raises(ValueError):
        parse_sweagent([], "r")


def test_minisweagent_file():
    t = load(FX / "sample.traj.json")
    assert t.source_format == "minisweagent" and t.run_id == "sample"
    assert t.model == "demo-model"
    assert [s.tool for s in t.steps] == ["read", "edit", "test", "other", "submit"]
    assert t.steps[0].exit_code == 0 and t.steps[2].exit_code == 1
    assert t.steps[2].observation == "1 failed"
    assert t.steps[3].command == "" and t.steps[3].observation.startswith("plain")
    assert t.steps[4].raw_action == "submit"
    assert t.extra["exit_status"] == "Submitted"
    m = run_metrics(t)
    assert m.patch_files == ["app.py"] and m.submitted


def test_minisweagent_list_and_helpers():
    msgs = [
        {"role": "assistant", "content": "a\n```bash\nls\n```"},
        {"role": "user", "content": [{"type": "text", "text": "x"}]},
        {"role": "assistant", "content": "```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT\n```"},
    ]
    t = parse_minisweagent(msgs, "r")
    assert [s.tool for s in t.steps] == ["search", "submit"]
    assert t.steps[1].observation == ""
    assert split_assistant("no fence") == ("no fence", "")
    assert parse_observation("<returncode>-1</returncode>\n<output>\nhi\n</output>") == (-1, "hi")
    assert parse_observation("bare") == (None, "bare")
    with pytest.raises(ValueError):
        parse_minisweagent({"info": {}}, "r")


def test_openhands_file():
    t = load(FX / "sample_openhands.json")
    assert t.source_format == "openhands"
    assert [s.tool for s in t.steps] == ["read", "edit", "test", "shell", "other", "submit"]
    assert [s.raw_action for s in t.steps][:2] == ["run", "edit"]
    assert t.steps[0].observation.startswith("def add")
    assert t.steps[0].exit_code == 0
    assert t.steps[2].exit_code == 1  # error observation without exit code
    assert t.steps[1].command == "edit src/calc.py"
    assert t.steps[0].tokens_in == 100 and t.steps[1].tokens_in == 150
    assert t.steps[2].tokens_in == 300 and t.steps[2].tokens_out == 12
    assert t.steps[0].timestamp is not None and t.steps[5].timestamp - t.steps[0].timestamp == 8
    assert t.extra["final_thought"] == "Fixed add()"
    m = run_metrics(t)
    assert m.tokens_in == 550 and m.duration_seconds == 8


def test_openhands_object_with_patch_and_errors():
    events = [{"id": 1, "action": "run", "args": {"command": "ls"}, "timestamp": 5}]
    t = parse_openhands({"events": events, "git_patch": "diff --git a/x b/x\n"}, "r")
    assert t.patch.startswith("diff") and t.steps[0].timestamp == 5.0
    ev = {"id": 1, "action": "run", "args": {"command": "ls"}, "timestamp": "bad"}
    t = parse_openhands([ev], "r")
    assert t.steps[0].timestamp is None and t.steps[0].observation == ""
    with pytest.raises(ValueError):
        parse_openhands({"no": "events"}, "r")


def test_detect_by_shape(tmp_path):
    assert detect_format(FX / "good.jsonl") == "generic"
    assert detect_format(FX / "sample.traj") == "sweagent"
    assert detect_format(FX / "sample.traj.json") == "minisweagent"
    assert detect_format(FX / "sample_openhands.json") == "openhands"
    assert detect_json({"trajectory_format": "mini-swe-agent-1.1"}) == "minisweagent"
    assert detect_json({"events": []}) == "openhands"
    assert detect_json({"history": []}) == "sweagent"
    assert detect_json([{"role": "user", "content": "x"}]) == "minisweagent"
    with pytest.raises(ValueError):
        detect_json({"other": 1})
    with pytest.raises(ValueError):
        detect_json([1, 2])
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError):
        detect_format(bad)
    for fmt in ("sweagent", "minisweagent", "openhands"):
        with pytest.raises(ValueError):
            load(FX / "good.jsonl", fmt=fmt) if fmt != "sweagent" else load(bad, fmt=fmt)


def test_patch_paths_crlf():
    patch = "\r\ndiff --git a/src/f.py b/src/f.py\r\n--- a/src/f.py\r\n+++ b/src/f.py\r\n"
    assert patch_paths(patch) == ["src/f.py"]
    assert patch_paths(json.dumps({"x": 1})) == []

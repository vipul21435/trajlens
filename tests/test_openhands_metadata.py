import json

from trajlens.adapters.openhands import parse_openhands
from trajlens.analysis import analyze_trajectories
from trajlens.cli import main

CMD = "python -m pytest tests/test_a.py"


def _events():
    events = []
    for i in range(4):
        events.append({"id": 2 * i, "source": "agent", "action": "run", "args": {"command": CMD}})
        events.append(
            {
                "id": 2 * i + 1,
                "source": "agent",
                "observation": "run",
                "cause": 2 * i,
                "content": f"FAILED tests/test_a.py::test_{i} - assert 0\n1 failed",
                "extras": {"command": CMD, "metadata": {"exit_code": 1, "pid": -1}},
            }
        )
    return events


def test_exit_code_from_metadata():
    traj = parse_openhands(_events(), "oh")
    assert [s.exit_code for s in traj.steps] == [1, 1, 1, 1]
    run = analyze_trajectories([traj])["runs"][0]
    assert run["metrics"]["error_steps"] == 4
    assert "tool_error_cascade" in {f["label"] for f in run["findings"]}


def test_top_level_exit_code_still_wins():
    events = _events()[:2]
    events[1]["extras"] = {"exit_code": 0, "metadata": {"exit_code": 1}}
    assert parse_openhands(events, "oh").steps[0].exit_code == 0


def test_cli_reports_errors(tmp_path, capsys):
    p = tmp_path / "oh_events.json"
    p.write_text(json.dumps(_events()))
    assert main(["analyze", str(p), "--format", "openhands"]) == 0
    out = capsys.readouterr().out
    assert "error rate 1.00" in out
    assert "tool_error_cascade" in out

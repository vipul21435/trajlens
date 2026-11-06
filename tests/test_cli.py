import json
from pathlib import Path

from trajlens.analysis import Options, analyze_trajectories
from trajlens.cli import aggregate, analyze_paths, main
from trajlens.model import Step, Trajectory

FX = Path(__file__).parent / "fixtures"


def test_main_writes_json_and_text(tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = main(["analyze", str(FX / "good.jsonl"), str(FX / "loop.jsonl"), "--json", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "good: 7 steps" in text
    assert "looping: exact repeat" in text
    assert "runs: 2" in text
    rep = json.loads(out.read_text())
    assert rep["version"] == "0.1.0"
    assert len(rep["runs"]) == 2
    assert rep["aggregate"]["runs_with_finding"]["looping"] == 1
    assert rep["aggregate"]["tool_mix"]["edit"] == 2


def test_main_bad_file(capsys):
    rc = main(["analyze", "does-not-exist.jsonl"])
    assert rc == 2
    assert "trajlens:" in capsys.readouterr().err


def test_flags_change_findings():
    rep = analyze_paths([FX / "near_loop.jsonl"], opts=Options(loop_min_steps=5))
    assert rep["runs"][0]["findings"] == []
    rep = analyze_paths([FX / "near_loop.jsonl"], opts=Options(near_threshold=0.99))
    assert rep["runs"][0]["findings"] == []


def test_aggregate_all_fixtures():
    rep = analyze_paths(sorted(FX.glob("*.jsonl")))
    a = rep["aggregate"]
    assert a["runs"] == 5
    assert a["runs_empty_patch"] == 2
    assert a["runs_patch_touches_tests"] == 1
    assert a["runs_with_loops"] == 2
    assert a["mean_first_test_step"] is not None


def test_aggregate_empty():
    a = aggregate([])
    assert a["runs"] == 0 and a["mean_steps"] is None
    assert main(["analyze", str(FX / "empty.jsonl")]) == 0


def test_by_outcome_split(capsys):
    edit = Step(index=0, tool="edit", command="edit a.py")
    ok = Trajectory("ok", [edit, Step(index=1, tool="test", command="pytest")], resolved=True)
    bad = Trajectory("bad", [edit], resolved=False)
    unknown = Trajectory("u", [edit])
    rep = analyze_trajectories([ok, bad, unknown])
    by = rep["aggregate"]["by_outcome"]
    assert by["resolved"]["runs"] == 1 and by["unresolved"]["runs"] == 1
    assert by["unresolved"]["runs_with_finding"]["never_tested"] == 1
    assert by["unresolved"]["share_with_finding"]["never_tested"] == 1.0
    assert by["resolved"]["runs_no_finding"] == 0  # gave_up: no submit, empty patch
    assert rep["runs"][0]["resolved"] is True and rep["runs"][2]["resolved"] is None
    assert "by_outcome" not in analyze_trajectories([unknown])["aggregate"]


def test_generic_meta_outcome(tmp_path, capsys):
    p = tmp_path / "r.jsonl"
    p.write_text(
        json.dumps({"type": "meta", "run_id": "r", "resolved": False, "exit_status": "exit_cost"})
        + "\n"
        + json.dumps({"action": "bash", "command": "ls"})
        + "\n"
    )
    rep = analyze_paths([p])
    assert rep["runs"][0]["resolved"] is False
    assert rep["runs"][0]["exit_status"] == "exit_cost"
    assert main(["analyze", str(p)]) == 0
    assert "unresolved runs: 1" in capsys.readouterr().out

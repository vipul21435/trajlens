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


def test_group_by_model(capsys):
    edit = Step(index=0, tool="edit", command="edit a.py")
    a = Trajectory("a", [edit], model="m1", resolved=True)
    b = Trajectory("b", [edit], model="m1", resolved=False)
    c = Trajectory("c", [edit], model="m2")
    rep = analyze_trajectories([a, b, c], group_by="model")
    g = rep["aggregate"]["groups"]
    assert list(g) == ["m1", "m2"]
    assert g["m1"]["runs"] == 2 and g["m1"]["resolve_rate"] == 0.5
    assert g["m2"]["resolve_rate"] is None
    assert g["m1"]["share_with_finding"]["never_tested"] == 1.0
    import pytest

    with pytest.raises(ValueError):
        analyze_trajectories([a], group_by="colour")


def test_group_by_cli(capsys):
    assert main(["analyze", str(FX / "good.jsonl"), "--group-by", "source_format"]) == 0
    assert "source_format generic: 1 runs; with finding: none" in capsys.readouterr().out


def test_show_marks_evidence_and_errors(capsys):
    assert main(["show", str(FX / "loop.jsonl"), "--width", "60"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("loop (generic): 7 steps")
    assert any(line.startswith("  looping:") for line in out)
    step2 = [line for line in out if line.startswith("   2")][0]
    assert step2[4:6] == "!*" and len(step2) <= 60
    assert main(["show", str(FX / "good.jsonl")]) == 0
    assert "  no findings" in capsys.readouterr().out
    assert main(["show", "missing.jsonl"]) == 2


def test_loop_flags_reach_the_metrics(tmp_path, capsys):
    p = tmp_path / "page.jsonl"
    rows = [
        {"action": "scroll_down", "command": "scroll_down", "observation": str(k) * 300}
        for k in range(5)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def labels(rep):
        return [f["label"] for f in rep["runs"][0]["findings"]]

    assert "looping" not in labels(analyze_paths([p]))
    assert "looping" in labels(analyze_paths([p], opts=Options(same_observation=False)))
    assert main(["analyze", str(p), "--loop-any-observation"]) == 0
    assert "looping: exact repeat" in capsys.readouterr().out


def test_fail_on(capsys):
    loop = str(FX / "loop.jsonl")
    good = str(FX / "good.jsonl")
    assert main(["analyze", good, "--fail-on", "any"]) == 0
    assert main(["analyze", loop, good, "--fail-on", "looping,edited_tests"]) == 1
    assert "failing: 1 runs" in capsys.readouterr().err
    assert main(["analyze", loop, "--fail-on", "edited_tests"]) == 0
    assert main(["analyze", loop, "--fail-on", "nonsense"]) == 2

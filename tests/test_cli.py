import json
from pathlib import Path

from trajlens.cli import aggregate, analyze_paths, main

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
    rep = analyze_paths([FX / "near_loop.jsonl"], loop_min_steps=5)
    assert rep["runs"][0]["findings"] == []
    rep = analyze_paths([FX / "near_loop.jsonl"], near_threshold=0.99)
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

import json

from trajlens.analysis import analyze_trajectories
from trajlens.cli import main
from trajlens.metrics import find_loops
from trajlens.model import Step

OBS = "FAILED tests/test_calc.py::test_add - AssertionError\n1 failed in 0.1s"
BASE = "python -m pytest tests/test_calc.py"


def _rows():
    rows = [{"action": "bash", "command": "sed -i 's/a/b/' src/calc.py", "observation": ""}]
    for cmd in (BASE, BASE, BASE, BASE + " -x", BASE + " -q"):
        rows.append({"action": "bash", "command": cmd, "observation": OBS, "exit_code": 1})
    return rows


def _steps():
    return [
        Step(
            i,
            "edit" if i == 0 else "test",
            r["command"],
            observation=r["observation"],
            exit_code=r.get("exit_code"),
        )
        for i, r in enumerate(_rows())
    ]


def test_short_exact_loop_joins_longer_near_loop():
    exact, near = find_loops(_steps(), keep_steps=4)
    assert exact == []
    assert [lp.step_ids for lp in near] == [[1, 2, 3, 4, 5]]
    # Without keep_steps every exact loop holds its steps, as before.
    exact, near = find_loops(_steps())
    assert [lp.step_ids for lp in exact] == [[1, 2, 3]]
    assert near == []


def test_long_exact_loop_keeps_its_steps():
    steps = _steps()
    exact, near = find_loops(steps, keep_steps=3)
    assert [lp.step_ids for lp in exact] == [[1, 2, 3]]
    assert near == []


def test_exact_loop_kept_when_no_longer_near_group():
    exact, near = find_loops(_steps()[:4], keep_steps=4)
    assert [lp.step_ids for lp in exact] == [[1, 2, 3]]
    assert near == []


def test_labelled_looping(tmp_path, capsys):
    p = tmp_path / "near.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _rows()) + "\n")
    assert main(["audit", str(p), "--label", "looping"]) == 0
    assert "1 of 1 runs carry looping" in capsys.readouterr().out
    from trajlens.adapters.generic import load_generic

    run = analyze_trajectories([load_generic(p)])["runs"][0]
    loops = [f for f in run["findings"] if f["label"] == "looping"]
    assert [f["step_ids"] for f in loops] == [[1, 2, 3, 4, 5]]

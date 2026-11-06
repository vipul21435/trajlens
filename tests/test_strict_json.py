"""Reports are strict JSON: no Infinity or NaN tokens."""

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from trajlens.adapters.generic import step_from_record
from trajlens.analysis import Options, analyze_trajectories
from trajlens.cli import main
from trajlens.html import render_html
from trajlens.model import Trajectory


def strict(text: str):
    def bad(token):
        raise ValueError(token)

    return json.loads(text, parse_constant=bad)


def test_empty_head_observations_give_strict_json(tmp_path):
    rows = [
        {"action": "bash", "command": "ls", "observation": "", "exit_code": 0},
        {"action": "bash", "command": "ls", "observation": "", "exit_code": 0},
        {"action": "bash", "command": "ls", "observation": "ok", "exit_code": 0, "timestamp": 1},
    ]
    src = tmp_path / "x.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out, page = tmp_path / "r.json", tmp_path / "r.html"
    assert main(["analyze", str(src), "--json", str(out), "--html", str(page)]) == 0
    report = strict(out.read_text(encoding="utf-8"))
    assert report["runs"][0]["metrics"]["observation_growth"] is None
    blob = page.read_text(encoding="utf-8").split('id="report">')[1].split("</script>")[0]
    strict(blob)


number = st.none() | st.integers(-5, 5) | st.floats(allow_nan=True, allow_infinity=True)
record = st.fixed_dictionaries(
    {
        "action": st.sampled_from(["bash", "edit", "submit", "open"]),
        "command": st.sampled_from(["ls", "pytest", "python t.py", "edit 1:2", ""]),
        "observation": st.sampled_from(["", "ok", "Traceback (most recent call last)", "x" * 50]),
        "exit_code": number,
        "timestamp": number,
        "tokens_in": number,
    }
)


@settings(max_examples=200, deadline=None)
@given(st.lists(record, max_size=12))
def test_any_generic_run_gives_strict_json(rows):
    traj = Trajectory("r", [step_from_record(i, r) for i, r in enumerate(rows)])
    report = analyze_trajectories([traj], Options(), steps=True)
    strict(json.dumps(report, allow_nan=False))
    render_html(report)

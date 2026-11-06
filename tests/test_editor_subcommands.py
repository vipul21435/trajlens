import json

from trajlens.adapters.generic import classify_tool
from trajlens.adapters.sweagent import parse_sweagent
from trajlens.analysis import analyze_trajectories
from trajlens.cli import main
from trajlens.model import Step
from trajlens.scripts import created_files

ACTIONS = [
    "str_replace_editor view /testbed",
    "str_replace_editor create /testbed/reproduce.py --file_text 'import calc'",
    "python reproduce.py",
    "str_replace_editor str_replace /testbed/src/calc.py --old_str 'a' --new_str 'b'",
    "python reproduce.py",
    "str_replace_editor view /testbed/src/calc.py",
    "submit",
]
PATCH = "diff --git a/src/calc.py b/src/calc.py\n--- a/src/calc.py\n+++ b/src/calc.py\n"


def _traj_data():
    steps = [{"action": a, "observation": "ok", "thought": ""} for a in ACTIONS]
    return {"trajectory": steps, "info": {"submission": PATCH, "exit_status": "submitted"}}


def test_editor_subcommand_decides_the_class():
    for name in ("str_replace_editor", "str_replace_based_edit_tool"):
        assert classify_tool(name, f"{name} view /testbed/x.py") == "read"
        assert classify_tool(name, "view /testbed/x.py") == "read"
        for sub in ("create", "str_replace", "insert", "undo_edit"):
            assert classify_tool(name, f"{name} {sub} /testbed/x.py") == "edit"
    assert classify_tool("bash", "str_replace_editor view /a.py") == "read"
    assert classify_tool("", "str_replace_editor insert /a.py") == "edit"


def test_editor_create_counts_as_created_script():
    step = Step(0, "edit", "str_replace_editor create /testbed/check.py --file_text 'x'")
    assert created_files(step) == ["check.py"]


def test_sweagent_v1_view_is_not_an_edit(tmp_path, capsys):
    traj = parse_sweagent(_traj_data(), "v1")
    assert [s.tool for s in traj.steps] == [
        "read", "edit", "test", "edit", "test", "read", "submit"
    ]  # fmt: skip
    run = analyze_trajectories([traj])["runs"][0]
    assert run["metrics"]["edits"] == 2
    labels = {f["label"] for f in run["findings"]}
    assert "untested_final_edit" not in labels
    assert "never_tested" not in labels
    p = tmp_path / "v1.traj"
    p.write_text(json.dumps(_traj_data()))
    assert main(["analyze", str(p)]) == 0
    out = capsys.readouterr().out
    assert "untested_final_edit" not in out

from pathlib import Path

import pytest

from trajlens.adapters import load
from trajlens.adapters.generic import classify_tool, load_generic

FX = Path(__file__).parent / "fixtures"


def test_load_good_fixture():
    t = load_generic(FX / "good.jsonl")
    assert t.run_id == "good"
    assert t.task_id == "calc-1"
    assert t.model == "demo"
    assert len(t) == 7
    assert t.steps[0].index == 0
    assert t.steps[3].exit_code == 1 and t.steps[3].is_error
    assert t.steps[4].tool == "edit"
    assert t.steps[5].tool == "test"
    assert t.steps[6].tool == "submit"
    assert t.patch.startswith("diff --git")
    assert t.steps[0].tokens_in == 500 and t.steps[0].timestamp == 1000.0


def test_load_registry_and_unknown_format(tmp_path):
    t = load(FX / "empty.jsonl")
    assert t.run_id == "empty"
    with pytest.raises(ValueError):
        load(FX / "empty.jsonl", fmt="nope")


def test_bad_json_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"action": "bash"}\nnot json\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_generic(p)


def test_non_object_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="expected an object"):
        load_generic(p)


def test_blank_lines_and_meta_patch(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(
        '{"type": "meta", "run_id": "r1", "patch": "+++ b/a.py"}\n\n'
        '{"action": "bash", "command": "ls", "exit_code": "x", "timestamp": "y"}\n'
    )
    t = load_generic(p)
    assert t.run_id == "r1" and t.patch == "+++ b/a.py"
    assert t.steps[0].exit_code is None and t.steps[0].timestamp is None


@pytest.mark.parametrize(
    "action,command,expected",
    [
        ("edit", "", "edit"),
        ("str_replace_editor", "", "edit"),
        ("bash", "pytest -q", "test"),
        ("bash", "cd x && npm test", "test"),
        ("bash", "cargo test --all", "test"),
        ("bash", "sed -i 's/a/b/' f.py", "edit"),
        ("bash", "git apply fix.patch", "edit"),
        ("bash", "cat f.py", "read"),
        ("open", "f.py", "read"),
        ("bash", "grep -rn foo .", "search"),
        ("bash", "ls -la", "search"),
        ("bash", "python main.py", "shell"),
        ("submit", "", "submit"),
        ("bash", "submit", "submit"),
        ("", "", "other"),
        ("think", "", "other"),
    ],
)
def test_classify_tool(action, command, expected):
    assert classify_tool(action, command) == expected

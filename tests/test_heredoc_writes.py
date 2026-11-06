"""Shell writes through cat, echo, printf and tee are edits, end to end."""

import json

import pytest

from trajlens.adapters.generic import classify_tool, load_generic
from trajlens.analysis import Options, analyze_trajectories
from trajlens.model import Step
from trajlens.scripts import created_files, promote_script_runs


@pytest.mark.parametrize(
    "command",
    [
        "cat > check.py << 'EOF'\nimport calc\nEOF",
        "cat <<'EOF' > newfile.py\nprint(1)\nEOF",
        "cd /repo && cat > a.py <<EOF\nx = 1\nEOF",
        "cat >> notes.txt << EOF\nmore\nEOF",
        "echo 'x = 1' > conf.py",
        "printf 'a\\n' > data.txt",
        "tee out.py << EOF\nprint(2)\nEOF",
        "cat > test_new.py << 'EOF'\nimport pytest\nEOF",
    ],
)
def test_shell_writes_are_edits(command):
    assert classify_tool("bash", command) == "edit"


@pytest.mark.parametrize(
    "command",
    [
        "cat a.py",
        "cat a.py 2>/dev/null",
        "cat a.py > /dev/null",
        "cat a.py 2>&1 | head",
        "echo hello",
        "cat a.py | tee /dev/null",
    ],
)
def test_reads_and_prints_stay(command):
    assert classify_tool("bash", command) in ("read", "shell")


def test_created_files_any_class():
    assert created_files(Step(0, "edit", "cat > check.py << 'EOF'\nx\nEOF")) == ["check.py"]
    assert created_files(Step(0, "read", "cat a.py > b.py")) == ["b.py"]
    assert created_files(Step(0, "edit", "tee c.py << EOF\nx\nEOF")) == ["c.py"]


def test_heredoc_script_run_end_to_end(tmp_path):
    rows = [
        {"action": "bash", "command": "cat > check.py << 'EOF'\nimport calc\nEOF", "exit_code": 0},
        {"action": "bash", "command": "sed -i 's/a-b/a+b/' calc.py", "exit_code": 0},
        {"action": "bash", "command": "python check.py", "exit_code": 0},
        {"action": "submit", "command": "submit"},
        {"type": "patch", "patch": "diff --git a/c.py b/c.py\n--- a/c.py\n+++ b/c.py\n"},
    ]
    path = tmp_path / "heredoc.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    traj = load_generic(path)
    assert [s.tool for s in traj.steps] == ["edit", "edit", "shell", "submit"]
    assert [s.tool for s in promote_script_runs(traj).steps][2] == "test"
    report = analyze_trajectories([traj], Options())
    labels = {f["label"] for f in report["runs"][0]["findings"]}
    assert "never_tested" not in labels
    assert report["runs"][0]["metrics"]["edits"] == 2


def test_heredoc_body_is_not_searched():
    body = "cat > a.py << 'EOF'\nif x > y.py:\n    pass\nEOF\npython b.py > out.py"
    assert created_files(Step(0, "edit", body)) == ["a.py", "out.py"]
    assert created_files(Step(0, "edit", "cd /r\ncat > c.py << EOF\nx\nEOF")) == ["c.py"]


@pytest.mark.parametrize(
    "command,tool",
    [
        ("cat << 'EOF' | tee f.py\nx\nEOF", "edit"),
        ("echo 'a|b' > f.txt", "edit"),
        ("echo a > x && echo b > y", "edit"),
        ('echo "" > cfg && python app.py -h', "shell"),
        ("printf 'x\\n' | tool fix - > /tmp/o.sql", "shell"),
        ("echo x > cfg && pytest -q", "test"),
    ],
)
def test_only_pure_writes_are_edits(command, tool):
    assert classify_tool("bash", command) == tool

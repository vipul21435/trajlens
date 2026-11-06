from trajlens.analysis import Options, analyze_trajectories
from trajlens.model import Step, Trajectory
from trajlens.scripts import created_files, promote_script_runs, run_script


def st(i, tool, cmd):
    return Step(index=i, tool=tool, command=cmd)


def test_created_and_run_detection():
    assert created_files(st(0, "edit", "create /w/repo/check_fix.py")) == ["check_fix.py"]
    assert created_files(st(0, "edit", "write 'a/b.py'")) == ["b.py"]
    assert created_files(st(0, "shell", "cat << 'EOF' > /tmp/try_it.py\nprint(1)\nEOF")) == [
        "try_it.py"
    ]
    assert created_files(st(0, "edit", "edit 1:3")) == []
    assert created_files(st(0, "read", "cat a.py")) == []
    assert run_script(st(0, "shell", "cd /w && python3 -u /tmp/try_it.py --x 1")) == "try_it.py"
    assert run_script(st(0, "shell", "ls")) is None
    assert run_script(st(0, "test", "python reproduce.py")) is None


def test_promote_only_after_creation():
    steps = [
        st(0, "shell", "python convert.py"),
        st(1, "edit", "create convert.py"),
        st(2, "edit", "edit 1:1"),
        st(3, "shell", "python convert.py"),
        st(4, "shell", "python other.py"),
    ]
    t = Trajectory("t", steps)
    out = promote_script_runs(t)
    assert [s.tool for s in out.steps] == ["shell", "edit", "edit", "test", "shell"]
    assert t.steps[3].tool == "shell"  # input untouched
    plain = Trajectory("p", [st(0, "shell", "ls")])
    assert promote_script_runs(plain) is plain


def test_analysis_option():
    steps = [st(0, "edit", "create check.py"), st(1, "edit", "edit 1:1"), st(2, "shell",
             "python check.py")]  # fmt: skip
    t = Trajectory("t", steps, patch="diff --git a/x.py b/x.py\n")
    labels = {f["label"] for f in analyze_trajectories([t])["runs"][0]["findings"]}
    assert "never_tested" not in labels
    off = analyze_trajectories([t], Options(script_tests=False))["runs"][0]["findings"]
    assert "never_tested" in {f["label"] for f in off}


def test_cli_flag(tmp_path, capsys):
    import json

    from trajlens.cli import main

    p = tmp_path / "r.jsonl"
    rows = [{"action": "create", "command": "create check.py"},
            {"action": "bash", "command": "python check.py"}]  # fmt: skip
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert main(["analyze", str(p)]) == 0
    assert "1 test runs" in capsys.readouterr().out
    assert main(["analyze", str(p), "--no-script-tests"]) == 0
    assert "0 test runs" in capsys.readouterr().out

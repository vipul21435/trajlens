import json

from trajlens.adapters.chat import parse_chat
from trajlens.adapters.minisweagent import parse_minisweagent
from trajlens.analysis import analyze_trajectories
from trajlens.cli import main

DIFF = (
    "diff --git a/src/div.py b/src/div.py\n--- a/src/div.py\n+++ b/src/div.py\n"
    "@@ -1 +1 @@\n-def division(a, b)\n+def division(a, b):\n"
)
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"


def _messages(final: str):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Fix the syntax error."},
        {"role": "assistant", "content": "Fix it.\n\n```bash\nsed -i 's/b)$/b):/' src/div.py\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\n</output>"},
        {"role": "assistant", "content": f"Submit.\n\n```bash\n{SUBMIT}\n```"},
        {"role": "user", "content": final},
    ]


def _labels(traj):
    return {f["label"] for f in analyze_trajectories([traj])["runs"][0]["findings"]}


def test_message_list_patch_from_submit_output():
    traj = parse_minisweagent(_messages(DIFF), "m")
    assert traj.patch == DIFF
    assert "empty_submission" not in _labels(traj)


def test_submit_output_without_diff_stays_empty():
    traj = parse_minisweagent(_messages(""), "m")
    assert traj.patch == ""
    assert "empty_submission" in _labels(traj)


def test_info_submission_wins():
    data = {"messages": _messages(DIFF), "info": {"submission": "diff --git a/x b/x\n"}}
    assert parse_minisweagent(data, "m").patch == "diff --git a/x b/x\n"


def test_chat_marker_submit_reads_patch():
    traj = parse_chat(_messages(DIFF), "c")
    assert traj.patch == DIFF
    assert "empty_submission" not in _labels(traj)


def test_cli_patch_files(tmp_path, capsys):
    p = tmp_path / "run.traj.json"
    p.write_text(json.dumps(_messages(DIFF)))
    assert main(["analyze", str(p), "--format", "minisweagent"]) == 0
    out = capsys.readouterr().out
    assert "patch files 1" in out
    assert "empty_submission" not in out

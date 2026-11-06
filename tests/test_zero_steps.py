"""A run with no steps is not a clean run."""

from trajlens.analysis import Options, analyze_trajectories
from trajlens.cli import format_text
from trajlens.model import Trajectory


def test_zero_steps_empty_patch_is_gave_up():
    report = analyze_trajectories([Trajectory("z", [])], Options())
    run = report["runs"][0]
    assert [(f["label"], f["step_ids"]) for f in run["findings"]] == [("gave_up", [])]
    assert report["aggregate"]["runs_no_finding"] == 0
    text = format_text(report)
    assert "warning: z has no steps" in text


def test_zero_steps_with_patch_gets_only_the_warning():
    report = analyze_trajectories([Trajectory("p", [], patch="diff --git a/a b/a\n")], Options())
    assert report["runs"][0]["findings"] == []
    assert "warning: p has no steps" in format_text(report)

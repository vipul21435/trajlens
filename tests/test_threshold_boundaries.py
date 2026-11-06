"""Each documented "at least" or "within" threshold, at the value and one below it."""

from trajlens.classify import classify_run
from trajlens.metrics import find_loops, near_duplicate, observation_growth, run_metrics
from trajlens.model import Step, Trajectory

FILLERS = ["ls", "pwd", "whoami", "date", "uname -a", "env", "df -h", "id", "hostname", "uptime"]


def _labels(steps, **kw):
    traj = Trajectory("t", steps, patch="diff --git a/x.py b/x.py\n")
    return {f.label for f in classify_run(traj, run_metrics(traj), **kw)}


def _failing(n):
    cmds = ["cat a.py", "grep -rn foo", "python run.py", "make build", "ls /nope", "head b.txt"]
    return [Step(i, "shell", cmds[i], observation=f"err {i}", exit_code=1) for i in range(n)]


def test_cascade_at_cascade_min():
    assert "tool_error_cascade" in _labels(_failing(4))
    assert "tool_error_cascade" not in _labels(_failing(3))
    assert "tool_error_cascade" in _labels(_failing(3), cascade_min=3)


def test_context_chars_reached_exactly():
    steps = [Step(0, "read", "cat a", observation="x" * 50),
             Step(1, "read", "cat b", observation="y" * 50)]  # fmt: skip
    assert "context_blow_up" in _labels(steps, context_chars=100, blowup_chars=1000)
    steps[1].observation = "y" * 49
    assert "context_blow_up" not in _labels(steps, context_chars=100, blowup_chars=1000)


def test_blowup_chars_reached_exactly():
    steps = [Step(0, "read", "cat a", observation="x" * 30)]
    assert "context_blow_up" in _labels(steps, context_chars=1000, blowup_chars=30)
    assert "context_blow_up" not in _labels(steps, context_chars=1000, blowup_chars=31)


def _spaced(gap, repeats=3):
    steps, fill = [], iter(FILLERS)
    for k in range(repeats):
        steps.append(Step(len(steps), "shell", "python check.py", observation="same"))
        if k < repeats - 1:
            for _ in range(gap - 1):
                steps.append(Step(len(steps), "shell", next(fill), observation=f"o{len(steps)}"))
    return steps


def test_window_boundary():
    exact, _ = find_loops(_spaced(5), window=6)
    assert [lp.step_ids for lp in exact] == [[0, 5, 10]]
    exact, near = find_loops(_spaced(6), window=6)
    assert exact == [] and near == []


def test_near_ratio_at_threshold():
    a, b = "abcdefghij", "abcdefghik"  # ratio 2 * 9 / 20 = 0.9
    assert near_duplicate(a, b, 0.9)
    assert not near_duplicate(a, b, 0.91)


def test_observation_similarity_at_threshold():
    obs = ["abcdefghij", "abcdefghik", "abcdefghik"]
    steps = [Step(i, "shell", "python check.py", observation=o) for i, o in enumerate(obs)]
    exact, _ = find_loops(steps, near_threshold=0.9)
    assert [lp.step_ids for lp in exact] == [[0, 1, 2]]
    exact, _ = find_loops(steps, near_threshold=0.91)
    assert exact == []


def test_loop_min_steps_boundary():
    steps = [Step(i, "shell", "python check.py", observation="same") for i in range(4)]
    assert "looping" in _labels(steps, loop_min_steps=4)
    assert "looping" not in _labels(steps, loop_min_steps=5)


def test_growth_from_three_steps():
    steps = [Step(i, "shell", f"c{i}", observation="x" * n) for i, n in enumerate((10, 20, 30))]
    assert observation_growth(steps) == 3.0
    assert observation_growth(steps[:2]) is None

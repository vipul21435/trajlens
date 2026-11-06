from trajlens.metrics import run_metrics
from trajlens.model import Step, Trajectory


def test_failed_edits_and_failed_test_runs_count_only_their_class():
    rows = [
        ("edit", "edit 1:2", 1),
        ("edit", "edit 3:4", 0),
        ("test", "pytest tests/test_a.py", 1),
        ("test", "pytest tests/test_b.py", 0),
        ("shell", "ls missing", 2),
        ("read", "cat nope.py", 1),
        ("search", "grep -rn foo src", 0),
    ]
    steps = [Step(i, t, c, exit_code=e) for i, (t, c, e) in enumerate(rows)]
    m = run_metrics(Trajectory("t", steps))
    assert m.edits == 2
    assert m.failed_edits == 1
    assert m.test_runs == 2
    assert m.failed_test_runs == 1
    assert m.error_steps == 4

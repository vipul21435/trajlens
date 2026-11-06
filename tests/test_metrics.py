from pathlib import Path

from trajlens.adapters import load
from trajlens.metrics import (
    error_streaks,
    find_loops,
    is_test_path,
    near_duplicate,
    normalise_command,
    observation_growth,
    patch_paths,
    run_metrics,
)
from trajlens.model import Step, Trajectory

FX = Path(__file__).parent / "fixtures"


def mk(cmds, exit_codes=None, obs=None, tools=None):
    steps = []
    for i, c in enumerate(cmds):
        steps.append(
            Step(
                index=i,
                tool=(tools[i] if tools else "shell"),
                command=c,
                observation=(obs[i] if obs else ""),
                exit_code=(exit_codes[i] if exit_codes else 0),
            )
        )
    return steps


def test_normalise_and_near():
    assert normalise_command("  Pytest   -q ") == "pytest -q"
    assert near_duplicate("abc", "abc", 0.9)
    assert not near_duplicate("", "abc", 0.9)
    assert near_duplicate("grep foo src/a.py", "grep foo src/b.py", 0.85)
    assert not near_duplicate("ls", "pytest -q tests", 0.9)


def test_exact_loop_within_window():
    steps = mk(["ls", "cat a", "ls", "ls", "cat b", "ls", "pytest"])
    exact, near = find_loops(steps, window=4, min_repeats=3)
    assert len(exact) == 1
    assert exact[0].step_ids == [0, 2, 3, 5]
    assert exact[0].command == "ls"
    assert near == []


def test_no_loop_when_spread_beyond_window():
    steps = mk(["ls", "a", "b", "c", "d", "ls", "e", "f", "g", "h", "ls"])
    exact, near = find_loops(steps, window=3, min_repeats=3)
    assert exact == [] and near == []


def test_near_loop_excludes_exact_members():
    steps = mk(["grep x a.py", "grep x b.py", "grep x c.py", "ls", "ls", "ls"])
    exact, near = find_loops(steps, window=6, min_repeats=3, near_threshold=0.8)
    assert [lp.step_ids for lp in exact] == [[3, 4, 5]]
    assert [lp.step_ids for lp in near] == [[0, 1, 2]]


def test_empty_commands_never_loop():
    steps = mk(["", "", "", ""])
    assert find_loops(steps) == ([], [])


def test_error_streaks():
    steps = mk(["a", "b", "c", "d", "e"], exit_codes=[1, 1, 0, 1, None])
    assert error_streaks(steps) == [[0, 1], [3]]


def test_observation_growth():
    assert observation_growth(mk(["a", "b"])) is None
    s = mk(["a"] * 6, obs=["x", "x", "xx", "xx", "xxxx", "xxxx"])
    assert observation_growth(s) == 4.0
    s = mk(["a"] * 3, obs=["", "", ""])
    assert observation_growth(s) is None
    s = mk(["a"] * 3, obs=["", "", "zz"])
    assert observation_growth(s) == float("inf")


def test_patch_paths_and_test_detection():
    patch = (
        "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n"
        "diff --git a/tests/test_x.py b/tests/test_x.py\n"
    )
    assert patch_paths(patch) == ["src/x.py", "tests/test_x.py"]
    assert patch_paths("--- a/f.py\n+++ b/f.py\n+++ /dev/null\n") == ["f.py"]
    assert patch_paths("") == []
    assert is_test_path("tests/test_x.py")
    assert is_test_path("pkg/x_test.go")
    assert is_test_path("src/foo.spec.ts")
    assert is_test_path("conftest.py")
    assert not is_test_path("src/testing_tools.py")
    assert not is_test_path("src/contest.py")
    assert not is_test_path("src/x.py")


def test_run_metrics_good_fixture():
    m = run_metrics(load(FX / "good.jsonl"))
    assert m.steps == 7
    assert m.edits == 1 and m.test_runs == 2
    assert m.edits_per_test_run == 0.5
    assert m.first_edit_step == 4 and m.first_test_step == 3
    assert m.tests_after_last_edit is True
    assert m.last_edit_step == 4 and m.last_test_step == 5
    assert m.patch_files == ["src/calc/parse.py"]
    assert m.patch_touches_tests is False and m.patch_empty is False
    assert m.submitted is True
    assert m.error_steps == 1 and m.max_error_streak == 1
    assert m.duration_seconds == 18.0
    assert m.tokens_in == sum(500 + 40 * i for i in range(7)) and m.tokens_out == 420
    assert m.tool_mix["search"] == 2 and m.tool_mix["read"] == 1
    d = m.to_dict()
    assert d["run_id"] == "good" and isinstance(d["exact_loops"], list)


def test_run_metrics_empty_trajectory():
    m = run_metrics(Trajectory(run_id="none"))
    assert m.steps == 0 and m.error_rate == 0.0
    assert m.largest_observation_step is None
    assert m.tests_after_last_edit is None
    assert m.duration_seconds is None and m.tokens_in is None
    assert m.patch_empty is True


def test_run_metrics_loop_fixture():
    m = run_metrics(load(FX / "loop.jsonl"))
    assert m.looped_steps == 5
    assert m.exact_loops[0].step_ids == [2, 3, 4, 5, 6]
    assert m.tests_after_last_edit is False
    assert m.max_error_streak == 5


def test_observation_looks_failed():
    from trajlens.metrics import observation_looks_failed

    assert observation_looks_failed('Traceback (most recent call last):\n  File "x"')
    assert observation_looks_failed("ModuleNotFoundError: No module named 'x'")
    assert observation_looks_failed("bash: foo: command not found")
    assert observation_looks_failed("cat: a.txt: No such file or directory")
    assert observation_looks_failed("Your proposed edit has introduced new syntax error(s).")
    assert observation_looks_failed("Your output was not formatted correctly. You must")
    assert not observation_looks_failed("All 3 tests passed")
    assert not observation_looks_failed("")


def test_step_errors_modes():
    import pytest

    from trajlens.metrics import step_errors

    no_codes = [
        Step(index=0, tool="shell", command="python a.py", observation="ValueError: bad"),
        Step(index=1, tool="shell", command="ls", observation="a.py"),
    ]
    assert step_errors(no_codes, "auto") == [True, False]
    assert step_errors(no_codes, "never") == [False, False]
    mixed = [
        Step(index=0, tool="shell", command="x", observation="ValueError: bad", exit_code=0),
        Step(index=1, tool="shell", command="y", observation="ValueError: bad"),
    ]
    # auto trusts the format once any exit code is recorded
    assert step_errors(mixed, "auto") == [False, False]
    assert step_errors(mixed, "always") == [False, True]
    with pytest.raises(ValueError):
        step_errors(mixed, "sometimes")


TB = "Traceback (most recent call last):"


def test_run_metrics_inferred_errors():
    steps = [
        Step(index=0, tool="shell", command="python r.py", observation=TB),
        Step(index=1, tool="shell", command="python r.py", observation="NameError: x"),
        Step(index=2, tool="shell", command="ls", observation="r.py"),
    ]
    m = run_metrics(Trajectory("t", steps))
    assert m.error_steps == 2 and m.error_step_ids == [0, 1]
    assert m.errors_inferred and not m.last_step_error
    assert m.max_error_streak == 2
    m2 = run_metrics(Trajectory("t", steps), infer_errors="never")
    assert m2.error_steps == 0 and not m2.errors_inferred


def test_edit_between_repeats_ends_group():
    tools = ["test", "edit", "test", "edit", "test", "edit", "test"]
    cmds = ["python r.py", "edit 1:1", "python r.py", "edit 2:2", "python r.py", "edit 3:3"]
    cmds.append("python r.py")
    steps = mk(cmds, tools=tools)
    exact, near = find_loops(steps, window=6, min_repeats=3, near_threshold=0.99)
    assert exact == [] and near == []
    exact, _ = find_loops(steps, window=6, min_repeats=3, edit_resets=False)
    assert [lp.step_ids for lp in exact] == [[0, 2, 4, 6]]


def test_repeated_edits_still_loop():
    steps = mk(["edit 5:5", "edit 5:5", "edit 5:5"], tools=["edit"] * 3)
    exact, _ = find_loops(steps, window=6, min_repeats=3)
    assert [lp.step_ids for lp in exact] == [[0, 1, 2]]

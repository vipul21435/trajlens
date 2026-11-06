from pathlib import Path

from trajlens.adapters import load
from trajlens.classify import FAILURE_CLASSES, classify_run
from trajlens.metrics import run_metrics
from trajlens.model import Step, Trajectory

FX = Path(__file__).parent / "fixtures"


def labels(name, **kw):
    t = load(FX / name)
    m = run_metrics(t)
    return {f.label: f for f in classify_run(t, m, **kw)}


def test_good_run_has_no_findings():
    assert labels("good.jsonl") == {}


def test_loop_fixture():
    f = labels("loop.jsonl")
    assert set(f) == {"looping", "never_tested", "gave_up", "tool_error_cascade"}
    assert f["looping"].step_ids == [2, 3, 4, 5, 6]
    assert f["never_tested"].step_ids == [1]
    assert f["gave_up"].step_ids == [6]
    assert f["tool_error_cascade"].step_ids == [2, 3, 4, 5, 6]
    assert "exact" in f["looping"].detail


def test_edited_tests_and_blow_up():
    f = labels("edited_tests.jsonl")
    assert set(f) == {"context_blow_up", "edited_tests"}
    assert f["context_blow_up"].step_ids == [3]
    assert "tests/test_parse.py" in f["edited_tests"].detail
    assert f["edited_tests"].step_ids == []
    # raise the floor and the blow-up disappears
    f2 = labels("edited_tests.jsonl", blowup_chars=10**6, growth_min_chars=10**6)
    assert set(f2) == {"edited_tests"}


def test_near_loop_threshold_flag():
    assert "looping" in labels("near_loop.jsonl")
    assert "looping" not in labels("near_loop.jsonl", loop_min_steps=5)


def test_empty_submission():
    f = labels("empty.jsonl")
    assert set(f) == {"empty_submission"}
    assert f["empty_submission"].step_ids == [1]


def test_untested_final_edit(tmp_path):
    p = tmp_path / "u.jsonl"
    p.write_text(
        '{"action":"bash","command":"pytest -q"}\n'
        '{"action":"edit","command":"a.py"}\n'
        '{"action":"submit"}\n'
        '{"type":"patch","patch":"diff --git a/a.py b/a.py\\n"}\n'
    )
    t = load(p)
    f = {x.label: x for x in classify_run(t, run_metrics(t))}
    assert set(f) == {"untested_final_edit"}
    assert f["untested_final_edit"].step_ids == [1, 0]


def test_gave_up_on_last_error(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"action":"bash","command":"python x.py","exit_code":2}\n'
        '{"type":"patch","patch":"diff --git a/a.py b/a.py\\n"}\n'
    )
    t = load(p)
    f = {x.label: x for x in classify_run(t, run_metrics(t))}
    assert set(f) == {"gave_up"}
    assert "exit code 2" in f["gave_up"].detail


def test_all_labels_documented():
    import trajlens.classify as c

    for lab in FAILURE_CLASSES:
        assert lab in c.__doc__
    for f in labels("loop.jsonl").values():
        assert f.to_dict()["label"] in FAILURE_CLASSES


def test_gave_up_on_inferred_last_error():
    steps = [Step(index=0, tool="shell", command="python r.py", observation="KeyError: 'a'")]
    t = Trajectory("t", steps, patch="diff --git a/x.py b/x.py\n")
    f = {x.label: x for x in classify_run(t, run_metrics(t))}
    assert "observation reads as an error" in f["gave_up"].detail


def test_loop_detail_is_short():
    cmd = "edit 1:3\n" + "x = 1\n" * 50 + "end_of_edit"
    steps = [Step(index=i, tool="edit", command=cmd) for i in range(4)]
    t = Trajectory("t", steps)
    f = [x for x in classify_run(t, run_metrics(t)) if x.label == "looping"][0]
    assert "'edit 1:3 ...'" in f.detail and len(f.detail) < 120

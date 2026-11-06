"""Malformed input ends in a one-line ``trajlens:`` message naming the file."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trajlens.adapters.generic import step_from_record
from trajlens.adapters.openhands import parse_openhands
from trajlens.adapters.sweagent import parse_sweagent
from trajlens.cli import main

BAD = {
    "inf.jsonl": '{"action":"bash","command":"ls","exit_code":Infinity}\n',
    "numaction.traj": '{"trajectory":[{"action":5,"observation":"x"}]}',
    "noaction.traj": '{"history":[{"role":"assistant","message_type":"action","content":"hi"},'
    '{"message_type":"observation","content":"x"}]}',
    "listid.json": '[{"id":[1],"action":"run","args":{"command":"ls"}},'
    '{"cause":[1],"observation":"run","content":"x"}]',
    "trunc.traj": '{"trajectory":[\n',
    "trunc.traj.json": '{"messages":[\n',
    "bin.traj": "\udcff\udcfe",
    "list.json": "[1,2,3]",
}


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8", "surrogateescape"))
    return path


@pytest.mark.parametrize("name", sorted(BAD))
def test_bad_file_gives_a_message(tmp_path, capsys, name):
    path = write(tmp_path, name, BAD[name])
    code = main(["analyze", str(path)])
    err = capsys.readouterr().err
    # Some of these files are readable after the fix (an int action, an
    # unhashable id); the rest must fail with a message naming the file.
    assert code in (0, 2)
    if code == 2:
        assert err.startswith("trajlens: ") and str(path) in err
        assert "Traceback" not in err


def test_values_that_used_to_crash_now_parse():
    assert step_from_record(0, {"action": "bash", "exit_code": float("inf")}).exit_code is None
    assert step_from_record(0, {"action": "bash", "timestamp": float("nan")}).timestamp is None
    t = parse_sweagent({"trajectory": [{"action": 5}]}, "x")
    assert t.steps[0].command == "5"
    t = parse_sweagent(
        {"history": [{"role": "assistant", "message_type": "action", "content": "hi"}]}, "x"
    )
    assert len(t.steps) == 1
    t = parse_openhands(
        [
            {"id": [1], "action": "run", "args": {"command": "ls"}},
            {"cause": [1], "observation": "run", "content": "x"},
        ],
        "x",
    )
    assert t.steps[0].command == "ls"


def test_skip_bad_continues(tmp_path, capsys):
    good = write(tmp_path, "good.jsonl", '{"action":"bash","command":"ls","exit_code":0}\n')
    bad = write(tmp_path, "bad.traj", BAD["trunc.traj"])
    assert main(["analyze", str(bad), str(good)]) == 2
    capsys.readouterr()
    assert main(["analyze", "--skip-bad", str(bad), str(good)]) == 0
    out, err = capsys.readouterr()
    assert "runs: 1" in out and "skipped" in err and str(bad) in err


@pytest.mark.parametrize("doc", ['{"runs":5}', "[1,2,3]", '{"runs":[1]}', '{"runs":[{"x":1}]}'])
def test_compare_bad_report(tmp_path, capsys, doc):
    a = write(tmp_path, "a.json", doc)
    assert main(["compare", str(a), str(a)]) == 2
    assert capsys.readouterr().err.startswith("trajlens: ")


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=8),
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=4)
    ),
    max_leaves=12,
)
step_keys = st.sampled_from(
    [
        "action",
        "command",
        "observation",
        "exit_code",
        "timestamp",
        "tokens_in",
        "thought",
        "tool",
        "id",
        "cause",
        "args",
        "message_type",
        "role",
        "content",
        "source",
        "extras",
    ]
)
records = st.dictionaries(step_keys, json_values, max_size=6)


@settings(max_examples=300, deadline=None)
@given(st.lists(records, max_size=5))
def test_adapters_raise_only_value_error(items):
    for parse in (
        lambda: [step_from_record(0, r) for r in items],
        lambda: parse_sweagent({"trajectory": items}, "x"),
        lambda: parse_sweagent({"history": items}, "x"),
        lambda: parse_openhands(items, "x"),
    ):
        try:
            parse()
        except ValueError:
            pass

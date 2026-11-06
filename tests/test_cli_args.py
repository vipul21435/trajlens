"""Argument checks of the analyze command."""

import pytest

from trajlens.cli import main

GOOD = "tests/fixtures/loop.jsonl"


def test_unwritable_output_is_a_message(tmp_path, capsys):
    target = tmp_path / "missing" / "r.json"
    assert main(["analyze", GOOD, "--json", str(target)]) == 2
    assert capsys.readouterr().err.startswith("trajlens: ")


def test_fail_on_names_are_stripped(capsys):
    assert main(["analyze", GOOD, "--fail-on", "looping, gave_up"]) == 1
    assert "unknown" not in capsys.readouterr().err


def test_unknown_fail_on_is_checked_before_writing(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main(["analyze", GOOD, "--json", str(out), "--fail-on", "lopping"]) == 2
    assert "unknown class lopping" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--window", "-1"),
        ("--min-repeats", "0"),
        ("--min-repeats", "1"),
        ("--loop-min-steps", "1"),
        ("--cascade-min", "0"),
        ("--blowup-chars", "0"),
        ("--context-chars", "-5"),
        ("--near-threshold", "1.5"),
        ("--near-threshold", "-0.1"),
        ("--window", "x"),
    ],
)
def test_thresholds_are_checked(flag, value, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["analyze", GOOD, flag, value])
    assert exc.value.code == 2

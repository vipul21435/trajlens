from pathlib import Path

import pytest

from trajlens.adapters import load
from trajlens.adapters.generic import dump_generic, load_generic
from trajlens.cli import main

FX = Path(__file__).parent / "fixtures"
SOURCES = ["sample.traj", "sample.traj.json", "sample_openhands.json", "good.jsonl"]


@pytest.mark.parametrize("name", SOURCES)
def test_round_trip(tmp_path, name):
    t = load(FX / name)
    t.resolved = True
    out = tmp_path / "r.jsonl"
    dump_generic(t, out)
    back = load_generic(out)
    assert back.run_id == t.run_id and back.patch == t.patch and back.resolved is True
    assert back.steps == t.steps


def test_convert_cli(tmp_path, capsys):
    out = tmp_path / "conv"
    args = ["convert", str(FX / "sample.traj"), str(FX / "sample.traj"), "--out", str(out)]
    assert main(args) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["sample-2.jsonl", "sample.jsonl"]
    assert "wrote 2 files" in capsys.readouterr().out
    assert main(["convert", str(tmp_path / "missing.traj"), "--out", str(out)]) == 2

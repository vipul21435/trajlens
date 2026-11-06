import json
import math
import random
from pathlib import Path

import pytest

from trajlens.cli import main
from trajlens.compare import compare_reports, newcombe_diff, wilson

FX = Path(__file__).parent / "fixtures"


def test_wilson_known_values():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3
    lo, hi = wilson(5, 10)
    assert math.isclose(lo + hi, 1.0)
    with pytest.raises(ValueError):
        wilson(1, 0)
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_newcombe_properties():
    rng = random.Random(0)
    for _ in range(300):
        na, nb = rng.randint(1, 60), rng.randint(1, 60)
        ka, kb = rng.randint(0, na), rng.randint(0, nb)
        d, lo, hi = newcombe_diff(ka, na, kb, nb)
        assert -1.0 <= lo <= d <= hi <= 1.0
        d2, lo2, hi2 = newcombe_diff(kb, nb, ka, na)
        assert math.isclose(d2, -d) and math.isclose(lo2, -hi) and math.isclose(hi2, -lo)


def test_newcombe_closed_form_by_hand():
    # 56/70 against 48/80, worked step by step outside the implementation.
    # Wilson limits with z = 1.96, from (p + z^2/2n +- z sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n):
    # (centre and half-width below are already divided by 1 + z^2/n)
    #   56/70: p = 0.8, 1 + z^2/n = 1.05488, centre 0.78439, half 0.09256 -> [0.6918, 0.8770]
    #   48/80: p = 0.6, 1 + z^2/n = 1.04802, centre 0.59542, half 0.10496 -> [0.4905, 0.7004]
    # Hybrid score limits for d = 0.8 - 0.6 = 0.2:
    #   lower = 0.2 - sqrt((0.8 - 0.6918)^2 + (0.7004 - 0.6)^2) = 0.2 - 0.1476 = 0.0524
    #   upper = 0.2 + sqrt((0.8770 - 0.8)^2 + (0.6 - 0.4905)^2) = 0.2 + 0.1339 = 0.3339
    lo1, hi1 = wilson(56, 70)
    lo2, hi2 = wilson(48, 80)
    assert abs(lo1 - 0.6918) < 1e-4 and abs(hi1 - 0.8770) < 1e-4
    assert abs(lo2 - 0.4905) < 1e-4 and abs(hi2 - 0.7004) < 1e-4
    lower = 0.2 - math.sqrt((0.8 - 0.6918) ** 2 + (0.7004 - 0.6) ** 2)
    upper = 0.2 + math.sqrt((0.8770 - 0.8) ** 2 + (0.6 - 0.4905) ** 2)
    d, lo, hi = newcombe_diff(48, 80, 56, 70)
    assert math.isclose(d, 0.2)
    assert abs(lo - lower) < 2e-4 and abs(hi - upper) < 2e-4
    assert abs(lo - 0.0524) < 1e-4 and abs(hi - 0.3339) < 1e-4


def test_newcombe_coverage_simulation():
    rng = random.Random(1)
    pa, pb, na, nb, reps = 0.3, 0.45, 80, 60, 2000
    hit = 0
    for _ in range(reps):
        ka = sum(rng.random() < pa for _ in range(na))
        kb = sum(rng.random() < pb for _ in range(nb))
        _, lo, hi = newcombe_diff(ka, na, kb, nb)
        hit += lo <= pb - pa <= hi
    assert 0.93 <= hit / reps <= 0.975


def test_compare_cli(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert main(["analyze", str(FX / "good.jsonl"), "--json", str(a)]) == 0
    assert main(["analyze", str(FX / "loop.jsonl"), "--json", str(b)]) == 0
    capsys.readouterr()
    assert main(["compare", str(a), str(b), "--names", "good", "loop"]) == 0
    out = capsys.readouterr().out
    assert "good - " not in out and "loop - good" in out
    assert "looping" in out and "0/1" in out and "1/1" in out
    rows = compare_reports(json.loads(a.read_text()), json.loads(b.read_text()))
    assert [r["label"] for r in rows][0] == "looping" and rows[0]["diff"] == 1.0
    empty = tmp_path / "e.json"
    empty.write_text(json.dumps({"runs": []}))
    assert main(["compare", str(a), str(empty)]) == 2

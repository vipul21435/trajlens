import json

from trajlens.adapters.generic import load_generic, parse_stamp
from trajlens.metrics import run_metrics


def test_parse_stamp_forms():
    assert parse_stamp(12.5) == 12.5
    assert parse_stamp("12.5") == 12.5
    a = parse_stamp("2026-01-02T10:00:00Z")
    b = parse_stamp("2026-01-02T10:01:30.500000+00:00")
    assert a is not None and b is not None and b - a == 90.5
    assert parse_stamp("not a time") is None
    assert parse_stamp(None) is None
    assert parse_stamp(float("inf")) is None


def test_generic_iso_timestamps_give_a_duration(tmp_path):
    p = tmp_path / "iso.jsonl"
    rows = [
        {"action": "bash", "command": "ls", "timestamp": "2026-01-02T10:00:00Z"},
        {"action": "bash", "command": "pytest", "timestamp": "2026-01-02T10:02:00Z"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    t = load_generic(p)
    assert run_metrics(t).duration_seconds == 120.0

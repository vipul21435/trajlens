"""Run fields on the generic meta line."""

import json

from trajlens.adapters.generic import load_generic


def load_meta(tmp_path, meta):
    path = tmp_path / "run7.jsonl"
    rows = [{"type": "meta", **meta}, {"action": "bash", "command": "ls"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return load_generic(path)


def test_null_fields_keep_their_defaults(tmp_path):
    t = load_meta(tmp_path, {"run_id": None, "task_id": None, "model": None})
    assert (t.run_id, t.task_id, t.model) == ("run7", "", "")


def test_resolved_accepts_one_and_zero(tmp_path):
    assert load_meta(tmp_path, {"resolved": 1}).resolved is True
    assert load_meta(tmp_path, {"resolved": 0}).resolved is False
    assert load_meta(tmp_path, {"resolved": "yes"}).resolved is None
    assert load_meta(tmp_path, {"resolved": True}).resolved is True

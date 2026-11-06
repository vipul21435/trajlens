import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wilson_interval():
    rd = _module("real_data_report")
    lo, hi = rd.wilson(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.28
    lo, hi = rd.wilson(5, 10)
    assert math.isclose(lo + hi, 1.0) and 0.23 < lo < 0.24
    assert all(math.isnan(x) for x in rd.wilson(0, 0))


def test_agreement_counts():
    rd = _module("real_data_report")
    runs = [
        {"findings": [{"label": "gave_up"}], "exit_status": "early_exit"},
        {"findings": [{"label": "gave_up"}], "exit_status": "submitted"},
        {"findings": [], "exit_status": "exit_cost"},
        {"findings": [], "exit_status": "submitted"},
    ]
    g = rd.agreement(runs, "gave_up", lambda r: not r["exit_status"].startswith("submitted"))
    assert (g["tp"], g["fp"], g["fn"], g["tn"]) == (1, 1, 1, 1)
    assert g["precision"] == 0.5 and g["recall"] == 0.5


def test_report_script_on_tiny_sample(tmp_path, capsys, monkeypatch):
    rd = _module("real_data_report")
    fence = "```"
    for i, (target, status) in enumerate([(True, "submitted"), (False, "exit_context")]):
        row = {
            "run_id": f"row{i:05d}",
            "target": target,
            "exit_status": status,
            "generated_patch": "",
            "trajectory": [
                {"role": "user", "text": "ISSUE"},
                {"role": "ai", "text": f"go\n{fence}\nls\n{fence}"},
                {"role": "user", "text": "a.py"},
            ],
        }
        (tmp_path / f"row{i:05d}.json").write_text(json.dumps(row))
    manifest = {"rows_total": 2, "dataset": "d", "seed": 0}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("sys.argv", ["x", str(tmp_path), "--tuning", "1"])
    rd.main()
    out = capsys.readouterr().out
    assert "tuning rows 1, held-out rows 1" in out
    assert "| gave_up | 1/1 = 100.0%" in out
    assert "context_blow_up vs exit_context | held-out" in out
    # No failure was read from text, so the labels drop "(inferred)".
    assert "| mean error rate | 0.000 |" in out and "(inferred)" not in out
    assert "nan" not in out
    monkeypatch.setattr("sys.argv", ["x", str(tmp_path), "--tuning", "0"])
    rd.main()
    out = capsys.readouterr().out
    assert "held-out rows only" not in out and "| gave_up vs not submitted | tuning" not in out


def test_num_formats_missing_values():
    rd = _module("real_data_report")
    assert rd._num(None) == "-" and rd._num(math.nan, ".2f") == "-"
    assert rd._num(0.5, ".2f") == "0.50"

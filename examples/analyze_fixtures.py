"""Analyse the fixture trajectories shipped with the repository.

Prints one line per run with its findings, then writes the JSON report and
the self-contained HTML report to OUT_DIR (default: trajlens-example).
Run from the repository root:

    uv run python examples/analyze_fixtures.py [OUT_DIR]
"""

import json
import sys
from pathlib import Path

from trajlens.adapters import load
from trajlens.analysis import analyze_trajectories
from trajlens.html import render_html

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = sorted((ROOT / "tests" / "fixtures").glob("*"))
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("trajlens-example")

trajs = [load(p) for p in FIXTURES]
report = analyze_trajectories(trajs, steps=True)
for run in report["runs"]:
    m = run["metrics"]
    labels = ", ".join(f["label"] for f in run["findings"]) or "none"
    print(
        f"{m['run_id']:16s} {run['source_format']:12s} steps={m['steps']:2d} "
        f"edits={m['edits']} tests={m['test_runs']} errors={m['error_steps']} -> {labels}"
    )
    for f in run["findings"]:
        print(f"    {f['label']}: {f['detail']}; evidence steps {f['step_ids']}")

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(out_dir / "report.html").write_text(render_html(report, "trajlens fixtures"), encoding="utf-8")
print(f"wrote {out_dir / 'report.json'} and {out_dir / 'report.html'}")

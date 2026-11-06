"""Analyse the fixture trajectories shipped with the repository and print a summary.

Run from the repository root:

    uv run python examples/analyze_fixtures.py
"""

from pathlib import Path

from trajlens.adapters import load
from trajlens.classify import classify_run
from trajlens.metrics import run_metrics

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = sorted((ROOT / "tests" / "fixtures").glob("*.jsonl"))

for path in FIXTURES:
    traj = load(path)
    m = run_metrics(traj)
    findings = classify_run(traj, m)
    labels = ", ".join(f.label for f in findings) or "none"
    print(
        f"{traj.run_id:14s} steps={m.steps:2d} edits={m.edits} tests={m.test_runs} "
        f"loops={m.looped_steps} errors={m.error_steps}  -> {labels}"
    )
    for f in findings:
        print(f"    {f.label}: {f.detail}; evidence steps {f.step_ids}")

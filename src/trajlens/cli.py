"""Command line entry point.

trajlens analyze RUN.jsonl [RUN2.jsonl ...] [--json OUT] [--format generic]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from trajlens import __version__
from trajlens.adapters import FORMATS, load
from trajlens.classify import FAILURE_CLASSES, classify_run
from trajlens.metrics import INFER_MODES, run_metrics


def analyze_paths(
    paths: list[Path],
    fmt: str = "auto",
    window: int = 6,
    min_repeats: int = 3,
    near_threshold: float = 0.9,
    loop_min_steps: int = 4,
    cascade_min: int = 4,
    blowup_chars: int = 20000,
    growth_ratio: float = 5.0,
    infer_errors: str = "auto",
) -> dict:
    """Analyse every file and return the report as a plain dict."""
    runs = []
    for p in paths:
        traj = load(p, fmt)
        m = run_metrics(traj, window, min_repeats, near_threshold, infer_errors)
        findings = classify_run(traj, m, loop_min_steps, cascade_min, blowup_chars, growth_ratio)
        runs.append(
            {
                "path": str(p),
                "metrics": m.to_dict(),
                "findings": [f.to_dict() for f in findings],
            }
        )
    return {"version": __version__, "runs": runs, "aggregate": aggregate(runs)}


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def aggregate(runs: list[dict]) -> dict:
    """Totals and means across runs."""
    n = len(runs)
    mets = [r["metrics"] for r in runs]
    labels = Counter(f["label"] for r in runs for f in r["findings"])
    runs_with = Counter(lab for r in runs for lab in {f["label"] for f in r["findings"]})
    tool_mix: Counter = Counter()
    for m in mets:
        tool_mix.update(m["tool_mix"])
    return {
        "runs": n,
        "mean_steps": _mean([m["steps"] for m in mets]),
        "mean_edits": _mean([m["edits"] for m in mets]),
        "mean_test_runs": _mean([m["test_runs"] for m in mets]),
        "mean_error_rate": _mean([m["error_rate"] for m in mets]),
        "mean_first_edit_step": _mean(
            [m["first_edit_step"] for m in mets if m["first_edit_step"] is not None]
        ),
        "mean_first_test_step": _mean(
            [m["first_test_step"] for m in mets if m["first_test_step"] is not None]
        ),
        "runs_with_loops": sum(1 for m in mets if m["looped_steps"]),
        "runs_tests_after_last_edit": sum(1 for m in mets if m["tests_after_last_edit"]),
        "runs_patch_touches_tests": sum(1 for m in mets if m["patch_touches_tests"]),
        "runs_empty_patch": sum(1 for m in mets if m["patch_empty"]),
        "tool_mix": dict(tool_mix),
        "findings_total": {lab: labels.get(lab, 0) for lab in FAILURE_CLASSES},
        "runs_with_finding": {lab: runs_with.get(lab, 0) for lab in FAILURE_CLASSES},
    }


def format_text(report: dict) -> str:
    lines = []
    for r in report["runs"]:
        m = r["metrics"]
        lines.append(
            f"{m['run_id']}: {m['steps']} steps, {m['edits']} edits, "
            f"{m['test_runs']} test runs, error rate {m['error_rate']:.2f}, "
            f"first edit {m['first_edit_step']}, first test {m['first_test_step']}, "
            f"tests after last edit {m['tests_after_last_edit']}, "
            f"patch files {len(m['patch_files'])}"
        )
        for f in r["findings"]:
            ids = ",".join(str(i) for i in f["step_ids"]) or "-"
            lines.append(f"  {f['label']}: {f['detail']} [steps {ids}]")
    a = report["aggregate"]
    lines.append(f"runs: {a['runs']}")
    if a["runs"]:
        lines.append(f"mean steps: {a['mean_steps']:.1f}")
        lines.append(f"mean error rate: {a['mean_error_rate']:.3f}")
        found = {k: v for k, v in a["runs_with_finding"].items() if v}
        shown = ", ".join(f"{k}={v}" for k, v in found.items()) or "none"
        lines.append("runs with finding: " + shown)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trajlens", description=__doc__.strip().splitlines()[0])
    p.add_argument("--version", action="version", version=f"trajlens {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="measure and classify trajectory files")
    a.add_argument("paths", nargs="+", type=Path)
    a.add_argument("--format", default="auto", choices=("auto", *FORMATS))
    a.add_argument("--json", type=Path, help="write the full report as JSON here")
    a.add_argument("--window", type=int, default=6, help="loop window in steps")
    a.add_argument("--min-repeats", type=int, default=3, help="repeats that make a loop")
    a.add_argument("--near-threshold", type=float, default=0.9, help="near-duplicate ratio")
    a.add_argument("--loop-min-steps", type=int, default=4, help="loop size that is a failure")
    a.add_argument("--cascade-min", type=int, default=4, help="error streak that is a cascade")
    a.add_argument("--blowup-chars", type=int, default=20000, help="observation size for blow-up")
    a.add_argument("--growth-ratio", type=float, default=5.0, help="observation growth for blow-up")
    a.add_argument(
        "--infer-errors",
        default="auto",
        choices=INFER_MODES,
        help="read failures from observation text when a step has no exit code",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze_paths(
            args.paths,
            args.format,
            args.window,
            args.min_repeats,
            args.near_threshold,
            args.loop_min_steps,
            args.cascade_min,
            args.blowup_chars,
            args.growth_ratio,
            args.infer_errors,
        )
    except (OSError, ValueError) as exc:
        print(f"trajlens: {exc}", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_text(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

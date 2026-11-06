"""Command line entry point.

trajlens analyze RUN.jsonl [RUN2.jsonl ...] [--json OUT] [--format generic]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trajlens import __version__
from trajlens.adapters import FORMATS
from trajlens.analysis import Options, aggregate, analyze_paths
from trajlens.metrics import INFER_MODES

__all__ = ["aggregate", "analyze_paths", "build_parser", "format_text", "main"]


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
        lines.append(f"runs with no finding: {a['runs_no_finding']}")
        for key, block in a.get("by_outcome", {}).items():
            found = {k: v for k, v in block["runs_with_finding"].items() if v}
            shown = ", ".join(f"{k}={v}" for k, v in found.items()) or "none"
            lines.append(f"{key} runs: {block['runs']}; with finding: {shown}")
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
        opts = Options(
            window=args.window,
            min_repeats=args.min_repeats,
            near_threshold=args.near_threshold,
            loop_min_steps=args.loop_min_steps,
            cascade_min=args.cascade_min,
            blowup_chars=args.blowup_chars,
            growth_ratio=args.growth_ratio,
            infer_errors=args.infer_errors,
        )
        report = analyze_paths(args.paths, args.format, opts)
    except (OSError, ValueError) as exc:
        print(f"trajlens: {exc}", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_text(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

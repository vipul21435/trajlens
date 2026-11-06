"""Command line entry point.

trajlens analyze RUN [RUN ...] [--json OUT] [--html OUT] [--format NAME]
trajlens convert RUN [RUN ...] --out DIR [--format NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trajlens import __version__
from trajlens.adapters import FORMATS, load
from trajlens.adapters.generic import dump_generic
from trajlens.analysis import Options, aggregate, analyze_paths
from trajlens.html import render_html
from trajlens.metrics import INFER_MODES

__all__ = ["aggregate", "analyze_paths", "build_parser", "convert", "format_text", "main"]


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
    a.add_argument(
        "--html", type=Path, help="write a self-contained HTML report here (adds step rows)"
    )
    a.add_argument("--title", default="trajlens report", help="title of the HTML report")
    a.add_argument("--window", type=int, default=6, help="loop window in steps")
    a.add_argument("--min-repeats", type=int, default=3, help="repeats that make a loop")
    a.add_argument("--near-threshold", type=float, default=0.9, help="near-duplicate ratio")
    a.add_argument("--loop-min-steps", type=int, default=4, help="loop size that is a failure")
    a.add_argument("--cascade-min", type=int, default=4, help="error streak that is a cascade")
    a.add_argument("--blowup-chars", type=int, default=20000, help="observation size for blow-up")
    a.add_argument(
        "--context-chars", type=int, default=60000, help="total observation size for blow-up"
    )
    a.add_argument(
        "--include-new-tests",
        action="store_true",
        help="count test files the patch creates as edited tests",
    )
    a.add_argument(
        "--no-edit-reset",
        action="store_true",
        help="let repeats of a command span edits (count edit-test cycles as loops)",
    )
    a.add_argument(
        "--infer-errors",
        default="auto",
        choices=INFER_MODES,
        help="read failures from observation text when a step has no exit code",
    )
    c = sub.add_parser("convert", help="rewrite trajectory files as generic JSONL")
    c.add_argument("paths", nargs="+", type=Path)
    c.add_argument("--format", default="auto", choices=("auto", *FORMATS))
    c.add_argument("--out", type=Path, required=True, help="directory for the .jsonl files")
    return p


def _safe_name(run_id: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_id)
    return keep.strip(".") or "run"


def convert(paths: list[Path], fmt: str, out: Path) -> list[Path]:
    """Write each file as ``out/<run_id>.jsonl``; a repeated run id gets a numeric suffix."""
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for p in paths:
        traj = load(p, fmt)
        base = _safe_name(traj.run_id)
        target = out / f"{base}.jsonl"
        n = 1
        while target in written:
            n += 1
            target = out / f"{base}-{n}.jsonl"
        dump_generic(traj, target)
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "convert":
        try:
            written = convert(args.paths, args.format, args.out)
        except (OSError, ValueError) as exc:
            print(f"trajlens: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {len(written)} files to {args.out}")
        return 0
    try:
        opts = Options(
            window=args.window,
            min_repeats=args.min_repeats,
            near_threshold=args.near_threshold,
            loop_min_steps=args.loop_min_steps,
            cascade_min=args.cascade_min,
            blowup_chars=args.blowup_chars,
            context_chars=args.context_chars,
            infer_errors=args.infer_errors,
            edit_resets=not args.no_edit_reset,
            include_new_tests=args.include_new_tests,
        )
        report = analyze_paths(args.paths, args.format, opts, steps=args.html is not None)
    except (OSError, ValueError) as exc:
        print(f"trajlens: {exc}", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.html:
        args.html.write_text(render_html(report, args.title), encoding="utf-8")
    print(format_text(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

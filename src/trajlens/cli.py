"""Command line entry point.

trajlens analyze RUN [RUN ...] [--json OUT] [--html OUT] [--format NAME] [--fail-on CLASSES]
trajlens convert RUN [RUN ...] --out DIR [--format NAME]
trajlens show RUN [--format NAME] [--width N]
trajlens audit RUN [RUN ...] --label CLASS [--n 20] [--seed 0] [--format NAME]
trajlens compare A.json B.json [--names A B]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trajlens import __version__
from trajlens.adapters import FORMATS, load
from trajlens.adapters.generic import dump_generic
from trajlens.analysis import (
    GROUP_KEYS,
    Options,
    aggregate,
    analyze_paths,
    analyze_run,
    write_csv,
)
from trajlens.classify import FAILURE_CLASSES
from trajlens.compare import compare_reports, format_comparison
from trajlens.html import render_html
from trajlens.metrics import INFER_MODES

__all__ = [
    "aggregate",
    "audit",
    "analyze_paths",
    "build_parser",
    "convert",
    "format_run",
    "format_text",
    "main",
]


def format_text(report: dict) -> str:
    """The plain-text summary ``trajlens analyze`` prints: one block per run, then totals."""
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
        if not m["steps"]:
            lines.append(f"  warning: {m['run_id']} has no steps; check the file and --format")
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
        for key, block in a.get("groups", {}).items():
            found = {k: v for k, v in block["runs_with_finding"].items() if v}
            shown = ", ".join(f"{k}={v}" for k, v in found.items()) or "none"
            rate = block["resolve_rate"]
            res = "" if rate is None else f", resolved {rate:.1%}"
            name = key or "(none)"
            lines.append(
                f"{a['group_by']} {name}: {block['runs']} runs{res}; with finding: {shown}"
            )
    return "\n".join(lines)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show "(default: X)" only where there is a default worth reading."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default in (None, False) or action.default is argparse.SUPPRESS:
            return action.help or ""
        return super()._get_help_string(action) or ""


DESCRIPTION = "Read coding-agent trajectories, measure behaviour, classify failures."
FORMAT_HELP = "input format; auto detects it from the file"


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the ``trajlens`` command with all its sub-commands."""
    p = argparse.ArgumentParser(prog="trajlens", description=DESCRIPTION)
    p.add_argument("--version", action="version", version=f"trajlens {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, help: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help, description=help, formatter_class=_HelpFormatter)

    a = add("analyze", "measure and classify trajectory files")
    a.add_argument("paths", nargs="+", type=Path, help="trajectory files")
    a.add_argument("--format", default="auto", choices=("auto", *FORMATS), help=FORMAT_HELP)
    a.add_argument("--json", type=Path, help="write the full report as JSON here")
    a.add_argument(
        "--html", type=Path, help="write a self-contained HTML report here (adds step rows)"
    )
    a.add_argument("--csv", type=Path, help="write one row of metrics and classes per run here")
    a.add_argument("--title", default="trajlens report", help="title of the HTML report")
    a.add_argument(
        "--fail-on",
        metavar="CLASSES",
        help="exit with status 1 when any run carries one of these comma-separated classes "
        "('any' for every class)",
    )
    a.add_argument(
        "--skip-bad",
        action="store_true",
        help="leave out files that cannot be read, list them on stderr, and go on",
    )
    a.add_argument("--group-by", choices=GROUP_KEYS, help="also count findings per value of this")
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
        "--no-script-tests",
        action="store_true",
        help="do not count runs of scripts the agent created as test runs",
    )
    a.add_argument(
        "--loop-any-observation",
        action="store_true",
        help="group repeated commands even when their observations differ",
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
    s = add("show", "print one run step by step with its findings")
    s.add_argument("path", type=Path, help="one trajectory file")
    s.add_argument("--format", default="auto", choices=("auto", *FORMATS), help=FORMAT_HELP)
    s.add_argument("--width", type=int, default=100, help="cut each line to this many characters")
    au = add("audit", "print a seeded random sample of runs carrying one class, for reading")
    au.add_argument("paths", nargs="+", type=Path, help="trajectory files")
    au.add_argument("--label", required=True, choices=FAILURE_CLASSES, help="class to sample")
    au.add_argument("--n", type=int, default=20, help="runs to draw")
    au.add_argument("--seed", type=int, default=0, help="seed of the random draw")
    au.add_argument("--format", default="auto", choices=("auto", *FORMATS), help=FORMAT_HELP)
    au.add_argument("--width", type=int, default=100, help="cut each line to this many characters")
    au.add_argument("--no-script-tests", action="store_true", help="as for analyze")
    cmp_ = add("compare", "compare class shares of two --json reports")
    cmp_.add_argument("a", type=Path, help="first report (from analyze --json)")
    cmp_.add_argument("b", type=Path, help="second report")
    cmp_.add_argument(
        "--names",
        nargs=2,
        default=("A", "B"),
        metavar=("NAME_A", "NAME_B"),
        help="column names for the two reports",
    )
    c = add("convert", "rewrite trajectory files as generic JSONL")
    c.add_argument("paths", nargs="+", type=Path, help="trajectory files")
    c.add_argument("--format", default="auto", choices=("auto", *FORMATS), help=FORMAT_HELP)
    c.add_argument("--out", type=Path, required=True, help="directory for the .jsonl files")
    return p


def format_run(run: dict, width: int = 100) -> str:
    """One run as text: findings first, then one line per step.

    Step lines read ``index tool exit observation-chars command``; a ``!``
    marks a failed step and a ``*`` a step some finding cites as evidence.
    """
    m = run["metrics"]
    cited: set[int] = {i for f in run["findings"] for i in f["step_ids"]}
    lines = [
        f"{m['run_id']} ({run['source_format']}): {m['steps']} steps, "
        f"resolved {run.get('resolved')}, exit status {run.get('exit_status') or '-'}"
    ]
    for f in run["findings"]:
        ids = ",".join(str(i) for i in f["step_ids"]) or "-"
        lines.append(f"  {f['label']}: {f['detail']} [steps {ids}]")
    if not run["findings"]:
        lines.append("  no findings")
    for st in run.get("steps", []):
        first = st["command"].strip().splitlines()[0] if st["command"].strip() else ""
        code = "-" if st["exit_code"] is None else str(st["exit_code"])
        mark = ("!" if st["error"] else " ") + ("*" if st["index"] in cited else " ")
        size = st["observation_chars"]
        line = f"{st['index']:4d}{mark} {st['tool']:6s} {code:>3s} {size:7d}  {first}"
        lines.append(line[:width])
    return "\n".join(lines)


def audit(
    paths: list[Path], fmt: str, label: str, n: int, seed: int, opts: Options | None = None
) -> tuple[int, list[dict]]:
    """Draw up to ``n`` runs carrying ``label`` with ``random.Random(seed).sample``.

    Runs are considered in the order of ``paths``; returns the number of runs
    carrying the label and the drawn runs (with step rows).
    """
    import random

    runs = [analyze_run(load(p, fmt), opts, str(p), steps=True) for p in paths]
    hits = [r for r in runs if any(f["label"] == label for f in r["findings"])]
    drawn = random.Random(seed).sample(hits, min(n, len(hits)))
    return len(hits), drawn


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
    """Run the command line; returns the exit status (0 ok, 1 --fail-on hit, 2 error)."""
    args = build_parser().parse_args(argv)
    if args.cmd == "show":
        try:
            run = analyze_run(load(args.path, args.format), Options(), str(args.path), steps=True)
        except (OSError, ValueError) as exc:
            print(f"trajlens: {exc}", file=sys.stderr)
            return 2
        print(format_run(run, args.width))
        return 0
    if args.cmd == "audit":
        try:
            opts = Options(script_tests=not args.no_script_tests)
            total, drawn = audit(args.paths, args.format, args.label, args.n, args.seed, opts)
        except (OSError, ValueError) as exc:
            print(f"trajlens: {exc}", file=sys.stderr)
            return 2
        print(f"{total} of {len(args.paths)} runs carry {args.label}; showing {len(drawn)}")
        for run in drawn:
            print()
            print(format_run(run, args.width))
        return 0
    if args.cmd == "compare":
        try:
            a = json.loads(args.a.read_text(encoding="utf-8"))
            b = json.loads(args.b.read_text(encoding="utf-8"))
            rows = compare_reports(a, b)
        except (OSError, ValueError, KeyError) as exc:
            print(f"trajlens: {exc}", file=sys.stderr)
            return 2
        print(format_comparison(rows, *args.names))
        return 0
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
            same_observation=not args.loop_any_observation,
            script_tests=not args.no_script_tests,
            include_new_tests=args.include_new_tests,
        )
        report = analyze_paths(
            args.paths,
            args.format,
            opts,
            steps=args.html is not None,
            group_by=args.group_by,
            skip_bad=args.skip_bad,
        )
    except (OSError, ValueError) as exc:
        print(f"trajlens: {exc}", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.html:
        args.html.write_text(render_html(report, args.title), encoding="utf-8")
    if args.csv:
        write_csv(report, args.csv)
    for bad in report.get("skipped", []):
        print(f"trajlens: skipped {bad['error']}", file=sys.stderr)
    print(format_text(report))
    if args.fail_on:
        wanted = set(FAILURE_CLASSES) if args.fail_on == "any" else set(args.fail_on.split(","))
        unknown = wanted - set(FAILURE_CLASSES)
        if unknown:
            print(f"trajlens: unknown class {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        hits = [
            r["metrics"]["run_id"]
            for r in report["runs"]
            if wanted & {f["label"] for f in r["findings"]}
        ]
        if hits:
            print(f"failing: {len(hits)} runs carry {', '.join(sorted(wanted))}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

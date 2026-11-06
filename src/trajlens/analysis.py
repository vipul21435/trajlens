"""Analysis of many runs: per-run metrics and findings, then the aggregate.

:func:`analyze_trajectories` is the library entry point; the command line
calls it through :func:`analyze_paths`. The report is a plain dict that
serialises to JSON as it is.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from trajlens import __version__
from trajlens.adapters import load
from trajlens.classify import FAILURE_CLASSES, classify_run
from trajlens.metrics import run_metrics
from trajlens.model import Trajectory


@dataclass
class Options:
    """Every threshold the metrics and rules use, with their defaults."""

    window: int = 6
    min_repeats: int = 3
    near_threshold: float = 0.9
    loop_min_steps: int = 4
    cascade_min: int = 4
    blowup_chars: int = 20000
    growth_ratio: float = 5.0
    infer_errors: str = "auto"


def analyze_run(traj: Trajectory, opts: Options | None = None, path: str = "") -> dict:
    """Metrics, findings and outcome of one run as a plain dict."""
    o = opts or Options()
    m = run_metrics(traj, o.window, o.min_repeats, o.near_threshold, o.infer_errors)
    findings = classify_run(
        traj, m, o.loop_min_steps, o.cascade_min, o.blowup_chars, o.growth_ratio
    )
    return {
        "path": path,
        "source_format": traj.source_format,
        "task_id": traj.task_id,
        "model": traj.model,
        "resolved": traj.resolved,
        "exit_status": traj.extra.get("exit_status") or "",
        "metrics": m.to_dict(),
        "findings": [f.to_dict() for f in findings],
    }


def analyze_trajectories(trajs: Iterable[Trajectory], opts: Options | None = None) -> dict:
    """Analyse loaded trajectories and return the full report."""
    runs = [analyze_run(t, opts) for t in trajs]
    return {"version": __version__, "runs": runs, "aggregate": aggregate(runs)}


def analyze_paths(paths: list[Path], fmt: str = "auto", opts: Options | None = None) -> dict:
    """Load every file in ``fmt`` (or detect it) and analyse it."""
    runs = [analyze_run(load(p, fmt), opts, str(p)) for p in paths]
    return {"version": __version__, "runs": runs, "aggregate": aggregate(runs)}


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def aggregate(runs: list[dict]) -> dict:
    """Totals and means across runs.

    When at least one run carries a ``resolved`` outcome, ``by_outcome``
    repeats the finding counts separately for resolved and unresolved runs,
    with the share of runs that carry each class.
    """
    n = len(runs)
    mets = [r["metrics"] for r in runs]
    labels = Counter(f["label"] for r in runs for f in r["findings"])
    runs_with = _runs_with(runs)
    tool_mix: Counter = Counter()
    for m in mets:
        tool_mix.update(m["tool_mix"])
    out = {
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
        "runs_submitted": sum(1 for m in mets if m["submitted"]),
        "runs_no_finding": sum(1 for r in runs if not r["findings"]),
        "tool_mix": dict(tool_mix),
        "findings_total": {lab: labels.get(lab, 0) for lab in FAILURE_CLASSES},
        "runs_with_finding": runs_with,
    }
    if any(r.get("resolved") is not None for r in runs):
        out["by_outcome"] = {
            key: _outcome_block([r for r in runs if r.get("resolved") is want])
            for key, want in (("resolved", True), ("unresolved", False))
        }
    return out


def _runs_with(runs: list[dict]) -> dict[str, int]:
    c = Counter(lab for r in runs for lab in {f["label"] for f in r["findings"]})
    return {lab: c.get(lab, 0) for lab in FAILURE_CLASSES}


def _outcome_block(runs: list[dict]) -> dict:
    n = len(runs)
    counts = _runs_with(runs)
    mets = [r["metrics"] for r in runs]
    return {
        "runs": n,
        "mean_steps": _mean([m["steps"] for m in mets]),
        "runs_no_finding": sum(1 for r in runs if not r["findings"]),
        "runs_with_finding": counts,
        "share_with_finding": {k: (v / n if n else None) for k, v in counts.items()},
    }

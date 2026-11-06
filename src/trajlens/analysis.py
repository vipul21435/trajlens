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
from trajlens.metrics import run_metrics, step_errors
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
    context_chars: int = 60000
    include_new_tests: bool = False
    infer_errors: str = "auto"
    edit_resets: bool = True
    same_observation: bool = True


def step_rows(traj: Trajectory, infer_errors: str = "auto", head: int = 160) -> list[dict]:
    """A compact view of every step for the HTML report.

    Commands are cut to 400 characters and observations to their first
    ``head`` characters; the full observation length is kept.
    """
    failed = step_errors(traj.steps, infer_errors)
    return [
        {
            "index": s.index,
            "tool": s.tool,
            "command": s.command[:400],
            "exit_code": s.exit_code,
            "error": f,
            "observation_chars": len(s.observation),
            "observation_head": s.observation[:head],
        }
        for s, f in zip(traj.steps, failed, strict=True)
    ]


def analyze_run(
    traj: Trajectory, opts: Options | None = None, path: str = "", steps: bool = False
) -> dict:
    """Metrics, findings and outcome of one run as a plain dict.

    With ``steps`` the dict also holds :func:`step_rows` under ``steps``.
    """
    o = opts or Options()
    m = run_metrics(
        traj,
        window=o.window,
        min_repeats=o.min_repeats,
        near_threshold=o.near_threshold,
        infer_errors=o.infer_errors,
        edit_resets=o.edit_resets,
        same_observation=o.same_observation,
    )
    findings = classify_run(
        traj,
        m,
        o.loop_min_steps,
        o.cascade_min,
        o.blowup_chars,
        o.context_chars,
        o.include_new_tests,
    )
    out = {
        "path": path,
        "source_format": traj.source_format,
        "task_id": traj.task_id,
        "model": traj.model,
        "resolved": traj.resolved,
        "exit_status": traj.extra.get("exit_status") or "",
        "metrics": m.to_dict(),
        "findings": [f.to_dict() for f in findings],
    }
    if steps:
        out["steps"] = step_rows(traj, o.infer_errors)
    return out


def analyze_trajectories(
    trajs: Iterable[Trajectory],
    opts: Options | None = None,
    steps: bool = False,
    group_by: str | None = None,
) -> dict:
    """Analyse loaded trajectories and return the full report."""
    runs = [analyze_run(t, opts, steps=steps) for t in trajs]
    return {"version": __version__, "runs": runs, "aggregate": aggregate(runs, group_by)}


def analyze_paths(
    paths: list[Path],
    fmt: str = "auto",
    opts: Options | None = None,
    steps: bool = False,
    group_by: str | None = None,
) -> dict:
    """Load every file in ``fmt`` (or detect it) and analyse it."""
    runs = [analyze_run(load(p, fmt), opts, str(p), steps) for p in paths]
    return {"version": __version__, "runs": runs, "aggregate": aggregate(runs, group_by)}


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


GROUP_KEYS = ("model", "source_format", "exit_status", "task_id")


def aggregate(runs: list[dict], group_by: str | None = None) -> dict:
    """Totals and means across runs.

    When at least one run carries a ``resolved`` outcome, ``by_outcome``
    repeats the finding counts separately for resolved and unresolved runs,
    with the share of runs that carry each class. With ``group_by`` (one of
    :data:`GROUP_KEYS`) ``groups`` holds the same block for every value of
    that run field, largest group first.
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
    if group_by is not None:
        if group_by not in GROUP_KEYS:
            raise ValueError(f"group_by must be one of {GROUP_KEYS}, got {group_by!r}")
        values = Counter(str(r.get(group_by) or "") for r in runs)
        out["group_by"] = group_by
        out["groups"] = {
            v: _outcome_block([r for r in runs if str(r.get(group_by) or "") == v])
            for v, _ in values.most_common()
        }
    return out


CSV_FIELDS = (
    "run_id", "path", "source_format", "task_id", "model", "resolved", "exit_status", "steps",
    "edits", "failed_edits", "test_runs", "failed_test_runs", "first_edit_step",
    "first_test_step", "tests_after_last_edit", "looped_steps", "error_steps", "error_rate",
    "errors_inferred", "max_error_streak", "observation_chars", "observation_growth",
    "largest_observation_chars",
    "submitted", "patch_empty", "duration_seconds", "tokens_in", "tokens_out",
)  # fmt: skip


def write_csv(report: dict, path: str | Path) -> None:
    """One row per run: the scalar metrics, then one 0/1 column per failure class."""
    import csv

    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([*CSV_FIELDS, *FAILURE_CLASSES])
        for r in report["runs"]:
            m = r["metrics"]
            labels = {f["label"] for f in r["findings"]}
            row = [m[k] if k in m else r.get(k) for k in CSV_FIELDS]
            row = ["" if v is None else v for v in row]
            w.writerow([*row, *(int(c in labels) for c in FAILURE_CLASSES)])


def _runs_with(runs: list[dict]) -> dict[str, int]:
    c = Counter(lab for r in runs for lab in {f["label"] for f in r["findings"]})
    return {lab: c.get(lab, 0) for lab in FAILURE_CLASSES}


def _outcome_block(runs: list[dict]) -> dict:
    n = len(runs)
    counts = _runs_with(runs)
    mets = [r["metrics"] for r in runs]
    resolved = [r["resolved"] for r in runs if r.get("resolved") is not None]
    return {
        "runs": n,
        "mean_steps": _mean([m["steps"] for m in mets]),
        "resolve_rate": (sum(resolved) / len(resolved)) if resolved else None,
        "runs_no_finding": sum(1 for r in runs if not r["findings"]),
        "runs_with_finding": counts,
        "share_with_finding": {k: (v / n if n else None) for k, v in counts.items()},
    }

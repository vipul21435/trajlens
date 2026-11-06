"""Measure trajlens on the downloaded nebius/SWE-agent-trajectories sample.

    uv run python scripts/real_data_report.py ../_data/trajlens/nebius \
        --chart docs/images/classes_by_outcome.png

Prints Markdown tables: aggregate behaviour metrics, the share of resolved
and unresolved runs carrying each failure class (with 95 % Wilson
intervals), and how well two rules agree with the agent framework's own
exit status. The sample is split by row offset: the 100 lowest offsets are
the rows the defaults were looked at on ("tuning"), the rest are held out.
The chart needs matplotlib (``uv run --with matplotlib ...``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from trajlens.adapters.chat import load_chat
from trajlens.analysis import Options, aggregate, analyze_run
from trajlens.classify import FAILURE_CLASSES


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for k successes in n trials."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return (max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom))


def share(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]" if n else "-"


def agreement(runs: list[dict], label: str, truth) -> dict:
    tp = fp = fn = tn = 0
    for r in runs:
        pred = any(f["label"] == label for f in r["findings"])
        real = truth(r)
        tp += pred and real
        fp += pred and not real
        fn += real and not pred
        tn += not pred and not real
    prec = tp / (tp + fp) if tp + fp else math.nan
    rec = tp / (tp + fn) if tp + fn else math.nan
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec}


def _num(value, spec: str = ".1f") -> str:
    return "-" if value is None else format(value, spec)


def metrics_table(a: dict, runs: list[dict]) -> list[str]:
    n = len(runs) or 1
    failed_edits = sum(r["metrics"]["failed_edits"] for r in runs) / n
    rows = [
        ("runs", a["runs"]),
        ("mean steps", _num(a["mean_steps"])),
        ("mean edits", _num(a["mean_edits"])),
        ("mean failed edits (inferred)", _num(failed_edits)),
        ("mean test runs", _num(a["mean_test_runs"])),
        ("mean error rate (inferred)", _num(a["mean_error_rate"], ".3f")),
        ("mean first edit step (runs with an edit)", _num(a["mean_first_edit_step"])),
        ("mean first test step (runs with a test)", _num(a["mean_first_test_step"])),
        ("runs with a loop", a["runs_with_loops"]),
        ("runs testing after the last edit", a["runs_tests_after_last_edit"]),
        ("runs submitted", a["runs_submitted"]),
        ("runs with an empty patch", a["runs_empty_patch"]),
        ("runs with no finding", a["runs_no_finding"]),
    ]
    total = sum(a["tool_mix"].values())
    mix = ", ".join(f"{k} {v / total:.1%}" for k, v in a["tool_mix"].items() if v)
    rows.append(("tool mix (share of steps)", mix))
    return ["| metric | value |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in rows]


def class_table(runs: list[dict]) -> list[str]:
    res = [r for r in runs if r["resolved"] is True]
    unres = [r for r in runs if r["resolved"] is False]
    lines = [
        f"| class | resolved runs (n={len(res)}) | unresolved runs (n={len(unres)}) |",
        "| --- | --- | --- |",
    ]
    for lab in FAILURE_CLASSES:
        k1 = sum(1 for r in res if any(f["label"] == lab for f in r["findings"]))
        k2 = sum(1 for r in unres if any(f["label"] == lab for f in r["findings"]))
        lines.append(f"| {lab} | {share(k1, len(res))} | {share(k2, len(unres))} |")
    k1 = sum(1 for r in res if not r["findings"])
    k2 = sum(1 for r in unres if not r["findings"])
    lines.append(f"| (no finding) | {share(k1, len(res))} | {share(k2, len(unres))} |")
    return lines


def chart(runs: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("resolved", True, "#4a6fa5"), ("unresolved", False, "#c46b3b")]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
    width = 0.38
    xs = range(len(FAILURE_CLASSES))
    for j, (name, want, colour) in enumerate(groups):
        sel = [r for r in runs if r["resolved"] is want]
        n = len(sel)
        vals, los, his = [], [], []
        for lab in FAILURE_CLASSES:
            k = sum(1 for r in sel if any(f["label"] == lab for f in r["findings"]))
            lo, hi = wilson(k, n)
            vals.append(k / n)
            los.append(k / n - lo)
            his.append(hi - k / n)
        pos = [x + (j - 0.5) * width for x in xs]
        ax.bar(pos, vals, width, yerr=[los, his], capsize=2, color=colour, label=f"{name} (n={n})")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(FAILURE_CLASSES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("share of runs with the class")
    ax.set_title("trajlens failure classes by outcome, nebius/SWE-agent-trajectories sample")
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path, help="directory written by fetch_nebius_sample.py")
    ap.add_argument("--tuning", type=int, default=100, help="lowest offsets kept for tuning")
    ap.add_argument("--chart", type=Path)
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args()

    files = sorted(args.data.glob("row*.json"))
    opts = Options()
    runs = [analyze_run(load_chat(p), opts, p.name) for p in files]
    tuning, held = runs[: args.tuning], runs[args.tuning :]
    manifest = json.loads((args.data / "MANIFEST.json").read_text())

    print(
        f"Sample: {len(runs)} rows of {manifest['rows_total']} ({manifest['dataset']}, "
        f"seed {manifest['seed']}); tuning rows {len(tuning)}, held-out rows {len(held)}.\n"
    )
    print("All rows, default options:\n")
    print("\n".join(metrics_table(aggregate(runs), runs)) + "\n")
    print("Failure classes by outcome, all rows (95 % Wilson intervals):\n")
    print("\n".join(class_table(runs)) + "\n")
    print("Failure classes by outcome, held-out rows only:\n")
    print("\n".join(class_table(held)) + "\n")

    def ctx(r):
        return "exit_context" in r["exit_status"]

    def no_submit(r):
        return not r["exit_status"].startswith("submitted")

    print("Agreement with the framework's exit status:\n")
    print("| rule vs status | rows | tp | fp | fn | tn | precision | recall |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, part in (("tuning", tuning), ("held-out", held)):
        for label, truth, tname in (
            ("context_blow_up", ctx, "exit_context"),
            ("gave_up", no_submit, "not submitted"),
        ):
            g = agreement(part, label, truth)
            print(
                f"| {label} vs {tname} | {name} | {g['tp']} | {g['fp']} | {g['fn']} | "
                f"{g['tn']} | {g['precision']:.2f} | {g['recall']:.2f} |"
            )
    if args.chart:
        chart(runs, args.chart)
        print(f"\nchart: {args.chart}")
    if args.json:
        rep = {"runs": runs, "aggregate": aggregate(runs)}
        args.json.write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

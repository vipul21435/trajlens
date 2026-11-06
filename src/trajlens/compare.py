"""Compare the failure-class shares of two reports.

Each report is the JSON written by ``trajlens analyze --json``. For every
class the share of runs carrying it is compared between the two, with the
difference B - A and a 95 % interval for it.

Intervals: the share of one report gets the Wilson score interval; the
difference of two independent shares gets Newcombe's hybrid score interval
(Newcombe, "Interval estimation for the difference between independent
proportions: comparison of eleven methods", Statistics in Medicine 17(8),
873-890, 1998), which combines the two Wilson intervals:

    d = pB - pA
    lower = d - sqrt((pB - lB)^2 + (uA - pA)^2)
    upper = d + sqrt((uB - pB)^2 + (pA - lA)^2)

Runs in the two reports are treated as independent samples, which is wrong
for paired designs (the same tasks run by two agents); for those, a paired
method is needed and is not implemented here.
"""

from __future__ import annotations

import math

from trajlens.classify import FAILURE_CLASSES

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def newcombe_diff(ka: int, na: int, kb: int, nb: int, z: float = Z95) -> tuple[float, float, float]:
    """Difference pB - pA with Newcombe's hybrid score interval: (d, lower, upper)."""
    pa, pb = ka / na, kb / nb
    la, ua = wilson(ka, na, z)
    lb, ub = wilson(kb, nb, z)
    d = pb - pa
    lower = d - math.sqrt((pb - lb) ** 2 + (ua - pa) ** 2)
    upper = d + math.sqrt((ub - pb) ** 2 + (pa - la) ** 2)
    return d, max(-1.0, lower), min(1.0, upper)


def _counts(report: object) -> tuple[int, dict[str, int]]:
    runs = report.get("runs") if isinstance(report, dict) else None
    ok = isinstance(runs, list) and all(
        isinstance(r, dict) and isinstance(r.get("findings"), list) for r in runs
    )
    if not ok:
        raise ValueError("not a trajlens report: expected 'runs', a list of runs with 'findings'")
    counts = {lab: 0 for lab in FAILURE_CLASSES}
    for r in runs:
        for lab in {f.get("label") for f in r["findings"] if isinstance(f, dict)}:
            if lab in counts:
                counts[lab] += 1
    return len(runs), counts


def compare_reports(a: object, b: object) -> list[dict]:
    """One row per class: counts, shares, and B - A with its interval."""
    na, ca = _counts(a)
    nb, cb = _counts(b)
    if na == 0 or nb == 0:
        raise ValueError("both reports need at least one run")
    rows = []
    for lab in FAILURE_CLASSES:
        d, lo, hi = newcombe_diff(ca[lab], na, cb[lab], nb)
        rows.append(
            {
                "label": lab,
                "a": ca[lab],
                "n_a": na,
                "b": cb[lab],
                "n_b": nb,
                "diff": d,
                "lower": lo,
                "upper": hi,
                "excludes_zero": lo > 0 or hi < 0,
            }
        )
    return rows


def format_comparison(rows: list[dict], name_a: str = "A", name_b: str = "B") -> str:
    """The rows of :func:`compare_reports` as a fixed-width text table."""
    head = f"{'class':20s} {name_a:>16s} {name_b:>16s}   {name_b} - {name_a} [95% interval]"
    lines = [head]
    for r in rows:
        sa = f"{r['a']}/{r['n_a']} {r['a'] / r['n_a']:6.1%}"
        sb = f"{r['b']}/{r['n_b']} {r['b'] / r['n_b']:6.1%}"
        mark = " *" if r["excludes_zero"] else ""
        lines.append(
            f"{r['label']:20s} {sa:>16s} {sb:>16s}   {r['diff']:+7.1%} "
            f"[{r['lower']:+7.1%}, {r['upper']:+7.1%}]{mark}"
        )
    lines.append("* interval excludes zero")
    return "\n".join(lines)

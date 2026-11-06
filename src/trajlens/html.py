"""One self-contained HTML report.

The page has no external references: no server, no CDN, no web fonts. The
styles are inline, the charts are plain HTML bars, and the only script is a
few lines that filter the run table by failure class; the page reads fully
with scripting off. The report dict is embedded as JSON so the page carries
its own data.
"""

from __future__ import annotations

import json
from html import escape

from trajlens.classify import FAILURE_CLASSES, RULES
from trajlens.model import TOOL_CLASSES

_CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1180px;padding:0 16px;
color:#1d2433;background:#fff}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;
border-bottom:1px solid #d9dde5;padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin:6px 0}
th,td{border-bottom:1px solid #e6e9ef;padding:4px 6px;text-align:left;vertical-align:top}
th{background:#f4f6f9;font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.bar{display:inline-block;height:10px;background:#4a6fa5;vertical-align:middle}
.bar.alt{background:#c46b3b}
.chip{display:inline-block;border-radius:3px;padding:0 5px;margin:1px;font-size:12px;
background:#eef1f6;border:1px solid #cfd6e2}
.chip a{color:inherit;text-decoration:none}
.muted{color:#5b6475}.err{color:#a3261b}
tr.ev td{background:#fff4d6}
code,pre{font:12px/1.35 ui-monospace,Menlo,Consolas,monospace}
pre{white-space:pre-wrap;word-break:break-word;margin:0}
details{margin:6px 0;border:1px solid #e6e9ef;border-radius:4px;padding:4px 8px}
summary{cursor:pointer;font-weight:600}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:2px 16px;
margin:6px 0}
"""

_JS = """
document.getElementById('flt').addEventListener('change',function(e){
var v=e.target.value;document.querySelectorAll('#runs tbody tr').forEach(function(r){
r.style.display=(!v||r.dataset.labels.split(' ').indexOf(v)>=0)?'':'none';});});
"""


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def _bar(share: float | None, cls: str = "") -> str:
    width = 0 if share is None else max(0.0, min(1.0, share)) * 160
    return f'<span class="bar {cls}" style="width:{width:.0f}px"></span>'


def _summary(a: dict) -> str:
    rows = [
        ("runs", a["runs"]),
        ("mean steps", a["mean_steps"]),
        ("mean edits", a["mean_edits"]),
        ("mean test runs", a["mean_test_runs"]),
        ("mean error rate", a["mean_error_rate"]),
        ("mean first edit step", a["mean_first_edit_step"]),
        ("mean first test step", a["mean_first_test_step"]),
        ("runs with loops", a["runs_with_loops"]),
        ("runs testing after last edit", a["runs_tests_after_last_edit"]),
        ("runs whose patch touches tests", a["runs_patch_touches_tests"]),
        ("runs with empty patch", a["runs_empty_patch"]),
        ("runs submitted", a.get("runs_submitted")),
        ("runs with no finding", a.get("runs_no_finding")),
    ]
    cells = "".join(f"<div>{escape(k)}: <b>{escape(_fmt(v))}</b></div>" for k, v in rows)
    return f'<div class="kv">{cells}</div>'


def _tool_mix(a: dict) -> str:
    mix = a.get("tool_mix", {})
    total = sum(mix.values()) or 1
    rows = "".join(
        f"<tr><td>{t}</td><td class=n>{mix.get(t, 0)}</td>"
        f"<td class=n>{mix.get(t, 0) / total:.1%}</td><td>{_bar(mix.get(t, 0) / total)}</td></tr>"
        for t in TOOL_CLASSES
    )
    return (
        "<table><thead><tr><th>tool class</th><th class=n>steps</th>"
        f"<th class=n>share</th><th></th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _classes(a: dict) -> str:
    n = a["runs"] or 1
    by = a.get("by_outcome")
    head = "<th>class</th><th class=n>runs</th><th class=n>share</th><th></th>"
    if by:
        head += (
            f"<th class=n>resolved ({by['resolved']['runs']})</th>"
            f"<th class=n>unresolved ({by['unresolved']['runs']})</th><th></th>"
        )
    head += "<th>rule</th>"
    rows = []
    for lab in FAILURE_CLASSES:
        k = a["runs_with_finding"].get(lab, 0)
        row = f"<td><b>{lab}</b></td><td class=n>{k}</td><td class=n>{k / n:.1%}</td>"
        row += f"<td>{_bar(k / n)}</td>"
        if by:
            rs = by["resolved"]["share_with_finding"].get(lab)
            us = by["unresolved"]["share_with_finding"].get(lab)
            row += f"<td class=n>{_fmt(rs, 3)}</td><td class=n>{_fmt(us, 3)}</td>"
            row += f"<td>{_bar(rs)}<br>{_bar(us, 'alt')}</td>"
        row += f"<td class=muted>{escape(RULES[lab])}</td>"
        rows.append(f"<tr>{row}</tr>")
    note = ""
    if by:
        note = (
            "<p class=muted>Resolved and unresolved columns give the share of runs in that "
            "outcome carrying the class (blue: resolved, orange: unresolved).</p>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>{note}"


def _groups(a: dict) -> str:
    groups = a.get("groups")
    if not groups:
        return ""
    head = "".join(f"<th class=n>{lab}</th>" for lab in FAILURE_CLASSES)
    rows = []
    for name, block in groups.items():
        shares = block["share_with_finding"]
        cells = "".join(f"<td class=n>{_fmt(shares[lab], 3)}</td>" for lab in FAILURE_CLASSES)
        rows.append(
            f"<tr><td>{escape(name or '(none)')}</td><td class=n>{block['runs']}</td>"
            f"<td class=n>{_fmt(block['resolve_rate'], 3)}</td>{cells}</tr>"
        )
    return (
        f"<h2>By {escape(a['group_by'])}</h2><p class=muted>Share of the group's runs carrying "
        "each class.</p><table><thead><tr><th>value</th><th class=n>runs</th>"
        f"<th class=n>resolved</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _run_table(runs: list[dict]) -> str:
    rows = []
    for i, r in enumerate(runs):
        m = r["metrics"]
        labels = sorted({f["label"] for f in r["findings"]})
        chips = "".join(
            f'<span class="chip"><a href="#run{i}">{escape(lab)}</a></span>' for lab in labels
        )
        rows.append(
            f'<tr data-labels="{" ".join(labels)}"><td><a href="#run{i}">'
            f"{escape(str(m['run_id']))}</a></td><td>{escape(r.get('source_format', ''))}</td>"
            f"<td class=n>{m['steps']}</td><td class=n>{m['edits']}</td>"
            f"<td class=n>{m['test_runs']}</td><td class=n>{m['error_rate']:.2f}</td>"
            f"<td>{_fmt(r.get('resolved'))}</td><td>{escape(r.get('exit_status') or '')}</td>"
            f"<td>{chips or '<span class=muted>none</span>'}</td></tr>"
        )
    options = "".join(f'<option value="{c}">{c}</option>' for c in FAILURE_CLASSES)
    return (
        '<p><label>Show runs with class <select id="flt"><option value="">any</option>'
        f"{options}</select></label></p>"
        '<table id="runs"><thead><tr><th>run</th><th>format</th><th class=n>steps</th>'
        "<th class=n>edits</th><th class=n>tests</th><th class=n>error rate</th>"
        "<th>resolved</th><th>exit status</th><th>findings</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _run_detail(i: int, r: dict) -> str:
    m = r["metrics"]
    evidence: dict[int, list[str]] = {}
    items = []
    for f in r["findings"]:
        links = ", ".join(f'<a href="#run{i}s{s}">{s}</a>' for s in f["step_ids"]) or "-"
        items.append(f"<li><b>{escape(f['label'])}</b>: {escape(f['detail'])} (steps {links})</li>")
        for s in f["step_ids"]:
            evidence.setdefault(s, []).append(f["label"])
    keys = [
        "steps", "edits", "failed_edits", "test_runs", "failed_test_runs", "first_edit_step",
        "first_test_step", "last_edit_step",
        "last_test_step", "tests_after_last_edit", "looped_steps", "error_steps",
        "errors_inferred", "max_error_streak", "observation_chars", "observation_growth",
        "largest_observation_chars", "submitted", "patch_empty", "duration_seconds",
        "tokens_in", "tokens_out",
    ]  # fmt: skip
    kv = "".join(f"<div>{k}: <b>{escape(_fmt(m.get(k)))}</b></div>" for k in keys)
    files = ", ".join(escape(p) for p in m["patch_files"]) or "none"
    body = [
        f'<details id="run{i}"><summary>{escape(str(m["run_id"]))}'
        f" <span class=muted>({m['steps']} steps, {len(r['findings'])} findings)</span>"
        "</summary>",
        f"<div class=kv>{kv}</div><p>patch files: {files}</p>",
        f"<ul>{''.join(items)}</ul>" if items else "<p class=muted>no findings</p>",
    ]
    steps = r.get("steps")
    if steps:
        rows = []
        for s in steps:
            ev = evidence.get(s["index"])
            cls = ' class="ev"' if ev else ""
            err = ' class="err"' if s.get("error") else ""
            rows.append(
                f'<tr id="run{i}s{s["index"]}"{cls}><td class=n>{s["index"]}</td>'
                f"<td>{escape(s['tool'])}</td><td><pre>{escape(s['command'])}</pre></td>"
                f"<td{err}>{_fmt(s.get('exit_code'))}{' error' if s.get('error') else ''}</td>"
                f"<td class=n>{s['observation_chars']}</td>"
                f"<td><pre>{escape(s['observation_head'])}</pre></td>"
                f"<td>{escape(', '.join(ev or []))}</td></tr>"
            )
        body.append(
            "<table><thead><tr><th class=n>step</th><th>tool</th><th>command</th>"
            "<th>exit</th><th class=n>obs chars</th><th>observation (start)</th>"
            f"<th>evidence for</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    body.append("</details>")
    return "".join(body)


def render_html(report: dict, title: str = "trajlens report") -> str:
    """Render a report dict from :mod:`trajlens.analysis` as one HTML page."""
    a = report["aggregate"]
    runs = report["runs"]
    # Every "<" becomes \u003c, which is still the same JSON string, so no
    # "</script" or "<!--" can end or re-enter the element early.
    data = json.dumps(report).replace("<", "\\u003c")
    details = "".join(_run_detail(i, r) for i, r in enumerate(runs))
    return (
        "<!DOCTYPE html><html lang=en><head><meta charset=utf-8>"
        f"<title>{escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{escape(title)}</h1><p class=muted>trajlens {escape(report['version'])}, "
        f"{a['runs']} runs</p>"
        f"<h2>Summary</h2>{_summary(a)}"
        f"<h2>Tool mix</h2>{_tool_mix(a)}"
        f"<h2>Failure classes</h2>{_classes(a)}{_groups(a)}"
        f"<h2>Runs</h2>{_run_table(runs)}"
        f"<h2>Run details</h2>{details}"
        f'<script type="application/json" id="report">{data}</script>'
        f"<script>{_JS}</script></body></html>\n"
    )

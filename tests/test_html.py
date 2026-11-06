import json
import re
from html.parser import HTMLParser
from pathlib import Path

from trajlens.analysis import analyze_paths, analyze_trajectories
from trajlens.classify import FAILURE_CLASSES
from trajlens.cli import main
from trajlens.html import render_html
from trajlens.model import Step, Trajectory

FX = Path(__file__).parent / "fixtures"


class _Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.hrefs = []
        self.srcs = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if "id" in d:
            self.ids.add(d["id"])
        if "href" in d:
            self.hrefs.append(d["href"])
        if "src" in d:
            self.srcs.append(d["src"])


def _parse(page):
    c = _Collect()
    c.feed(page)
    return c


def test_page_is_self_contained_and_links_resolve():
    rep = analyze_paths(sorted(FX.glob("*.jsonl")), steps=True)
    page = render_html(rep)
    c = _parse(page)
    assert c.srcs == []
    assert not re.search(r"https?://", page.split('id="report">')[0])
    for href in c.hrefs:
        assert href.startswith("#") and href[1:] in c.ids, href
    for lab in FAILURE_CLASSES:
        assert lab in page


def test_evidence_steps_are_marked():
    rep = analyze_paths([FX / "loop.jsonl"], steps=True)
    page = render_html(rep, "loops")
    assert "<title>loops</title>" in page
    for sid in rep["runs"][0]["findings"][0]["step_ids"]:
        assert f'<tr id="run0s{sid}" class="ev">' in page


def test_embedded_json_round_trips_and_escapes():
    nasty = Step(index=0, tool="shell", command="echo '</script><b>x</b>'", observation="<i>")
    rep = analyze_trajectories([Trajectory("r<1>", [nasty])], steps=True)
    page = render_html(rep)
    assert "<b>x</b>" not in page and "r&lt;1&gt;" in page
    blob = page.split('<script type="application/json" id="report">')[1].split("</script>")[0]
    assert json.loads(blob.replace("<\\/", "</")) == rep


def test_outcome_columns_and_cli(tmp_path):
    ok = Trajectory("ok", [Step(index=0, tool="test", command="pytest")], resolved=True)
    bad = Trajectory("bad", [Step(index=0, tool="edit", command="edit a")], resolved=False)
    page = render_html(analyze_trajectories([ok, bad], group_by="task_id"))
    assert "resolved (1)" in page and "unresolved (1)" in page
    assert "<h2>By task_id</h2>" in page and "(none)" in page
    out = tmp_path / "r.html"
    assert main(["analyze", str(FX / "good.jsonl"), "--html", str(out), "--title", "t"]) == 0
    assert out.read_text().startswith("<!DOCTYPE html>")

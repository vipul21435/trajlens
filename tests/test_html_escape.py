"""The embedded report JSON cannot break out of its script element."""

import json

import html5lib

from trajlens.analysis import Options, analyze_trajectories
from trajlens.html import render_html
from trajlens.model import Step, Trajectory


def test_comment_then_script_in_observation():
    obs = '<!--[if lt IE 9]><script src="html5shiv.js"></script>'
    traj = Trajectory("r", [Step(0, "shell", "curl page", observation=obs, exit_code=0)])
    page = render_html(analyze_trajectories([traj], Options(), steps=True))
    doc = html5lib.parse(page, namespaceHTMLElements=False)
    scripts = doc.findall(".//script")
    assert len(scripts) == 2
    data = json.loads(scripts[0].text)
    assert data["runs"][0]["steps"][0]["observation_head"].startswith("<!--")
    assert "addEventListener" in scripts[1].text

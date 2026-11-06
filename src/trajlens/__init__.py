"""trajlens: read coding-agent trajectories, measure behaviour, classify failures.

Typical use::

    from trajlens import Options, analyze_trajectories, load

    report = analyze_trajectories([load("run.traj")], Options(), steps=True)
    print(report["aggregate"]["runs_with_finding"])
"""

__version__ = "0.1.0"

from trajlens.adapters import load  # noqa: E402
from trajlens.analysis import Options, analyze_trajectories  # noqa: E402
from trajlens.classify import Finding, classify_run  # noqa: E402
from trajlens.metrics import RunMetrics, run_metrics  # noqa: E402
from trajlens.model import Step, Trajectory  # noqa: E402

__all__ = [
    "Finding",
    "Options",
    "RunMetrics",
    "Step",
    "Trajectory",
    "__version__",
    "analyze_trajectories",
    "classify_run",
    "load",
    "run_metrics",
]

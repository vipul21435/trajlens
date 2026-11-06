"""trajlens: read coding-agent trajectories, measure behaviour, classify failures."""

from trajlens.classify import classify_run
from trajlens.metrics import run_metrics
from trajlens.model import Step, Trajectory

__version__ = "0.1.0"
__all__ = ["Step", "Trajectory", "classify_run", "run_metrics", "__version__"]

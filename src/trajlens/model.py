"""One step model shared by every adapter.

A trajectory is a run of an agent on one task. It is a list of steps in order.
Each step holds what the agent thought, the tool or action it chose, the
command it issued, what came back, and whatever bookkeeping the source format
carried (exit code, timestamps, token counts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tool classes used by the metrics. Adapters map their own action names onto
# these; anything they cannot place becomes "other".
EDIT = "edit"
TEST = "test"
READ = "read"
SEARCH = "search"
SHELL = "shell"
SUBMIT = "submit"
OTHER = "other"

TOOL_CLASSES = (EDIT, TEST, READ, SEARCH, SHELL, SUBMIT, OTHER)


@dataclass
class Step:
    """One agent turn: an action and its observation."""

    index: int
    tool: str
    command: str = ""
    thought: str = ""
    observation: str = ""
    exit_code: int | None = None
    timestamp: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    raw_action: str = ""

    @property
    def is_error(self) -> bool:
        """True when the step ended in a non-zero exit code."""
        return self.exit_code is not None and self.exit_code != 0


@dataclass
class Trajectory:
    """A whole run: its steps, the final patch and the source it came from."""

    run_id: str
    steps: list[Step] = field(default_factory=list)
    patch: str = ""
    source_format: str = "generic"
    task_id: str = ""
    model: str = ""
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

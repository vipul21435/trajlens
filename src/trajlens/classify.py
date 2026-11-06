"""Failure classes by transparent rules.

Each rule reads the RunMetrics of one run and returns a Finding that names
the step ids it rests on. A run can carry several classes at once. The
thresholds are parameters of :func:`classify_run` with plain defaults.

Rules (defaults in brackets):

looping             a loop of at least ``loop_min_steps`` [4] steps, exact or
                    near; evidence: the loop's step ids
never_tested        the run has at least one edit and no test run at all;
                    evidence: the edit step ids
untested_final_edit the run has test runs but none after the last edit;
                    evidence: the last edit step and the last test step
gave_up             no submit step and the final patch is empty, or the last
                    step is an error; evidence: the last step
tool_error_cascade  an error streak of at least ``cascade_min`` [4] steps;
                    evidence: the streak's step ids
context_blow_up     a single observation of at least ``blowup_chars`` [20000]
                    characters, or observation growth of at least
                    ``growth_ratio`` [5.0] when the largest observation has at
                    least ``growth_min_chars`` [2000] characters; evidence: the
                    largest observation step
edited_tests        the final patch touches a test file; evidence: none (patch
                    level), the file names are listed in the detail
empty_submission    the run submitted and the patch is empty; evidence: the
                    submit step
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from trajlens.metrics import RunMetrics
from trajlens.model import SUBMIT, Trajectory

FAILURE_CLASSES = (
    "looping",
    "never_tested",
    "untested_final_edit",
    "gave_up",
    "tool_error_cascade",
    "context_blow_up",
    "edited_tests",
    "empty_submission",
)


@dataclass
class Finding:
    label: str
    detail: str
    step_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def classify_run(
    traj: Trajectory,
    m: RunMetrics,
    loop_min_steps: int = 4,
    cascade_min: int = 4,
    blowup_chars: int = 20000,
    growth_ratio: float = 5.0,
    growth_min_chars: int = 2000,
) -> list[Finding]:
    """Apply every rule to one run and return the findings in rule order."""
    out: list[Finding] = []
    steps = traj.steps

    for lp in m.exact_loops + m.near_loops:
        if len(lp.step_ids) >= loop_min_steps:
            out.append(
                Finding(
                    "looping",
                    f"{lp.kind} repeat of {lp.command!r} across {len(lp.step_ids)} steps",
                    list(lp.step_ids),
                )
            )

    edit_ids = [s.index for s in steps if s.tool == "edit"]
    if m.edits and m.test_runs == 0:
        out.append(Finding("never_tested", f"{m.edits} edits and no test run", edit_ids))
    elif m.edits and m.tests_after_last_edit is False:
        ids = [m.last_edit_step]
        if m.last_test_step is not None:
            ids.append(m.last_test_step)
        out.append(
            Finding(
                "untested_final_edit",
                f"last edit at step {m.last_edit_step}, last test run at step {m.last_test_step}",
                [i for i in ids if i is not None],
            )
        )

    if steps:
        last = steps[-1]
        if not m.submitted and (m.patch_empty or last.is_error):
            why = "empty patch" if m.patch_empty else f"last step exit code {last.exit_code}"
            out.append(Finding("gave_up", f"no submit step and {why}", [last.index]))

    if m.max_error_streak >= cascade_min:
        out.append(
            Finding(
                "tool_error_cascade",
                f"{m.max_error_streak} consecutive error steps",
                list(m.error_streak_steps),
            )
        )

    big = m.largest_observation_chars >= blowup_chars
    grown = (
        m.observation_growth is not None
        and m.observation_growth >= growth_ratio
        and m.largest_observation_chars >= growth_min_chars
    )
    if big or grown:
        parts = []
        if big:
            parts.append(f"largest observation {m.largest_observation_chars} chars")
        if grown:
            parts.append(f"observation growth x{m.observation_growth:.1f}")
        ids = [m.largest_observation_step] if m.largest_observation_step is not None else []
        out.append(Finding("context_blow_up", ", ".join(parts), ids))

    if m.patch_touches_tests:
        out.append(Finding("edited_tests", "patch touches " + ", ".join(m.patch_test_files), []))

    if m.submitted and m.patch_empty:
        submit_ids = [s.index for s in steps if s.tool == SUBMIT]
        out.append(Finding("empty_submission", "submitted with an empty patch", submit_ids))

    return out

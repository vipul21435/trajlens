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
                    evidence: the last test step and the last edit step,
                    in step order
gave_up             no submit step and the final patch is empty, or the last
                    step is an error (exit code, or observation text when the
                    format has no exit codes); evidence: the last step. A run
                    with no steps at all and an empty patch is gave_up too,
                    with no evidence ids (an empty or wrongly parsed file
                    looks like this, and it is not a clean run)
tool_error_cascade  an error streak of at least ``cascade_min`` [4] steps;
                    evidence: the streak's step ids
context_blow_up     the observations together reach ``context_chars`` [60000]
                    characters (everything the agent has been shown so far
                    stays in its prompt), or a single observation has at least
                    ``blowup_chars`` [20000] characters; evidence: the step
                    where the running total crosses the limit and the largest
                    observation step
edited_tests        the final patch modifies or deletes a test file that existed
                    before the run; a test file the patch creates (a new test
                    or a reproduction script named test_*.py) does not count
                    unless ``include_new_tests`` is set; evidence: the edit
                    steps aimed at those files (a path argument, or the file
                    last opened for line-range editors), none when no edit
                    step names the file; the file names are listed in the
                    detail
empty_submission    the run submitted and the patch is empty; evidence: the
                    submit step
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from trajlens.metrics import RunMetrics, edit_steps_for
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


def _short(text: str, limit: int = 80) -> str:
    """First line of ``text``, cut to ``limit`` characters."""
    lines = text.strip().splitlines()
    first = lines[0] if lines else ""
    more = len(lines) > 1 or len(first) > limit
    return first[:limit] + (" ..." if more else "")


# One line per class, shown in the HTML report next to the counts.
RULES: dict[str, str] = {
    "looping": "an exact or near-duplicate command loop of at least loop_min_steps steps",
    "never_tested": "at least one edit and no test run",
    "untested_final_edit": "test runs exist but none after the last edit",
    "gave_up": "no submit step, and an empty patch or a failed last step (or no steps at all "
    "and an empty patch)",
    "tool_error_cascade": "at least cascade_min consecutive failed steps",
    "context_blow_up": "observations reach context_chars characters in total, or one "
    "observation has at least blowup_chars",
    "edited_tests": "the final patch modifies or deletes an existing test file",
    "empty_submission": "a submit step with an empty patch",
}


@dataclass
class Finding:
    """One failure class on one run: its label, a readable detail and the evidence step ids."""

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
    context_chars: int = 60000,
    include_new_tests: bool = False,
) -> list[Finding]:
    """Apply every rule to one run and return the findings in rule order."""
    out: list[Finding] = []
    steps = traj.steps

    for lp in m.exact_loops + m.near_loops:
        if len(lp.step_ids) >= loop_min_steps:
            out.append(
                Finding(
                    "looping",
                    f"{lp.kind} repeat of {_short(lp.command)!r} across {len(lp.step_ids)} steps",
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
                sorted(i for i in ids if i is not None),
            )
        )

    if not steps and m.patch_empty:
        out.append(Finding("gave_up", "no steps and empty patch", []))
    if steps:
        last = steps[-1]
        if not m.submitted and (m.patch_empty or m.last_step_error):
            if m.patch_empty:
                why = "empty patch"
            elif last.exit_code is not None:
                why = f"last step exit code {last.exit_code}"
            else:
                why = "last step observation reads as an error"
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
    total = 0
    crossed = None
    for step in steps:
        total += len(step.observation)
        if total >= context_chars:
            crossed = step.index
            break
    if big or crossed is not None:
        parts = []
        if crossed is not None:
            parts.append(f"observations reach {context_chars} chars at step {crossed}")
        if big:
            parts.append(f"largest observation {m.largest_observation_chars} chars")
        ids = [crossed] if crossed is not None else []
        if big and m.largest_observation_step not in ids:
            ids.append(m.largest_observation_step)
        out.append(Finding("context_blow_up", ", ".join(parts), sorted(ids)))

    touched = [
        p for p in m.patch_test_files if include_new_tests or p not in m.patch_new_test_files
    ]
    if touched:
        out.append(
            Finding(
                "edited_tests",
                "patch touches " + ", ".join(touched),
                edit_steps_for(steps, touched),
            )
        )

    if m.submitted and m.patch_empty:
        submit_ids = [s.index for s in steps if s.tool == SUBMIT]
        out.append(Finding("empty_submission", "submitted with an empty patch", submit_ids))

    return out

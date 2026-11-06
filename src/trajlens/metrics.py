"""Per-run behaviour metrics.

Every number here is computed from the step list alone. The window and
similarity thresholds are parameters with plain defaults so the caller can
tune them from the command line.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from trajlens.model import EDIT, TEST, TOOL_CLASSES, Step, Trajectory

_WS = re.compile(r"\s+")

# Paths that count as test files in the final patch.
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing|spec|specs|__tests__)(/|$)|(^|/)(test_[^/]*|[^/]*_test\.[^/]+|"
    r"[^/]*\.test\.[^/]+|[^/]*\.spec\.[^/]+|conftest\.py)$"
)
_DIFF_PATH_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
_PLUS_PATH_RE = re.compile(r"^\+\+\+ (?:b/)?(\S+)$", re.MULTILINE)

# Observation text that marks a failed step when the format records no exit
# code (SWE-agent .traj files, for one). Each pattern is a message printed by
# the Python interpreter, a POSIX shell or the SWE-agent editor itself.
ERROR_PATTERNS: tuple[str, ...] = (
    r"Traceback \(most recent call last\)",
    r"^\w*(Error|Exception): ",
    r": command not found$",
    r": No such file or directory$",
    r"Your proposed edit has introduced new syntax error",
    r"^Your output was not formatted correctly",
    r"^Usage: ",
)
_ERROR_RE = re.compile("|".join(f"(?:{p})" for p in ERROR_PATTERNS), re.MULTILINE)
INFER_MODES = ("auto", "always", "never")


def observation_looks_failed(observation: str) -> bool:
    """True when the observation text matches one of :data:`ERROR_PATTERNS`."""
    return bool(_ERROR_RE.search(observation or ""))


def step_errors(steps: list[Step], infer: str = "auto") -> list[bool]:
    """Whether each step failed.

    A recorded exit code always decides. For a step without one, ``infer``
    says what to do: ``never`` counts it as a success; ``always`` reads the
    observation text; ``auto`` (the default) reads the observation text only
    when no step of the run carries an exit code, so a format that records
    exit codes is never second-guessed.
    """
    if infer not in INFER_MODES:
        raise ValueError(f"infer must be one of {INFER_MODES}, got {infer!r}")
    any_code = any(s.exit_code is not None for s in steps)
    use_text = infer == "always" or (infer == "auto" and not any_code)
    out = []
    for s in steps:
        if s.exit_code is not None:
            out.append(s.exit_code != 0)
        else:
            out.append(use_text and observation_looks_failed(s.observation))
    return out


def normalise_command(command: str) -> str:
    """Collapse whitespace and lower-case a command for duplicate detection."""
    return _WS.sub(" ", command.strip()).lower()


def near_duplicate(a: str, b: str, threshold: float) -> bool:
    """True when two normalised commands are at least ``threshold`` similar.

    Similarity is difflib's ratio, 2*M/T where M is the number of matched
    characters and T the total length of both strings.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


@dataclass
class Loop:
    """A run of repeated commands inside one window."""

    kind: str  # "exact" or "near"
    command: str
    step_ids: list[int]


@dataclass
class RunMetrics:
    run_id: str
    steps: int
    tool_mix: dict[str, int]
    edits: int
    failed_edits: int
    test_runs: int
    failed_test_runs: int
    edits_per_test_run: float | None
    first_edit_step: int | None
    first_test_step: int | None
    exact_loops: list[Loop]
    near_loops: list[Loop]
    looped_steps: int
    error_steps: int
    error_step_ids: list[int]
    errors_inferred: bool
    last_step_error: bool
    error_rate: float
    max_error_streak: int
    error_streak_steps: list[int]
    observation_chars: int
    observation_growth: float | None
    largest_observation_step: int | None
    largest_observation_chars: int
    tests_after_last_edit: bool | None
    last_edit_step: int | None
    last_test_step: int | None
    patch_files: list[str]
    patch_test_files: list[str]
    patch_new_test_files: list[str]
    patch_touches_tests: bool
    patch_empty: bool
    submitted: bool
    duration_seconds: float | None
    tokens_in: int | None
    tokens_out: int | None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def patch_paths(patch: str) -> list[str]:
    """File paths touched by a unified diff, in order, without duplicates.

    Carriage returns are dropped first: some agent frameworks store the
    patch with CRLF line ends, which would hide every path from the regexes.
    """
    patch = patch.replace("\r", "")
    seen: list[str] = []
    for m in _DIFF_PATH_RE.finditer(patch):
        p = m.group(2)
        if p not in seen:
            seen.append(p)
    if not seen:
        for m in _PLUS_PATH_RE.finditer(patch):
            p = m.group(1)
            if p != "/dev/null" and p not in seen:
                seen.append(p)
    return seen


def new_files(patch: str) -> list[str]:
    """Paths the diff creates (``new file mode`` or ``--- /dev/null``)."""
    patch = patch.replace("\r", "")
    out: list[str] = []
    current = None
    prev = ""
    for line in patch.splitlines():
        m = _DIFF_PATH_RE.match(line)
        if m:
            current = m.group(2)
        elif line.startswith("new file mode") and current and current not in out:
            out.append(current)
        elif line.startswith("+++ ") and prev == "--- /dev/null":
            path = line[4:].strip()
            path = path[2:] if path.startswith("b/") else path
            if path not in out:
                out.append(path)
        prev = line
    return out


def is_test_path(path: str) -> bool:
    return bool(_TEST_PATH_RE.search(path))


def _collect(
    commands: list[str],
    start: int,
    window: int,
    skip: set[int],
    same,
    barriers: list[bool] | None = None,
) -> list[int]:
    """Greedily gather step ids whose command matches, each within ``window``
    steps of the previous match. A barrier step between two matches (an edit,
    when the repeated command is not itself an edit) ends the group."""
    ids = [start]
    j = start + 1
    while j < len(commands) and j - ids[-1] < window:
        if j not in skip and commands[j] and same(start, j):
            ids.append(j)
        elif barriers is not None and barriers[j]:
            break
        j += 1
    return ids


def find_loops(
    steps: list[Step],
    window: int = 6,
    min_repeats: int = 3,
    near_threshold: float = 0.9,
    edit_resets: bool = True,
    same_observation: bool = True,
    observation_head: int = 300,
) -> tuple[list[Loop], list[Loop]]:
    """Find exact and near-duplicate command loops.

    Starting from each step, gather later steps whose normalised command
    matches, as long as each match is within ``window`` steps of the previous
    one. A group of at least ``min_repeats`` steps is a loop. Exact loops
    compare normalised strings for equality; near loops use difflib ratio at
    or above ``near_threshold`` against the first command of the group, and
    exclude steps already in an exact loop. Each step is reported in at most
    one loop of each kind.

    With ``edit_resets`` (the default) an edit step between two repeats of a
    command that is not an edit ends the group: rerunning the same test after
    changing the code is the normal edit-test cycle, not a loop. Repeated
    edits are still gathered, since an edit that does not take is the loop.

    With ``same_observation`` (the default) a repeated command joins a loop
    only when the first ``observation_head`` characters of its observation
    are also at least ``near_threshold`` similar to those of the group's
    first step. Paging through a file with ``scroll_down``, or reading ten
    different files under one long directory, repeats the command but brings
    new information each time; the same edit rejected with the same error,
    or the same search with the same empty result, does not.
    """
    commands = [normalise_command(s.command) if s.command else "" for s in steps]
    is_edit = [s.tool == EDIT for s in steps]
    heads = [s.observation[:observation_head] for s in steps]

    def same_obs(a: int, b: int) -> bool:
        if not same_observation or heads[a] == heads[b]:
            return True
        return SequenceMatcher(None, heads[a], heads[b]).ratio() >= near_threshold

    def barriers_for(i: int) -> list[bool] | None:
        return None if not edit_resets or is_edit[i] else is_edit

    exact: list[Loop] = []
    exact_steps: set[int] = set()
    for i, c in enumerate(commands):
        if not c or i in exact_steps:
            continue
        ids = _collect(
            commands,
            i,
            window,
            exact_steps,
            lambda a, b: commands[a] == commands[b] and same_obs(a, b),
            barriers_for(i),
        )
        if len(ids) >= min_repeats:
            exact.append(Loop("exact", steps[i].command.strip(), ids))
            exact_steps.update(ids)

    def near(a: int, b: int) -> bool:
        return near_duplicate(commands[a], commands[b], near_threshold) and same_obs(a, b)

    near_loops: list[Loop] = []
    near_steps: set[int] = set()
    skip = exact_steps | near_steps
    for i, c in enumerate(commands):
        if not c or i in skip:
            continue
        ids = _collect(
            commands,
            i,
            window,
            skip,
            near,
            barriers_for(i),
        )
        if len(ids) >= min_repeats:
            near_loops.append(Loop("near", steps[i].command.strip(), ids))
            near_steps.update(ids)
            skip = exact_steps | near_steps
    return exact, near_loops


def error_streaks(steps: list[Step], errors: list[bool] | None = None) -> list[list[int]]:
    """Maximal runs of consecutive error steps, as lists of step ids.

    ``errors`` gives the failure flag of each step; by default it is the
    recorded exit code (:attr:`Step.is_error`).
    """
    if errors is None:
        errors = [s.is_error for s in steps]
    streaks: list[list[int]] = []
    cur: list[int] = []
    for s, failed in zip(steps, errors, strict=True):
        if failed:
            cur.append(s.index)
        elif cur:
            streaks.append(cur)
            cur = []
    if cur:
        streaks.append(cur)
    return streaks


def observation_growth(steps: list[Step]) -> float | None:
    """Ratio of mean observation length in the last third to the first third.

    None when there are fewer than three steps.
    """
    if len(steps) < 3:
        return None
    third = len(steps) // 3
    head = [len(s.observation) for s in steps[:third]]
    tail = [len(s.observation) for s in steps[-third:]]
    head_mean = sum(head) / len(head)
    tail_mean = sum(tail) / len(tail)
    if head_mean == 0:
        return None if tail_mean == 0 else float("inf")
    return tail_mean / head_mean


def run_metrics(
    traj: Trajectory,
    window: int = 6,
    min_repeats: int = 3,
    near_threshold: float = 0.9,
    infer_errors: str = "auto",
    edit_resets: bool = True,
    same_observation: bool = True,
) -> RunMetrics:
    """Compute every per-run metric for one trajectory.

    ``infer_errors`` is passed to :func:`step_errors`.
    """
    steps = traj.steps
    n = len(steps)
    mix = Counter(s.tool for s in steps)
    tool_mix = {t: mix.get(t, 0) for t in TOOL_CLASSES}
    edit_ids = [s.index for s in steps if s.tool == EDIT]
    test_ids = [s.index for s in steps if s.tool == TEST]
    edits = len(edit_ids)
    test_runs = len(test_ids)
    exact, near = find_loops(
        steps, window, min_repeats, near_threshold, edit_resets, same_observation
    )
    looped = {i for lp in exact + near for i in lp.step_ids}
    failed = step_errors(steps, infer_errors)
    errors = [s.index for s, f in zip(steps, failed, strict=True) if f]
    inferred = any(f and s.exit_code is None for s, f in zip(steps, failed, strict=True))
    streaks = error_streaks(steps, failed)
    longest = max(streaks, key=len) if streaks else []
    obs_lengths = [len(s.observation) for s in steps]
    largest_idx = max(range(n), key=lambda i: obs_lengths[i]) if n else None
    last_edit = edit_ids[-1] if edit_ids else None
    last_test = test_ids[-1] if test_ids else None
    if last_edit is None:
        tests_after = None
    else:
        tests_after = last_test is not None and last_test > last_edit
    files = patch_paths(traj.patch)
    test_files = [p for p in files if is_test_path(p)]
    created = set(new_files(traj.patch))
    new_test_files = [p for p in test_files if p in created]
    stamps = [s.timestamp for s in steps if s.timestamp is not None]
    duration = (max(stamps) - min(stamps)) if len(stamps) >= 2 else None
    tin = [s.tokens_in for s in steps if s.tokens_in is not None]
    tout = [s.tokens_out for s in steps if s.tokens_out is not None]
    # Some formats only carry run totals (SWE-agent model_stats); use them
    # when no step has its own count.
    total_in = sum(tin) if tin else traj.extra.get("tokens_in")
    total_out = sum(tout) if tout else traj.extra.get("tokens_out")
    return RunMetrics(
        run_id=traj.run_id,
        steps=n,
        tool_mix=tool_mix,
        edits=edits,
        failed_edits=sum(1 for s, f in zip(steps, failed, strict=True) if f and s.tool == EDIT),
        test_runs=test_runs,
        failed_test_runs=sum(1 for s, f in zip(steps, failed, strict=True) if f and s.tool == TEST),
        edits_per_test_run=(edits / test_runs) if test_runs else None,
        first_edit_step=edit_ids[0] if edit_ids else None,
        first_test_step=test_ids[0] if test_ids else None,
        exact_loops=exact,
        near_loops=near,
        looped_steps=len(looped),
        error_steps=len(errors),
        error_step_ids=errors,
        errors_inferred=inferred,
        last_step_error=bool(failed[-1]) if failed else False,
        error_rate=(len(errors) / n) if n else 0.0,
        max_error_streak=len(longest),
        error_streak_steps=list(longest),
        observation_chars=sum(obs_lengths),
        observation_growth=observation_growth(steps),
        largest_observation_step=largest_idx,
        largest_observation_chars=obs_lengths[largest_idx] if largest_idx is not None else 0,
        tests_after_last_edit=tests_after,
        last_edit_step=last_edit,
        last_test_step=last_test,
        patch_files=files,
        patch_test_files=test_files,
        patch_new_test_files=new_test_files,
        patch_touches_tests=bool(test_files),
        patch_empty=not traj.patch.strip(),
        submitted=any(s.tool == "submit" for s in steps),
        duration_seconds=duration,
        tokens_in=total_in,
        tokens_out=total_out,
    )

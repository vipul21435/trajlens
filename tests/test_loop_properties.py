"""Properties of loop and error-streak detection over random step lists."""

from hypothesis import given, settings
from hypothesis import strategies as st

from trajlens.metrics import error_streaks, find_loops, step_errors
from trajlens.model import Step

step = st.builds(
    lambda tool, cmd, obs, code: (tool, cmd, obs, code),
    st.sampled_from(["edit", "test", "shell", "read"]),
    st.sampled_from(["ls", "ls ", "pytest", "pytest -x", "edit 1:2", "cat a.py"]),
    st.sampled_from(["", "ok", "E"]),
    st.sampled_from([None, 0, 1]),
)


def build(rows):
    return [Step(i, t, c, observation=o, exit_code=e) for i, (t, c, o, e) in enumerate(rows)]


@settings(max_examples=300, deadline=None)
@given(
    st.lists(step, max_size=30),
    st.integers(1, 8),
    st.integers(2, 5),
    st.booleans(),
    st.booleans(),
)
def test_loop_invariants(rows, window, min_repeats, resets, same_obs):
    steps = build(rows)
    exact, near = find_loops(
        steps, window, min_repeats, edit_resets=resets, same_observation=same_obs
    )
    for loops in (exact, near):
        seen: set[int] = set()
        for lp in loops:
            ids = lp.step_ids
            assert len(ids) >= min_repeats
            assert ids == sorted(ids)
            assert all(b - a <= window for a, b in zip(ids, ids[1:], strict=False))
            assert not seen & set(ids)
            seen |= set(ids)
    in_exact = {i for lp in exact for i in lp.step_ids}
    assert not in_exact & {i for lp in near for i in lp.step_ids}


@settings(max_examples=300, deadline=None)
@given(st.lists(step, max_size=30))
def test_error_streaks_match_a_plain_count(rows):
    steps = build(rows)
    errors = step_errors(steps)
    expected, cur = [], []
    for s, bad in zip(steps, errors, strict=True):
        if bad:
            cur.append(s.index)
        elif cur:
            expected.append(cur)
            cur = []
    if cur:
        expected.append(cur)
    assert error_streaks(steps, errors) == expected

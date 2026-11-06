# trajlens

Read coding-agent trajectories, measure behaviour, classify failures.

Status: first version. The step model, the generic JSONL adapter, the per-run
metrics, the aggregate, and the rule-based failure classes work end to end on
the fixtures in `tests/fixtures/`. Not implemented yet: the SWE-agent,
mini-swe-agent and OpenHands adapters, the HTML report, and results on public
trajectory data.

## Why this is hard

A trajectory is a long, loosely structured log: free text, tool calls, shell
commands and their output, interleaved in whatever shape the agent framework
chose. Pass/fail on the task says nothing about how the agent got there.
Deciding that an agent looped, never ran the tests, or gave up means reading
every step, matching commands that differ only in whitespace or a file name,
and keeping track of which step the judgement rests on so a reader can check
it. Doing that by hand does not scale past a handful of runs; doing it with
opaque heuristics makes the labels untrustworthy. trajlens keeps every rule
small, documented and tied to the step ids it used.

## Install

```
pip install git+https://github.com/vipul21435/trajlens
```

## Quickstart

```
$ trajlens analyze tests/fixtures/loop.jsonl tests/fixtures/good.jsonl
loop: 7 steps, 1 edits, 0 test runs, error rate 0.71, first edit 1, first test None, tests after last edit False, patch files 0
  looping: exact repeat of "python -c 'import calc'" across 5 steps [steps 2,3,4,5,6]
  never_tested: 1 edits and no test run [steps 1]
  gave_up: no submit step and empty patch [steps 6]
  tool_error_cascade: 5 consecutive error steps [steps 2,3,4,5,6]
good: 7 steps, 1 edits, 2 test runs, error rate 0.14, first edit 4, first test 3, tests after last edit True, patch files 1
runs: 2
mean steps: 7.0
mean error rate: 0.429
runs with finding: looping=1, never_tested=1, gave_up=1, tool_error_cascade=1
```

`--json report.json` writes every metric and finding as JSON. Thresholds are
flags: `--window`, `--min-repeats`, `--near-threshold`, `--loop-min-steps`,
`--cascade-min`, `--blowup-chars`, `--growth-ratio`.

## Input format

One trajectory per JSONL file, one step per line. Keys: `thought`, `action`,
`command`, `observation`, `exit_code`, `timestamp`, `tokens_in`,
`tokens_out`. A line with `"type": "meta"` carries `run_id`, `task_id`,
`model` and optionally `patch`; a line with `"type": "patch"` carries the
final diff. See `src/trajlens/adapters/generic.py` and the fixtures.

## How it works

Every step is mapped to one tool class: edit, test, read, search, shell,
submit, other. Known action names win (`edit`, `str_replace_editor`, `open`,
`submit`); a shell action is classified by its command with a small set of
regular expressions (test runners such as `pytest`, `npm test`, `cargo test`;
in-place edits such as `sed -i` and `git apply`; pagers; grep and find).

Per run: step count, tool mix, edits, test runs, edits per test run, step of
the first edit and first test, last edit and last test and whether a test ran
after the last edit, error steps and the longest error streak, total
observation characters, the largest observation, observation growth (mean
length of the last third of steps over the first third), files in the final
patch and which of them are test files, whether a submit step exists, wall
clock from timestamps and token totals when present.

Loops: commands are normalised (whitespace collapsed, lower-cased). Starting
from each step, later steps whose command matches are gathered as long as each
match is within `--window` (6) steps of the previous match; a group of at
least `--min-repeats` (3) is a loop. Exact loops compare strings; near loops
use difflib's ratio (2M/T) at or above `--near-threshold` (0.9) against the
first command of the group.

Failure classes, each with the step ids it rests on (defaults in brackets):

| class | rule |
| --- | --- |
| looping | an exact or near loop of at least `--loop-min-steps` [4] steps |
| never_tested | at least one edit and no test run |
| untested_final_edit | test runs exist but none after the last edit |
| gave_up | no submit step and (empty patch or last step is an error) |
| tool_error_cascade | at least `--cascade-min` [4] consecutive error steps |
| context_blow_up | one observation of at least `--blowup-chars` [20000] characters, or growth of at least `--growth-ratio` [5.0] with a largest observation of at least 2000 characters |
| edited_tests | the final patch touches a test file |
| empty_submission | a submit step with an empty patch |

## Results on real public data

None yet. This version was run only on the synthetic fixtures in
`tests/fixtures/` (5 runs, 2 to 7 steps each). Test suite: 45 tests, 100 %
line coverage of `src/trajlens` as measured by `uv run pytest -q --cov=trajlens`.

## Limitations

- Only the generic JSONL format is read. The other adapters are not written.
- Tool classification is regular expressions over the command string; a test
  run inside a script, or an editor invoked under an unusual name, is missed.
- Test-file detection is by path only.
- Near-duplicate detection compares strings, so two commands that mean the
  same thing but look different are not matched.
- The rules were tuned on five synthetic runs. Thresholds are flags for that
  reason.
- No HTML report.

## Related tools

- https://github.com/vipul21435/evalstats
- https://github.com/vipul21435/testlogs
- https://github.com/vipul21435/patchmin
- https://github.com/vipul21435/agentenv
- https://github.com/vipul21435/taskledger

## Licence

MIT. See LICENSE.

Built by Vipul Raj Jha · https://vipul21435.github.io

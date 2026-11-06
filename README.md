# trajlens

Read coding-agent trajectories from several open formats, measure how the agent behaved, and
label failures with small documented rules that each point at the steps they rest on.

## Why this is hard

A trajectory is a long, loosely structured log: free text, tool calls, shell commands and
their output, interleaved in whatever shape the agent framework chose. Pass or fail on the
task says nothing about how the agent got there. Deciding that an agent looped, never ran the
tests, or gave up means reading every step, matching commands that differ only in whitespace
or a line number, telling an edit-test cycle apart from a stuck agent, and noticing failures
in formats that record no exit codes at all. Doing that by hand does not scale past a handful
of runs; doing it with opaque heuristics makes the labels untrustworthy. trajlens keeps every
rule small, documented, configurable from the command line, and tied to the step ids it used,
so a reader can open the run at that step and check the call.

## Install

```
pip install git+https://github.com/vipul21435/trajlens
```

Python 3.10 or later; no runtime dependencies outside the standard library.

## Quickstart

On three of the fixtures in `tests/fixtures/` (two generic JSONL files and one SWE-agent
`.traj`):

```
$ trajlens analyze tests/fixtures/loop.jsonl tests/fixtures/good.jsonl tests/fixtures/sample.traj --html report.html
loop: 7 steps, 1 edits, 0 test runs, error rate 0.71, first edit 1, first test None, tests after last edit False, patch files 0
  looping: exact repeat of "python -c 'import calc'" across 5 steps [steps 2,3,4,5,6]
  never_tested: 1 edits and no test run [steps 1]
  gave_up: no submit step and empty patch [steps 6]
  tool_error_cascade: 5 consecutive error steps [steps 2,3,4,5,6]
good: 7 steps, 1 edits, 2 test runs, error rate 0.14, first edit 4, first test 3, tests after last edit True, patch files 1
sample: 5 steps, 1 edits, 1 test runs, error rate 0.00, first edit 2, first test 3, tests after last edit True, patch files 1
runs: 3
mean steps: 6.3
mean error rate: 0.286
runs with finding: looping=1, never_tested=1, gave_up=1, tool_error_cascade=1
runs with no finding: 2
```

`--json report.json` writes every metric and finding. `--html report.html` writes one
self-contained page (inline styles, no server, no CDN, no external requests): a summary, the
tool mix, the failure-class table with the rule text, a run table that filters by class, and
per-run step tables in which every evidence step is highlighted and linked from its finding.
The report data is embedded in the page as JSON.

`trajlens convert RUN... --out DIR` rewrites any supported input as generic JSONL.

The example script analyses every fixture and writes both reports:

```
uv run python examples/analyze_fixtures.py out/
```

## How it works

### Inputs and the step model

Every adapter produces the same `Trajectory`: a run id, an ordered list of `Step`s, the
final patch, and optional run-level fields (`task_id`, `model`, `exit_status`, `resolved`).
A step holds `thought`, the tool class, the raw action name, `command`, `observation`,
`exit_code`, `timestamp`, `tokens_in` and `tokens_out`; whatever the source format lacks
stays `None`.

| `--format` | input | detected from |
| --- | --- | --- |
| `generic` | one JSONL file per run, documented below | `*.jsonl` |
| `sweagent` | SWE-agent `.traj` (`trajectory` list, or `history` messages), `info.submission` as the patch, `info.model_stats` token totals | `*.traj`, or a JSON object with `trajectory` |
| `minisweagent` | mini-swe-agent `.traj.json` (message list, or object with `messages` and `info`); exit code from `<returncode>` | `*.traj.json`, or role/content message lists |
| `openhands` | OpenHands event list (`action`/`observation` events linked by `cause`), exit code from `extras.exit_code`, ISO timestamps, cumulative token usage turned into per-step counts | a JSON list of events, or an object with `events` |
| `chat` | a model transcript: messages with `role` (`assistant`/`ai` for the agent, `user`/`tool` for observations) and `content` or `text`; the command is the last fenced block of each agent turn, as SWE-agent's thought-action parser takes it | never guessed; pass `--format chat` |

Generic JSONL: one step per line with any of `thought`, `action`, `command`, `observation`,
`exit_code`, `timestamp` (epoch seconds), `tokens_in`, `tokens_out`, and optionally `tool`
(one of the classes below). A line with `"type": "meta"` carries `run_id`, `task_id`,
`model`, `patch`, `exit_status` and `resolved` (true or false); a line with
`"type": "patch"` carries the final diff.

The OpenHands adapter was written against the `event_to_dict` serialisation in the OpenHands
source and tested on a hand-written event list; it has not been run on a published OpenHands
trajectory set (see Limitations).

### Tool classes

Every step gets one class: `edit`, `test`, `read`, `search`, `shell`, `submit`, `other`.
Known action names decide first (`edit`, `create`, `str_replace_editor`, `open`, `submit`,
OpenHands `edit`/`write`/`read`/`finish`). A shell command is classified by regular
expressions: test runners (`pytest`, `tox`, `go test`, `cargo test`, `npm test`, `jest`,
`mvn test`, ...) and running a test or reproduction script directly (`python reproduce.py`,
`python tests/test_io.py`) are `test`; `sed -i`, `patch`, `git apply` are `edit`; pagers and
`cat` are `read`; `grep`, `find`, `ls` are `search`. Counting `python reproduce.py` as a test
run follows SWE-agent's default instructions, which ask the agent to reproduce the bug with a
script and rerun it after the fix; without it nearly every SWE-agent run would be labelled
`never_tested`.

### Failed steps

A recorded exit code decides whether a step failed. SWE-agent `.traj` files and chat
transcripts record none, so by default (`--infer-errors auto`) a run with no exit codes at
all has its failures read from the observation text: a Python traceback, a line starting
with `SomethingError:` or `SomethingException:`, `command not found`,
`No such file or directory`, a `Usage:` line, and the two messages SWE-agent itself prints
for a rejected edit and for a malformed reply. `always` applies the patterns to every step
without an exit code; `never` turns inference off. The report marks runs whose error count
was inferred (`errors_inferred`).

### Per-run metrics

Steps; tool mix; edits and failed edits; test runs and failed test runs; edits per test run;
step of the first and last edit and test; whether a test ran after the last edit; error steps,
error rate (error steps / steps) and the longest streak of consecutive failed steps; total
observation characters, the largest observation, and observation growth (mean observation
length of the last third of steps divided by the first third); files in the final patch,
which of them are test files (by path: `tests/`, `test_*.py`, `*_test.*`, `*.spec.*`,
`conftest.py`, ...), and which the patch creates; whether a submit step exists; duration from
timestamps; token totals.

### Loops

Commands are normalised (whitespace collapsed, lower-cased). Starting from each step, later
steps whose command matches are gathered while each match is within `--window` [6] steps of
the previous one; a group of at least `--min-repeats` [3] is a loop. Exact loops compare the
normalised strings; near loops compare with difflib's ratio, 2M/T where M is the number of
matched characters and T the total length of both strings, at or above `--near-threshold`
[0.9] against the group's first command, and skip steps already in an exact loop. An edit
between two repeats of a non-edit command ends the group: rerunning the same test after
changing the code is the edit-test cycle, not a loop. Repeated edits are still grouped,
because an edit that keeps failing to apply is exactly the loop to catch.
`--no-edit-reset` turns this off. On the 400-run sample below, the edit reset takes resolved
runs labelled `looping` from 16 of 76 to 4 of 76 and unresolved runs from 149 of 324 to 116
of 324 (compare `trajlens analyze --format chat nebius-sample/row*.json` with and without
`--no-edit-reset`).

### Failure classes

Each class is a rule over the metrics; a run can carry several. Defaults in brackets.

| class | rule | evidence step ids |
| --- | --- | --- |
| looping | an exact or near loop of at least `--loop-min-steps` [4] steps | the loop's steps |
| never_tested | at least one edit and no test run | the edit steps |
| untested_final_edit | test runs exist but none after the last edit | last edit, last test |
| gave_up | no submit step, and an empty patch or a failed last step | the last step |
| tool_error_cascade | at least `--cascade-min` [4] consecutive failed steps | the streak |
| context_blow_up | observations reach `--context-chars` [60000] characters in total, or one observation has at least `--blowup-chars` [20000] | the step where the total crosses the limit; the largest observation |
| edited_tests | the final patch modifies or deletes a test file that existed before the run (created test files count only with `--include-new-tests`) | none; the files are named in the detail |
| empty_submission | a submit step and an empty patch | the submit steps |

Why these rules: `never_tested`, `untested_final_edit` and `edited_tests` follow the SWE-bench
setup, where a patch is judged by the repository's own tests and the evaluation harness
applies the task's test patch over the agent's before running them (`swebench/harness` in
github.com/SWE-bench/SWE-bench), so edits to existing test files are at best overwritten and at
worst a sign of making tests pass without fixing the code. `context_blow_up` counts the total of
all observations because an agent framework keeps the history in the prompt, so it is the
total, not the largest single output, that exhausts the context window; the first version
used an observation-growth ratio, and on the real sample the runs that the framework stopped
for context length had lower growth than the others, so the rule was replaced. The 60000
default was set by looking at the first 100 rows of the sample (every run stopped for context
had at least 61065 characters of observations) and is then checked on the 300 rows held out.
`gave_up` and `empty_submission` separate a run that stopped from one that submitted nothing.

## Results on real public data

Data: 400 runs drawn from
[nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
(80,036 SWE-agent runs on GitHub issues, with the final patch, the framework's exit status,
and whether the task's tests passed on the patch; licence CC BY 4.0, checked on the dataset
page). The sample is 100 random offsets with seed 0, four consecutive rows each, read through
the public datasets-server rows endpoint; the test log column is dropped. 76 of the 400 runs
are resolved. No row is committed to this repository; the script downloads them (about
26 MB).

```
uv run python scripts/fetch_nebius_sample.py --out nebius-sample
uv run --with matplotlib python scripts/real_data_report.py nebius-sample --chart docs/images/classes_by_outcome.png
trajlens analyze --format chat nebius-sample/row*.json --json report.json --html report.html
```

The last command, on all 400 files including the HTML report, took 2.9 s wall clock on the
machine used here (`time`, one run). The tables are the output of the second command, unchanged.

Behaviour over all 400 runs, default options:

| metric | value |
| --- | --- |
| runs | 400 |
| mean steps | 25.2 |
| mean edits | 8.5 |
| mean failed edits (inferred) | 3.3 |
| mean test runs | 3.0 |
| mean error rate (inferred) | 0.195 |
| mean first edit step (runs with an edit) | 2.3 |
| mean first test step (runs with a test) | 4.6 |
| runs with a loop | 182 |
| runs testing after the last edit | 209 |
| runs submitted | 267 |
| runs with an empty patch | 49 |
| runs with no finding | 128 |
| tool mix (share of steps) | edit 33.8%, test 12.0%, read 17.9%, search 21.9%, shell 11.8%, submit 2.7% |

"Runs with a loop" counts any loop of three or more repeats; the `looping` class below needs
four.

Share of runs carrying each class, resolved against unresolved (95 % Wilson intervals):

| class | resolved runs (n=76) | unresolved runs (n=324) |
| --- | --- | --- |
| looping | 4/76 = 5.3% [2.1%, 12.8%] | 116/324 = 35.8% [30.8%, 41.2%] |
| never_tested | 21/76 = 27.6% [18.8%, 38.6%] | 88/324 = 27.2% [22.6%, 32.3%] |
| untested_final_edit | 1/76 = 1.3% [0.2%, 7.1%] | 69/324 = 21.3% [17.2%, 26.1%] |
| gave_up | 0/76 = 0.0% [0.0%, 4.8%] | 35/324 = 10.8% [7.9%, 14.7%] |
| tool_error_cascade | 4/76 = 5.3% [2.1%, 12.8%] | 59/324 = 18.2% [14.4%, 22.8%] |
| context_blow_up | 3/76 = 3.9% [1.4%, 11.0%] | 114/324 = 35.2% [30.2%, 40.5%] |
| edited_tests | 2/76 = 2.6% [0.7%, 9.1%] | 6/324 = 1.9% [0.9%, 4.0%] |
| empty_submission | 0/76 = 0.0% [0.0%, 4.8%] | 14/324 = 4.3% [2.6%, 7.1%] |
| (no finding) | 46/76 = 60.5% [49.3%, 70.8%] | 82/324 = 25.3% [20.9%, 30.3%] |

![Share of runs with each failure class, resolved against unresolved](docs/images/classes_by_outcome.png)

On the 300 held-out rows alone (52 resolved, 248 unresolved) the pattern is the same:
looping 7.7 % against 37.1 %, untested_final_edit 0.0 % against 20.6 %, context_blow_up
3.8 % against 37.5 %, no finding 61.5 % against 23.4 %.

Agreement of two rules with the exit status the framework recorded (the rules never read it):

| rule vs status | rows | tp | fp | fn | tn | precision | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| context_blow_up vs exit_context | tuning | 18 | 4 | 0 | 78 | 0.82 | 1.00 |
| gave_up vs not submitted | tuning | 10 | 0 | 0 | 90 | 1.00 | 1.00 |
| context_blow_up vs exit_context | held-out | 86 | 9 | 11 | 194 | 0.91 | 0.89 |
| gave_up vs not submitted | held-out | 23 | 2 | 0 | 275 | 0.92 | 1.00 |

Grouped by the exit status the framework recorded
(`trajlens analyze --format chat nebius-sample/row*.json --group-by exit_status`, runs carrying
each class):

| exit status | runs | resolved | looping | untested_final_edit | gave_up | tool_error_cascade | context_blow_up | empty_submission |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| submitted | 258 | 28.7% | 26 | 16 | 0 | 17 | 11 | 5 |
| submitted (exit_context) | 100 | 2.0% | 77 | 44 | 2 | 34 | 92 | 0 |
| early_exit | 18 | 0.0% | 7 | 10 | 18 | 5 | 2 | 0 |
| exit_context | 15 | 0.0% | 10 | 0 | 15 | 7 | 12 | 0 |
| submitted_no_patch | 9 | 0.0% | 0 | 0 | 0 | 0 | 0 | 9 |

77 of the 100 runs that were cut off for context length and then auto-submitted contain a
loop of four or more steps, against 26 of the 258 runs that submitted on their own. That is
consistent with repeated actions filling the context, though counts like these cannot
separate cause from effect.

What this says and does not say. `looping`, `untested_final_edit`, `context_blow_up` and
`gave_up` are far more common in unresolved runs; `never_tested` is not (27.6 % against
27.2 %), so on this data an agent that edits without running anything is no less likely to
pass, and the class describes process, not outcome. `edited_tests` is rare in both groups.
These are associations in one dataset produced by one agent scaffold with three Llama-based
models (360 of the 400 runs come from the 70B one), not causes, and resolved runs are a minority (19 %), so their intervals are wide.

Smaller checks on the other formats, all from the MIT-licensed SWE-agent and mini-swe-agent
repositories (licences read from the LICENSE files on GitHub):

```
trajlens analyze swe-agent-trajectories/*.traj
```

on the 22 `.traj` files under `trajectories/demonstrations/` and `tests/test_data/trajectories/`
of github.com/SWE-agent/SWE-agent (9 of them capture-the-flag demonstrations, not code
fixes): 22 runs, mean 10.5 steps, mean error rate 0.057 (inferred), findings looping 4,
never_tested 9, context_blow_up 1, edited_tests 2, empty_submission 1, no finding 10. The two
mini-swe-agent sample files under `tests/test_data/` of github.com/SWE-agent/mini-swe-agent
parse to 10 and 2 steps; both are labelled empty_submission because they are plain message
lists, a form that carries no submission field, so the patch is empty.

## Limitations

- Tool classes come from action names and regular expressions over the command. A test run
  hidden inside a shell script, an editor under an unusual name, or a `python -c` one-liner
  that exercises the fix is missed; `python reproduce.py` counts as a test even when the script
  checks nothing.
- Error inference reads observation text with a fixed list of patterns. A failing step that
  prints none of them is a success to trajlens, and a step that prints an old traceback it
  merely displayed (for example `cat`-ing a log) is a failure.
- Test-file detection is by path only.
- Near-duplicate detection compares strings, so two commands that do the same thing but look
  different are not matched, and two different edits to neighbouring lines can be.
- The `context_blow_up` default was chosen on 100 rows of one dataset with one scaffold; other
  frameworks truncate observations or summarise history differently, so `--context-chars`
  should be set per framework. Characters, not tokens, are counted; no tokenizer is used.
- The OpenHands adapter has only been run on a synthetic event list; no public OpenHands
  trajectories were analysed here. The mini-swe-agent adapter has only been run on the two
  sample files in that repository.
- `edited_tests` sees only the final patch, not test files edited and later reverted.
- No per-step outcome model: a run's resolution is read from the input when present, never
  computed. The rules label behaviour; they do not predict success.

## Related tools

- [graderguard](https://github.com/vipul21435/graderguard): audit graders for reward hacking
  and flakiness, the grader-side view of what `edited_tests` flags in a run.
- [taskledger](https://github.com/vipul21435/taskledger): schema validation and lint rules for
  benchmark-task bundles.
- [datasieve](https://github.com/vipul21435/datasieve): validate and deduplicate SFT and
  preference data, for when trajectories become training data.
- [reporewind](https://github.com/vipul21435/reporewind): resolve Python fix commits into
  source and test patches.

## Licence

MIT. See LICENSE.

Built by Vipul Raj Jha · https://vipul21435.github.io

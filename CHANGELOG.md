# Changelog

## 0.1.0 - 2026-11-06

### Added

- One step model shared by every adapter: thought, tool class, raw action, command,
  observation, exit code, timestamp and token counts per step; patch, task id, model, exit
  status and resolved flag per run.
- Adapters for generic JSONL (documented), SWE-agent `.traj`, mini-swe-agent `.traj.json`,
  OpenHands event JSON, chat transcripts with fenced commands, and function-calling
  transcripts; format detection by file name and document shape.
- Per-run metrics: tool mix, edits and test runs (with failed counts), first and last edit
  and test, tests after the last edit, error rate and streaks (failures inferred from
  observation text when a format has no exit codes), observation size and growth, patch
  files, test files and created files.
- Loop detection for exact and near-duplicate repeats, with `--loop-min-steps` and related
  thresholds as flags.
- Eight failure classes, each with the step ids it rests on: looping, never_tested,
  untested_final_edit, gave_up, tool_error_cascade, context_blow_up, edited_tests,
  empty_submission.
- `trajlens analyze`: aggregate report with a resolved/unresolved split and `--group-by`;
  text, JSON, CSV and one self-contained HTML output; `--fail-on` exit status; `--skip-bad`
  to go on past malformed files.
- `trajlens show` to print one run step by step.
- `trajlens audit` to draw a seeded sample of runs carrying one class for manual checking.
- `trajlens compare` for class-share differences between two reports with Newcombe
  intervals.
- `trajlens convert` to write any supported format as generic JSONL.
- Python API: `load`, `analyze_trajectories` and `Options` exported from the package.
- Scripts to fetch a seeded sample of nebius/SWE-agent-trajectories or
  nebius/SWE-rebench-openhands-trajectories, and to report class shares by outcome with
  Wilson intervals and a matplotlib chart.
- README with the step model, rule definitions, results on 400 SWE-agent and 60 OpenHands
  public runs, and a Limitations section.

### Known limitations

See the Limitations section of the README for what is not implemented and where the rules
are approximate.

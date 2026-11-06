# Changelog

## 0.1.0

- One step model shared by every adapter: thought, tool class, raw action, command,
  observation, exit code, timestamp, token counts; run-level patch, task id, model, exit
  status and resolved flag.
- Adapters: generic JSONL (documented), SWE-agent `.traj`, mini-swe-agent `.traj.json`,
  OpenHands event JSON, chat transcripts with fenced commands, and function-calling
  transcripts (`tool_calls` / `tool` messages, OpenHands exit-code lines); format detection
  by file name and document shape.
- Per-run metrics: tool mix, edits and test runs (with failed counts), first and last edit
  and test, tests after the last edit, exact and near-duplicate loops (an edit between repeats
  ends a loop, and repeats must also bring a similar observation),
  error rate and streaks (with failures inferred from observation text when a format has no
  exit codes), observation size and growth, patch files, test files and created files.
- Failure classes with evidence step ids: looping, never_tested, untested_final_edit,
  gave_up, tool_error_cascade, context_blow_up, edited_tests, empty_submission.
- Aggregate report with a resolved/unresolved split and `--group-by`; JSON, CSV and one
  self-contained HTML report; `--fail-on` exit status; `trajlens show` for one run step by
  step; `trajlens convert` to generic JSONL.
- Scripts to fetch a seeded sample of nebius/SWE-agent-trajectories or
  nebius/SWE-rebench-openhands-trajectories and to report the class shares by outcome with
  Wilson intervals and a matplotlib chart.

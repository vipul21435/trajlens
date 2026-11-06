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
  exit codes), runs of scripts the agent created counted as test runs, observation size and growth, patch files, test files and created files.
- Failure classes with evidence step ids: looping, never_tested, untested_final_edit,
  gave_up, tool_error_cascade, context_blow_up, edited_tests, empty_submission.
- Aggregate report with a resolved/unresolved split and `--group-by`; JSON, CSV and one
  self-contained HTML report; `--fail-on` exit status; `trajlens show` for one run step by
  step; `trajlens audit` for a seeded sample of runs carrying one class; `trajlens compare`
  for class-share differences between two reports with Newcombe intervals; `trajlens
  convert` to generic JSONL.
- Scripts to fetch a seeded sample of nebius/SWE-agent-trajectories or
  nebius/SWE-rebench-openhands-trajectories and to report the class shares by outcome with
  Wilson intervals and a matplotlib chart.
- Review fixes before the first release: `cat`/`echo`/`printf`/`tee` file writes are edits
  and count as created scripts in any adapter; chat and tool-call rows under a `trajectory`
  key are detected; the HTML report escapes every `<` in its embedded JSON; OpenHands actions
  no longer take a later action's observation; malformed files give a one-line error naming
  the file, and `--skip-bad` goes on past them; observation growth is null instead of
  infinity and reports are strict JSON; a run with no steps and an empty patch is `gave_up`;
  deleted, renamed and quoted test paths and Django `tests.py` count for `edited_tests`; a
  test runner counts only as a command word, with Django's runners added; OpenHands timeout
  notices are failed steps; analyze thresholds and `--fail-on` are checked up front; null
  generic meta fields keep their defaults; `load`, `analyze_trajectories` and `Options` are
  exported from the package.
- Third review round: `str_replace_editor view` is a read and only its other sub-commands
  are edits, in every adapter; `str_replace_editor create` counts as a created script;
  OpenHands exit codes are also read from `extras.metadata.exit_code`; an exact loop shorter
  than `--loop-min-steps` can be part of a near loop; mini-swe-agent message lists take their
  patch from the submit command's output; `untested_final_edit` evidence is in step order;
  generic JSONL accepts ISO 8601 timestamps.

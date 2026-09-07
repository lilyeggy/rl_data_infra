# Certified Pi + selected SWE-bench loop

`scripts/run_selected_swebench.py` is the real-host collection entry point for
the constrained A6000 setup. It runs Pi against the local controlled model,
records model token IDs and behavior logprobs at the proxy, saves raw Pi output
and workspace changes, then finalizes a canonical `Episode` and
`ExecutionBundle`.

The current evaluator is intentionally named
`selected-swebench-isolated-test-patch`: `scripts/verify_swebench_selected.py`
checks the agent patch in a fresh clone at the pinned base commit, applies the
hidden selected test patch, and runs its changed tests using that task's pinned
virtual environment. It is a useful reproducible gate, but it is **not** the
official SWE-bench Docker evaluator and must not be reported as one.

Each run directory is immutable and contains:

- `raw-events.jsonl`, `model-evidence.jsonl`, and content-addressed `objects/`;
- the frozen execution manifest, harness launch plan, producer artifact, and
  verifier output;
- `finalized/episode.json` and `finalized/execution-bundle.json` when evidence
  joins successfully.

Run fixed-policy comparisons with `scripts/evaluate_harness_regression.py`.
It reads only finalized complete/valid runs, requires three per policy before a
promotion decision, and requires strict solve-rate improvement. An inconclusive
or rejected result must not promote a Harness candidate.

The first real Flask runs established the data-path behavior: both baseline and
reproduction-first candidate were complete and verifier-valid failures. They
are observable regression evidence, not SFT positive examples; the candidate
remains `INCONCLUSIVE` until the fixed batch is sufficiently replicated.

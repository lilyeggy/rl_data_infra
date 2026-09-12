# Stage D diagnostics run1 — NO LEARNING SIGNAL (16/16 unresolved)

Run: `dstage1`, 2026-09-09 ~23:47 – 2026-09-10 ~00:05 CST.
Candidates (fixed rule: first 4 sorted train IDs not in holdout):
`apps-train-000000 … 000003`. 4 trajectories each, sequential, GPU1 only.
Evidence: `run1/diagnostics.jsonl`, `run1/selection.json`.

Result: 16/16 `resolved: false`. Tool activity present in 4/16 trajectories
(33–44 toolCall refs); remaining 12 ended after a single model call with no
tool use. Verifier path verified healthy (manifest well-formed, case executed,
solution outputs TODO → fail is genuine).

Per closeout §D this batch yields `BLOCKED_NO_LEARNING_SIGNAL` **for these 4
tasks**: no intra-group reward variance, must not fabricate. It is NOT a
project-level block: the plan allows up to 4 fixed candidates, and run2
(tasks 5–8, same frozen rule) is underway to find tasks with variance.

# Canonical Agent Data Plane → SFT/RL data flow

This document is the canonical, up-to-date data flow for turning a real
Harness execution into multiple training/analysis views. It replaces the older
ad-hoc flow and is the contract the project should be measured against.

```text
Real Harness (Pi today; other harnesses later)
        │  raw protocol events
        ▼
[ capture ]  TraceEvent immutable stream  (src/capture)
        │  harness-specific tool names/absolute paths are EVIDENCE only
        ▼
[ assemble + certify ]  AgentEpisode, integrity, verifier  (already built)
        │  verifier PASSED + execution VALID + integrity COMPLETE
        ▼
[ Pi → Canonical adapter ]  CanonicalEpisode  (src/capture/pi_canonical_adapter.py)
        │  canonical actions; workspace-relative paths; lossy markers;
        │  strict call/result pairing; no first-action drop; quarantine
        ▼
[ multi-view exporters ]  (src/exporters/canonical/)
        ├── Generic Agent SFT view      ── decision-role examples, neutral
        ├── Model-native Tool SFT       ── Qwen/OpenAI/JSON renderers (format only)
        └── Harness Improvement metrics ── tool/token/path/recovery analysis
        ▼
[ canonical SFT dataset ]  (src/learning + scripts/build_canonical_*)
        │  role stats, leak gate, dedup, dataset card, manifest/checksum
        ▼
[ training ]  (scripts/train_qwen_lora_sft.py, LoRA on A6000)
        ▼
[ layered eval ]  (src/evaluation/layered_eval.py)
        ├── FormatEval            (legal/schema/parseable — FORMAT_FAILURE isolated)
        ├── ActionSelectionEval   (tool choice, path, normalization, verifier)
        ├── TrajectoryReplayEval  (recovery, adjust, avoid repeats)
        └── End-to-End (harness)  (Gates 2-4)
        ▼
[ harness improvement loop ]  (src/evaluation + regression gate)
```

## Honest boundaries (must not be violated)

- **Pi tool names / absolute paths / Pi system prompt** appear only in the Pi
  adapter and in `arguments`/evidence. Canonical actions, training views, and
  exported data must not leak them.
- **`<tool_call>` and model-specific grammars** appear only inside the
  model-native exporter renderers. The CanonicalEpisode never contains them.
- **Format ability is NOT agent ability.** A model that emits a legal action is
  only FORMAT-passing; action selection/replay/end-to-end decide capability.
- **Loss decrease is not capability.** Report train loss as a training signal
  only; capability claims come from layered eval and DEV/TEST.
- **Negative/infra-invalid trajectories** are never SFT-positive. They are used
  only for failure classification, preference pairs, recovery, verifier RL, or
  harness analysis.

## Canonical Action contract (recap)

See `docs/canonical-agent-contract.md`. The 9 semantic action types are:
`read_file`, `search_code`, `list_directory`, `run_command`, `edit_file`,
`write_file`, `finish`, `tool_error`, `environment_observation`.

## SFT training data policy

- Only `execution VALID`, `integrity COMPLETE`, `verifier PASSED` episodes
  become SFT-positive rows (negative/paired rows marked explicitly otherwise).
- Task-level split: TRAIN/DEV/TEST disjoint by task id; DEV/TEST never trained.
- Loss mask: assistant-only (no Pi-private token weighting by default).
- Workspace-relative paths only; host-path markers rejected by
  `assert_training_view_is_leak_free`.
- Dataset card + manifest + checksum + model/teacher/lineage recorded.

## Layer-1 current implementation status

Built, tested, and migrated on real 9-task teacher data:
- CanonicalAction + path/command normalization (unit-tested)
- Pi → CanonicalEpisode adapter (strict pairing, quarantine, first-action kept)
- Migration of 9 TRAIN teacher episodes → `canonical-teacher/v1` (0 quarantined,
  0 host-path leaks in clean actions)
- Multi-view exporters (generic, model-native Qwen/OpenAI/JSON, harness metrics)
- Canonical SFT dataset builder → `canonical-sft/v1` (241 decision examples,
  per-role stats, leak gate, dedup)
- Canonical generic JSON training package (`canonical-generic-json-v1`)
- Layered eval (format / action-selection / replay), unit-tested

## Layer-2 (evaluation/training) current findings

- A canonical generic JSON LoRA SFT run completed (train loss ~0.33-0.38),
  but offline generation shows the model has **NOT yet reliably learned the
  single-JSON-action output format** — it tends to produce degenerate/enumerated
  JSON. This is a FORMAT gap, correctly isolated by the layered eval, not a
  claim of agent capability.
- This is the same class of failure the v6/v7 archive documents: pushing a
  format boundary is necessary but not sufficient for a usable decision loop.
- Therefore as of this status, **no gate beyond training has passed**; the honest
  claim is that the canonical infrastructure is real and migrating real data,
  while model quality / harness integration remain the active next step.

## What still must be proven before claiming the loop is closed

1. A model (base or trained) that reliably emits a legal canonical action in
   the chosen format (FormatEval legal ratio high).
2. Action-selection accuracy vs certified references > base.
3. A working harness adapter that converts model output → executable action and
   back (TRAIN E2E diagnostic).
4. DEV (`pytest-dev__pytest-10081`) improvement over base under a fixed policy
   (Gate 3).
5. A harness-policy candidate that passes a fixed-model regression gate.
6. Full manifest/checksum/evidence for all runs; TEST untouched.
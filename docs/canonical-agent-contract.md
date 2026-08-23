# Canonical Harness-neutral Agent Action/Observation contract

> Status: **authoritative** — this is the project's cross-harness core, replacing
> harness- and model-specific tool representations in training data.

## Why this existed

v1–v7 (see `archive/experiments/v1-v7-pi-tool-protocol.md`) showed that training
a model to emit a Pi/Qwen `<tool_call>` grammar is harness-specific and is not a
general agent capability improvement. The fix is to decouple:

```text
Canonical Agent Semantics  →  Model-specific Training View  →  Thin Harness Adapter
```

The **canonical layer** (this contract) is harness-neutral and model-neutral.
Harness- and model-specific details live only in the thin adapters and the
exporter renderers.

## Canonical Action schema (Stage A)

Module: `src/contracts/canonical_action.py` (schema `canonical-action/v1`)

| Field | Meaning |
|---|---|
| `action_id` | stable, deterministic content-derived id |
| `action_type` | one of the 9 canonical semantics (below) |
| `canonical_tool_name` | semantic tool name used in training views |
| `arguments` | original harness arguments (evidence) |
| `normalized_arguments` | workspace-relative / neutralized arguments (for training) |
| `observation` | tool result (evidence + for observations) |
| `result_status` | SUCCEEDED / FAILED / ERROR / TIMEOUT / NONE |
| `source_tool_name` | Pi tool name (provenance only) |
| `source_harness` | `pi` |
| `parent_action_id` | link to a previous action |
| `action_timestamp` / `result_timestamp` | UTC instants |
| `evidence_event_ids` | original TraceEvent ids (raw evidence) |
| `environment_state_refs` | workspace snapshot refs |
| `lossy` / `lossy_reasons` | irreversible / outside-workspace normalization markers |

### Canonical action types

`read_file`, `search_code`, `list_directory`, `run_command`, `edit_file`,
`write_file`, `finish`, `tool_error`, `environment_observation`.

Pi's `read/bash/edit/write/grep/find/ls` appear **only** in the Pi adapter
(`src/capture/pi_canonical_adapter.py`) and are mapped:

| Pi tool | canonical |
|---|---|
| `read` | `read_file` |
| `grep` / `find` | `search_code` |
| `ls` / `glob` | `list_directory` / `search_code` |
| `bash` | `run_command` |
| `edit` | `edit_file` |
| `write` | `write_file` |

Unknown Pi tools are **fail closed** → represented as `tool_error` and marked
`lossy`, never silently re-labeled.

## Path and path-command normalization

- Absolute paths under the workspace root → workspace-relative (e.g.
  `/home/.../swebench/sympy-23824/sympy/core/a.py` → `sympy/core/a.py`).
- Paths outside the workspace → left unchanged but marked `lossy`.
- Shell commands: the workspace root token is rewritten to `$WORKSPACE`; a
  command still referencing other absolute host paths is marked `lossy`.
- Original values are always preserved in `arguments` / `observation` as
  evidence; training views must read `normalized_arguments` for a leakage-free
  representation.

## Canonical Episode (Stage B output)

Module: `src/capture/pi_canonical_adapter.py` (schema `canonical-episode/v1`)

A `CanonicalEpisode` is one episode's ordered, harness-neutral action stream,
carrying `episode_id`, `task_id`, `source_harness`, `model_id`,
`verifier_status`, and the ordered `actions`. It is the stable input to every
training-view exporter and every evaluation.

Guarantees implemented by the Pi adapter:

- Tool call/result pairs are strictly matched by `pi_tool_call_id`.
- The **first** assistant action is never dropped.
- No success is fabricated for an unpaired tool call (it becomes a
  `tool_error`/NONE action).
- Unmappable tools → `tool_error` + quarantine in the migration manifest.

## Migration of certified teacher data

`scripts/migrate_teacher_to_canonical.py` converts each finalized `episode.json`
into a `canonical-*` episode plus a migration manifest with before/after
checksums and quarantine. Server output under `canonical-teacher/v1/`.

Current real migration (9 TRAIN teacher episodes):

| task | actions | lossy | verifier |
|---|---|---|---|
| psf__requests-5414 | 27 | 0 | PASSED |
| pytest-dev__pytest-10051 | 37 | 11 | PASSED |
| sympy__sympy-23824 | 20 | 1 | PASSED |
| sympy__sympy-24213 | 31 | 7 | PASSED |
| psf__requests-6028 | 34 | 10 | PASSED |
| sympy__sympy-23950 | 46 | 9 | PASSED |
| psf__requests-2931 | 24 | 0 | PASSED |
| sympy__sympy-24539 | 20 | 1 | PASSED |
| sympy__sympy-23534 | 23 | 0 | PASSED |

Total 262 actions, 39 lossy (15%). All 9 verifier PASSED, 0 quarantined, 0
host-path leaks in clean actions.

The workspace root is taken from each run's authoritative `launch-plan.json`
and only falls back to inference (from bash evidence) when no launch plan
exists. `workspace_root` can also be passed explicitly.

## Not yet proven

- A second harness (non-Pi) adapter to demonstrate the contract generalizes.
- Multi-view exporter renderers consuming `CanonicalEpisode` (Stage C, next).
- Layered evaluation on top of this contract.
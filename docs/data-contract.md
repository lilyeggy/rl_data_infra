# Canonical Agent Execution Data Contract v2

> Status: design frozen for the new project direction.
> `rollout-record/v1` remains supported as an optional Training View.

## 1. Purpose

The core package sits between heterogeneous Agent Harnesses and multiple data consumers. It records observable execution facts, assembles them into comparable Episodes, and prevents unsupported inferences from becoming canonical data.

```text
Harness / Model Proxy / Sandbox / Verifier / Hook
                         ↓
                  TraceEvent stream
                         ↓
                  EpisodeAssembler
                         ↓
                    AgentEpisode
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       Analyze        Compare        Export
                                      ↓
                             RolloutRecord v1
```

The contract has no runtime dependency on Polar, Slime, Megatron, a model backend, or a specific Harness.

## 2. Design rules

1. **Facts before interpretation**: raw events are append-only; metrics and diagnoses are derived records.
2. **Observable only**: unavailable Harness internals are `NOT_OBSERVABLE`, never guessed from text.
3. **Explicit capabilities**: every Episode states which capture paths were active.
4. **Stable identity**: events, spans, Episodes, runs and artifacts have independent IDs.
5. **Lineage everywhere**: derived outputs include input checksums and producer/rule versions.
6. **Partial is valid**: an interrupted trace may be stored and diagnosed without being presented as complete.
7. **Comparable by proof**: Harness comparison requires compatible experiment manifests.
8. **Training is a view**: training-specific token/mask/reward fields do not define the universal Episode schema.

## 3. Versioned objects

| Object | Schema version | Responsibility |
|---|---|---|
| `TraceEvent` | `trace-event/v1` | One immutable observable execution fact |
| `AgentEpisode` | `agent-episode/v1` | One assembled task execution |
| `HarnessManifest` | `harness-manifest/v1` | Harness identity and configuration digest |
| `EnvironmentManifest` | `environment-manifest/v1` | Sandbox/runtime identity |
| `ExperimentManifest` | `experiment-manifest/v1` | Comparison control variables |
| `ArtifactRef` | `artifact-ref/v1` | Content-addressed external evidence |
| `Diagnosis` | `diagnosis/v1` | Versioned derived failure attribution |
| `EpisodeComparison` | `episode-comparison/v1` | Paired or aggregate Harness differences |
| `GateResult` | `gate-result/v1` | Versioned release decision and evidence |
| `RolloutRecord` | `rollout-record/v1` | Existing optional training projection |

Unknown producer fields belong in `attributes` or source-specific metadata. Unknown top-level canonical fields are rejected so schema drift cannot pass silently.

## 4. TraceEvent

### 4.1 Required envelope

```text
schema_version
event_id
run_id
episode_id
trace_id
span_id
parent_span_id             # nullable only for root span
sequence
timestamp
event_type
component
status
attempt
attributes
artifact_refs
```

Requirements:

- `event_id` is globally unique and supports idempotent ingestion;
- `sequence` is producer-assigned and monotonic within an Episode when possible;
- timestamp is timezone-aware UTC;
- `span_id`/`parent_span_id` model causal nesting, not merely display order;
- `attempt` distinguishes retries of the same logical action;
- attributes must be JSON-compatible and must not contain secrets;
- large stdout, patches, files or binary payloads use `ArtifactRef`.

### 4.2 Core event types

```text
MODEL_REQUEST
MODEL_RESPONSE
TOOL_CALL
TOOL_RESULT
SANDBOX_STARTED
SANDBOX_COMMAND
SANDBOX_FINISHED
VERIFICATION_STARTED
VERIFICATION_FINISHED
EPISODE_FINISHED
```

Optional Hook events:

```text
HARNESS_DECISION
CONTEXT_SELECTED
CONTEXT_COMPACTED
RETRY_SCHEDULED
TERMINATION_DECIDED
```

Producer-specific types may be preserved as namespaced values but cannot silently alter core semantics.

### 4.3 Status and component

Core component values:

```text
MODEL
HARNESS
TOOL
SANDBOX
MODEL_BACKEND
EVALUATOR
EXTERNAL_SERVICE
```

Core event status values:

```text
STARTED
SUCCEEDED
FAILED
ERROR
TIMEOUT
CANCELLED
UNKNOWN
```

`FAILED` means an operation completed with a valid negative result. `ERROR` and `TIMEOUT` mean the operation could not produce a trustworthy result. They must not be collapsed.

## 5. AgentEpisode

### 5.1 Identity and manifests

```text
schema_version
episode_id
run_id
task_id
attempt
harness_manifest
model_manifest
environment_manifest
evaluator_manifest
experiment_manifest_ref
```

### 5.2 Data

```text
capabilities
events
artifact_refs
started_at
ended_at
outcome
termination
integrity
source_lineage
```

`events` are deterministically ordered by assembler rules. The original ingestion order/checksum remains in lineage.

### 5.3 Outcome

Outcome separates task result from infrastructure validity:

```text
task_status: SUCCESS | FAILURE | UNKNOWN
execution_validity: VALID | INFRA_INVALID | UNKNOWN
verifier_status
score/reward optional
evidence_event_ids
```

An Episode may be `task_status=FAILURE` and `execution_validity=VALID`. A verifier crash should instead produce `task_status=UNKNOWN` and `execution_validity=INFRA_INVALID`.

### 5.4 Integrity

```text
state: COMPLETE | PARTIAL | CORRUPT
duplicate_event_count
sequence_gaps
orphan_span_ids
missing_expected_event_types
assembler_version
input_checksum
output_checksum
```

The assembler may deduplicate and order events, but it never rewrites raw input. A `PARTIAL` Episode remains queryable; consumers decide whether their operation permits it.

### 5.5 Multi-agent boundary (open design item)

`RolloutRecord v1` represents one linear trainable trace. It does not provide
producer-agnostic fields for sub-agent identity, parent/child agent ownership,
parallel branch structure, or cross-agent reward/credit aggregation.

`TraceEvent` and `AgentEpisode` improve the execution view by retaining
`episode_id`, `trace_id`, `span_id`, and `parent_span_id`. These fields can
preserve causal branches observed during a multi-agent run, but they do not by
themselves settle the training semantics of those branches.

Before claiming full multi-agent support, the project must validate real
sub-agent artifacts and decide:

```text
agent_id / parent_agent_id / agent role
one Episode versus nested child Episodes
branch and concurrency ordering
episode-, trace-, or agent-level reward ownership
cross-agent credit aggregation
multi-trace Episode → RolloutRecord export policy
```

Until then, source-specific agent metadata may be retained in event attributes
or lineage, but core consumers must not infer these relationships. The project
must report multi-agent training semantics as `NOT_OBSERVABLE` or
`INSUFFICIENT_EVIDENCE`, and `RolloutRecord v1` remains a single-trace Training
View rather than a complete multi-agent rollout model.

## 6. Capabilities

Episode-level capability values:

```text
MODEL_IO
MODEL_TOKEN_USAGE
MODEL_TOKEN_IDS
MODEL_LOGPROBS
TOOL_IO
SANDBOX_LIFECYCLE
SANDBOX_COMMAND_IO
FILE_ARTIFACTS
VERIFIER_EVIDENCE
HARNESS_DECISIONS
CONTEXT_SELECTION
CONTEXT_COMPACTION
RETRY_DECISIONS
TERMINATION_DECISIONS
```

A capability is present only if the active capture path can reliably provide it for the Episode. Batch/run capability is the intersection when a consumer requires the field across every Episode.

Consumers declare requirements:

```text
Trace viewer: no hard capability beyond core envelope
Tool recovery metric: TOOL_IO
Compaction diagnosis: CONTEXT_COMPACTION + HARNESS_DECISIONS
Verifier gate analysis: VERIFIER_EVIDENCE + TERMINATION_DECISIONS
Training exporter: its own token/mask/reward/policy requirements
```

Missing capability produces a structured `CAPABILITY_MISSING` or `INSUFFICIENT_EVIDENCE`, not a fabricated value.

## 7. Manifest contract

### 7.1 HarnessManifest

```text
name
version
revision
config_digest
policy_flags
hook_version optional
```

### 7.2 EnvironmentManifest

```text
runtime_type
image/revision
resource_limits
network_policy
workspace/task snapshot
```

### 7.3 ExperimentManifest

```text
experiment_id
task_set_revision
model/provider/revision
sampling_config
environment_revision
tool_schema_digest
evaluator_revision
seeds
timeouts
candidate_variable
```

Before paired comparison, all fields except `candidate_variable` must be compatible. Mismatches are listed as confounders.

## 8. ArtifactRef

```text
schema_version
artifact_id
kind
uri
media_type
sha256
size_bytes
producer_event_id
created_at
```

Examples include stdout/stderr, patch, final answer, file snapshot, verifier report and raw provider payload. Checksums use SHA-256 over exact bytes. An artifact reference without retrievable content is allowed only when marked missing in integrity metadata.

## 9. Diagnosis

Diagnosis is derived and never overwrites Episode facts:

```text
schema_version
diagnosis_id
episode_id
layer
reason_code
evidence_event_ids
evidence_artifact_ids
confidence
rule_version
explanation
created_at
input_episode_checksum
```

Allowed top-level layers:

```text
MODEL
HARNESS
SANDBOX
MODEL_BACKEND
EVALUATOR
EXTERNAL_SERVICE
UNKNOWN
```

Diagnoses may be multi-label. If a rule requires unavailable data, it returns `INSUFFICIENT_EVIDENCE` rather than a low-quality guess.

## 10. Comparison and Gate

### 10.1 EpisodeComparison

```text
comparison_id
control_run_id
candidate_run_id
manifest_compatibility
paired_task_results
aggregate_metrics
failure_slices
severe_regressions
sample_size
uncertainty
input_checksums
comparison_version
```

Infrastructure-invalid Episodes are reported separately and excluded from task-success denominators by default. The exclusion count is always visible.

### 10.2 GateResult

```text
gate_id
comparison_id
verdict: ACCEPT | REJECT | INSUFFICIENT_EVIDENCE
rules
thresholds
passed_rules
failed_rules
evidence_refs
gate_version
created_at
```

The same comparison and gate configuration must produce deterministic output.

## 11. Raw JSONL interchange

Each line stores one event envelope:

```json
{"schema_version":"trace-event/v1","event_id":"evt_001","episode_id":"ep_001","event_type":"MODEL_REQUEST"}
```

Full required fields are omitted above for readability. Import behavior:

- blank lines may be ignored;
- malformed lines are reported without discarding valid neighbors;
- duplicate `event_id` is detected;
- exact input-line checksum is retained;
- unknown schema versions are rejected or quarantined, never coerced silently.

## 12. Existing RolloutRecord v1 compatibility

`RolloutRecord` is now explicitly a Training View. It remains useful when an Episode contains trustworthy training fields.

```text
AgentEpisode
  → TrainingViewExporter capability check
  → token/action/reward/policy projection
  → RolloutRecord v1
  → optional TrainingReadyBatch / trainer adapter
```

Rules retained from v1:

- text is never retokenized and labelled as sampled token IDs;
- missing logprobs, masks, rewards or policy identity remain missing;
- Polar-specific fields remain inside `PolarSourceAdapter`;
- `INVALID_INFRASTRUCTURE` is not converted into model reward zero;
- policy/group requirements remain enforced for training consumers;
- checksums and source lineage remain stable.

The exporter is optional and does not determine whether the main execution-data project is complete.

## 13. Error semantics

| Code | Meaning |
|---|---|
| `CAPTURE_ERROR` | Capture path failed before a reliable event was produced |
| `CONTRACT_INVALID` | Canonical fields violate schema invariants |
| `CAPABILITY_MISSING` | Consumer prerequisite is not observable |
| `DUPLICATE_EVENT` | Event ID has already been ingested |
| `SEQUENCE_GAP` | Expected event order is incomplete |
| `ORPHAN_SPAN` | Parent span is unavailable |
| `MANIFEST_MISMATCH` | Runs are not directly comparable |
| `ARTIFACT_MISSING` | Referenced evidence cannot be retrieved |
| `INSUFFICIENT_EVIDENCE` | A diagnosis or decision cannot be supported |
| `SOURCE_WARNING` | Input is retained but needs attention |

These errors describe data reliability. They are not automatically task failures.

## 14. Test boundary

Core contract, assembly and analysis tests must run without installing Polar, Slime, Megatron or a live model server:

```bash
python3 -m unittest discover -s tests -v
```

Real Harness integration is covered by fixtures plus a separately reproducible end-to-end run.

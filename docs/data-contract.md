# Canonical Data Contract v1

## 1. Purpose

The core package sits between a rollout producer and a trainer. It records what a
producer actually emitted, exposes which training capabilities are present, and
rejects incompatible input before a Processor or Trainer Adapter consumes it.

The package has no Polar, Slime, Megatron, tokenizer, model, or network dependency.

```text
producer payload
→ SourceAdapter
→ AdapterResult
→ RolloutBatch
→ Processors
→ TrainingReadyBatch / ResampleRequest
```

## 2. Versioned objects

| Object | Schema version | Responsibility |
|---|---|---|
| `RolloutRecord` | `rollout-record/v1` | One canonical trajectory and its evidence |
| `RolloutBatch` | `rollout-batch/v1` | Duplicate-free Processor input |
| `TrainingReadyBatch` | `training-ready-batch/v1` | One policy-consistent trainable group |
| `ResampleRequest` | `resample-request/v1` | A request for missing group members |
| JSONL envelope | `rollout-jsonl/v1` | Public offline interchange format |

Unknown producer fields belong in `opaque_metadata`. Adding an unknown top-level
canonical field is a contract error so schema drift cannot pass silently.

## 3. RolloutRecord

### Identity

Always required for ingestion:

- `trajectory_id`
- `task_id`
- `source_type`
- `source_record_id`

Representable but required before training:

- `group_id`
- `policy_version`

An Adapter may ingest a source that lacks `group_id` or `policy_version`, but the
corresponding capability remains absent and a training precondition will reject it.

### Training payload

- `token_ids`: native numeric IDs emitted by the source trace;
- `prompt_token_count`: boundary between prompt and response inside `token_ids`;
- exactly zero or one of `action_mask` and `loss_mask`;
- `old_logprobs`: source values for either all tokens or masked trainable tokens;
- `reward`: a finite source reward.

Rules:

- text is never retokenized and labelled as sampled token IDs;
- missing log probabilities, masks, rewards, or policy identities remain `None`;
- masks contain only `0` and `1` and have the same length as `token_ids`;
- `old_logprobs` requires token IDs and may align with all tokens, the response
  segment, or masked trainable tokens;
- NaN and infinity are rejected.

### Execution status

`RolloutStatus`, `ComponentStatus`, and `VerifierStatus` are separate because a
task failure is different from an infrastructure failure.

For example, `verifier_status=failed` means the verifier ran and rejected the
answer. `verifier_status=error` or `timeout` means the outcome is not a valid task
failure and should later be classified as infrastructure-invalid.

### Evidence and lineage

- `verifier_evidence_ref`
- `source_payload_ref`
- `source_payload_sha256`
- model and tokenizer identity
- offset-aware start/end timestamps
- frozen JSON `tool_events`
- frozen JSON `opaque_metadata`

Checksums use UTF-8 canonical JSON with sorted keys, compact separators, and no
NaN. They are lineage identifiers, not signatures.

## 4. Capabilities

The v1 capability enum is:

```text
TOKEN_IDS
ACTION_MASK
OLD_LOGPROBS
REWARD
GROUP_ID
POLICY_VERSION
VERIFIER_EVIDENCE
TOOL_EVENTS
```

`ACTION_MASK` means that either canonical `action_mask` or `loss_mask` is present.
A record's capabilities are derived from its actual fields. A batch exposes the
intersection across all records, so a consumer may rely on every declared batch
capability.

The first training gate requires:

```text
TOKEN_IDS + ACTION_MASK + REWARD + GROUP_ID + POLICY_VERSION
```

`OLD_LOGPROBS` is optional in the core v1 contract. A Trainer Adapter must add it
to its own required set if its configured loss needs it.

## 5. AdapterResult and errors

Every Source Adapter returns:

```text
records
capabilities guaranteed across records
warnings
errors
```

Stable error semantics:

| Code | Meaning |
|---|---|
| `ADAPTER_ERROR` | Source bytes or structure cannot be converted |
| `CAPABILITY_MISSING` | A configured consumer prerequisite is absent |
| `CONTRACT_INVALID` | Canonical fields contradict contract invariants |
| `SOURCE_WARNING` | Input can be retained but needs attention |

`CAPABILITY_MISSING` is not a trajectory task failure. An `AdapterResult` with
errors cannot become a `RolloutBatch`.

## 6. JSONL interchange

Each non-empty line is one envelope:

```json
{"schema_version":"rollout-jsonl/v1","record":{"schema_version":"rollout-record/v1"}}
```

Export is deterministic and newline-terminated. Import replaces only the current
source envelope:

```text
source_type=jsonl
source_record_id=line:<n>
source_payload_ref=<path>#line:<n> when reading a file
source_payload_sha256=SHA256(exact UTF-8 line without newline)
```

All non-envelope record semantics remain identical. Malformed lines are reported
without silently discarding valid neighbouring lines.

## 7. Polar fixture mapping

`PolarSourceAdapter` reads reviewed fixture JSON and never imports the upstream
Polar package. For each pinned `prefix_merging` trace it maps:

```text
token_ids = prompt_ids + response_ids
prompt_token_count = len(prompt_ids)
loss_mask = zeros(len(prompt_ids)) + source loss_mask
old_logprobs = source response_logprobs
reward = source trace.reward
```

The `response_logprobs → old_logprobs` mapping is based on the pinned Polar
builder contract: these are sampled-policy log probabilities aligned with the
response and its loss mask. The original JSON path remains in opaque lineage.
The Adapter does not create policy identity; the current Day 2 fixture therefore
still lacks `POLICY_VERSION` and cannot become a `TrainingReadyBatch`.

An infrastructure failure with no trace becomes a canonical record with no
tokens or reward. This lets `FailureClassifier` observe and exclude it without
mistaking it for a model reward of zero.

## 8. TrainingReadyBatch

Every record must share exactly:

```text
task_id + group_id + policy_version
```

Every record must also provide the configured required capabilities. Duplicate
trajectory IDs are rejected. `processing_versions` and the source batch checksum
make the produced training group auditable.

The contract enforces structural readiness. Day 5 Processors remain responsible
for validity classification, reward signal, deterministic group sizing, and
resampling decisions.

## 9. ResampleRequest

A request contains task/group/policy identity, a positive missing count, a stable
reason, JSON sampling constraints, and an optional source hint. It never executes
rollout itself. That keeps the core pipeline producer-agnostic.

## 10. Import and test boundary

Core tests run with:

```bash
python3 -m unittest discover -s tests -v
```

They do not install or import Polar, Slime, or Megatron.

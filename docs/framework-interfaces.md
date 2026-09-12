# Harness Data Plane: framework interfaces

Status: wrap-up interface layer, added after the RL and SFT validation work.
These interfaces do not rewrite any trainer or harness. They name the seams that
already existed in the project, so the project can be described and extended as
a framework.

## Positioning

```text
Model + Harness + Tasks + Verifier
        |
        v
  Harness Data Plane
  - identity / evidence
  - verifier verdict
  - certification and cleaning
  - consumer views
        |
        v
SFT / Preference / ON_POLICY_RL / Evaluation
```

The project owns data production, evidence, certification and cleaning.
It does not own agent orchestration or trainer internals.
Existing consumers:

| Consumer | Train owner |
|---|---|
| SFT messages view | HF TRL / LLaMA-Factory style trainers |
| ON_POLICY_RL admission | verl RayPPOTrainer |
| Preference / Evaluation | consumer compilers |

## Interfaces

Defined in `src/framework/interfaces.py`:

- `TaskRef` - one task with statement and metadata
- `TaskSource.tasks()` - pluggable task provider
- `ModelBackend.complete()` - pluggable model endpoint
- `ModelRequest` / `ModelResponse` - messages, tools, sampling, native token ids, logprobs
- `Verdict` - `PASSED` / `FAILED` / infrastructure status, score, detail
- `HarnessAdapter.run()` - real execution, returns `ExecutionResult`
- `Verifier.verify()` - verifier over workspace and evidence
- `ConsumerCompiler.compile()` - turns an `ExecutionResult` into a consumer view
- `RunSpec` - one run: task, harness, model, verifier, consumer, output
- `DataPlane` - thin facade that runs one spec and writes `run-manifest.json`

## Mapping to the existing project

| Framework interface | Existing implementation |
|---|---|
| `ModelBackend` | Model Proxy HTTP forwarder / OpenAI-compatible endpoint |
| `HarnessAdapter` | `PiHostExecutionOrchestrator`, Pi / Polar producers |
| `Verifier` | `verify_mbpp_src.py`, `verify_apps.py`, project verifiers |
| evidence objects | `ExecutionIdentity`, `AgentEpisode`, `ExecutionBundle`, `ProducerArtifact` |
| certification | `certify_for`, consumer profiles |
| cleaning | fail-closed checks, dedup, split, path normalization |
| SFT consumer | `export_sft_messages_jsonl.py` then TRL `SFTTrainer` |
| RL consumer | `AdmittedVerlSequence`, `CertifiedVerlAgentLoopManager`, verl |

## Example usage

```python
from pathlib import Path

from src.framework import DataPlane, RunSpec, TaskRef

spec = RunSpec(
    run_id="demo-run",
    task=TaskRef(task_id="Mbpp/2", statement="implement the function"),
    harness=my_harness,
    model=my_model,
    verifier=my_verifier,
    consumer="SFT",
    output_dir=Path("artifacts/demo-run"),
)

result = DataPlane([my_sft_compiler]).run(spec)
view = DataPlane([my_sft_compiler]).compile(result, "SFT", Path("artifacts/demo-sft"))
```

`DataPlane.run` writes `run-manifest.json` with the selected harness, model,
verifier, consumer, verdict and checksum. The actual evidence chain is still
owned by the existing execution/orchestration modules.

## What is not done

- The interfaces are intentionally thin and are not yet wired as the default
  entry point of the old scripts.
- The existing APPS SFT builder still uses its legacy prompt/completion
  serialization. The standard `messages` / `tools` exporter was validated
  separately with TRL.
- The SFT framework route has a real 14B LoRA one-step smoke, not a full
  benchmark run.
- Trainer-neutral consumer compilers for Preference and Evaluation are not yet
  implemented as classes here.

## Wrap-up

With these interfaces the project can honestly be described as a harness and
trainer-neutral data plane:

- plug in a model backend;
- plug in a harness adapter;
- plug in tasks and a verifier;
- get identity-bound, verifier-backed evidence;
- compile SFT / RL / Preference / Evaluation consumer views;
- hand SFT to TRL and RL to verl without maintaining a trainer.

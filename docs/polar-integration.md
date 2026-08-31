# Polar integration boundary

## What is current

The live path uses Polar's documented stable HTTP service:

- `POST /rollout/task/submit` submits one task with `num_samples=N`;
- `GET /rollout/task/{task_id}` polls task status and terminal results;
- Polar expands the task into N sessions and returns one `SessionResult` per
  session;
- each trajectory can contain token-aligned `response_ids`, `loss_mask`,
  `response_logprobs`, messages and evaluator-attached reward.

Primary upstream references:

- [Polar rollout service](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/stable/src/polar/rollout/README.md)
- [Polar trajectory model](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/stable/src/polar/trajectory/README.md)
- [Official Polar–Slime bridge](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/stable/src/slime_bridge/README.md)

## Layers in this repository

`src/integrations/polar/client.py` is a dependency-free transport boundary. It
does not import Polar and does not infer result fields.

`src/integrations/polar/adapter.py` pins the documented stable result semantics.
It allocates one `ExecutionIdentity` per session and rejects:

- a non-terminal task;
- task/session identity mismatch;
- missing or duplicated sessions;
- missing data-plane identity metadata;
- token, mask or logprob type/length violations;
- non-finite logprobs or rewards.

`src/producers/polar_live.py` injects the reserved `agent_data_plane` metadata
before submission and collects terminal artifacts. It is explicitly a batch
producer because Polar session ids do not exist before task submission.

`src/sources/polar_fixture.py` is only an offline historical fixture importer.
It is not the live Polar producer and cannot prove current on-policy identity.

## Evidence rules

- token/logprob/mask capabilities are claimed only from actual aligned arrays;
- evaluator reward becomes verifier evidence only for a completed session with
  an immutable evaluator fingerprint;
- `ERROR` and `TIMEOUT` never become verifier-backed reward-zero failures;
- policy identity is a SHA-256 fingerprint, not a mutable model alias;
- a Polar task id is a group/request id, never an episode id.

`src/assembly/execution_bundle_assembler.py` joins producer artifacts with the
same session's canonical Episode and verifier report. RL certification then
points to that immutable bundle and exact policy artifact. This path is covered
by offline end-to-end fixtures; the same join still requires validation against
a real live Polar/Harness session.

## Limited-resource deployment rule

Polar is an optional batch producer behind the Local Launcher-first data plane.
This project does not copy an upstream multi-GPU topology. The supported
single-rented-GPU development sequence is time-sliced:

```text
serve policy-vN + Polar rollout-only
→ stop/unload serving
→ certify and freeze dataset manifest
→ Slime train-only/replay
→ checkpoint policy-vN+1
→ reload serving
→ fixed holdout evaluation
```

The upstream Polar–Slime bridge remains the preferred training integration.
This repository should add admission/certification around it, not fork Slime's
trainer, Megatron backend or weight-sync implementation.

`src/orchestration/single_gpu.py` makes this phase order machine-checkable. It
does not execute shell commands; a deployment driver must attach the required
rollout bundle, dataset, checkpoint and evaluation checksums at each transition.
`src/orchestration/workflow.py` adds atomic state persistence, file locking,
checksum compare-and-swap and an argv-only dry-run plan for safe resume.

`src/integrations/slime/admission.py` is the final local trust boundary. It
revalidates the dataset member, certification decision, evidence bundle and
policy artifact, checks aligned trainable arrays and finite rewards, prevents
mixed-policy batches, and measures group size by unique session rather than
trace count. It intentionally does not define or import Slime's `Sample` class.

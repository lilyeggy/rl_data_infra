# Local Launcher-first execution flow

Local Docker Launcher is the default way to start a Harness. Polar is an
optional batch producer and Slime is an optional training consumer.

```text
LocalExecutionSpec (secret-free)
        │
        └─ execute-local
              ├─ freeze ExecutionRunManifest + launch plan
              ├─ Model Proxy on host ──> controlled model endpoint
              ├─ Harness event ingress (execution bearer token)
              ├─ Local Docker Launcher ──> Harness in task sandbox
              ├─ snapshot diff/patch ──> content-addressed objects
              ├─ isolated verifier container ──> verifier report
              └─ deterministic finalize + cleanup
                         │
              model/tool/verifier/hook facts
                         │
          raw-events + model-evidence + producer-artifact
                         │
                    finalize-local
                         │
              AgentEpisode + ExecutionBundle
```

The model may run on a rented GPU. “Local/self-hosted model” means the project
controls the weights, tokenizer, sampling configuration and serving revision;
it does not mean the model must run on the Mac. An OpenAI-compatible API is a
transport protocol and is supported by the proxy.

## Evidence levels

- Any backend can provide redacted request/response observability.
- Text-only external APIs do not receive token or RL capability.
- A controlled backend must return `agent_data_plane_evidence` containing an
  immutable `backend_model_revision`, native `response_token_ids`, and aligned
  `response_logprobs` for an RL-usable model call.
- Text is never re-tokenized by the proxy and relabelled as sampled tokens.

## Isolation and authentication

The launcher exposes two explicit sandbox profiles instead of silently changing
the isolation level:

- `FAST` + `runc` is the default single-tenant rollout-pool profile. It reuses a
  prebuilt Harness image while giving every Episode its own container and
  workspace. The launcher pins `linux/amd64`, the image digest, CPU, memory,
  PID and tmpfs limits, and enables read-only root, `cap-drop=ALL`, and
  `no-new-privileges`.
- `SECURE` + `runsc` is the multi-tenant profile. The launcher checks Docker's
  registered runtimes before starting any Harness and fails the execution as
  infrastructure-invalid when `runsc` is unavailable. It never falls back to
  `runc`, so a requested security boundary cannot be weakened unnoticed.

The selected profile and concrete Docker runtime are written into launch and
producer evidence. The current A6000 host only registers `runc` and
`io.containerd.runc.v2`; therefore live benchmarks use `FAST/runc` and
`SECURE/runsc` remains unavailable until gVisor is installed on that host.

When a Harness needs the Model Proxy, network access must be explicitly changed
to `BRIDGE_UNRESTRICTED`. The proxy requires a per-execution bearer token. The
token is injected only through process/container environment and is excluded
from launch-plan serialization and evidence. Bridge mode is broad egress, not
an endpoint allowlist; it is acceptable for the small local validation phase,
but a multi-tenant deployment still needs an egress gateway or equivalent
network policy.

## Preferred commands

Copy `configs/local-execution.example.json`, replace every placeholder and use
an absolute existing workspace path. The spec contains no access token or
upstream credential; `--upstream-auth-env` names the host environment variable
whose value is forwarded only in memory.

```bash
python3 -m src.cli prepare-local \
  --spec configs/my-run.json \
  --output-dir artifacts/prepared-run

python3 -m src.cli execute-local \
  --spec configs/my-run.json \
  --output-dir artifacts/run-001 \
  --sandbox-profile FAST \
  --sandbox-runtime runc \
  --upstream-auth-env MODEL_SERVER_AUTHORIZATION
```

`execute-local` requires a new or empty output directory. It owns the complete
lifecycle and writes the manifest, redacted/content-addressed raw artifacts,
events, model evidence, producer artifact, Episode and ExecutionBundle. A
verifier exit code of 1 is a valid failed task verdict; Docker startup failure
or verifier timeout is infrastructure-invalid and is not converted to reward
zero.

## Lower-level commands

```bash
python3 -m src.cli plan-local ... -- <harness argv>
python3 -m src.cli run-local ... --artifact-output producer-artifact.json \
  -- <harness argv>
python3 -m src.cli serve-model-proxy ...
python3 -m src.cli finalize-local \
  --manifest execution-run-manifest.json \
  --producer-artifact producer-artifact.json \
  --events raw-events.jsonl \
  --model-evidence model-evidence.jsonl \
  --artifacts artifact-refs.json \
  --output-dir finalized
```

The lower-level commands are debugging surfaces. Production-like local runs
should use `execute-local`, which makes terminal/verifier facts host-owned. A
proxy-only trace remains partial and is not promoted to training.

## Harness adapter contract

The container receives `OPENAI_BASE_URL`/`OPENAI_API_KEY` for model calls and
`AGENT_TRACE_URL`/`AGENT_TRACE_API_KEY` for Harness-native facts. Python
Harnesses can use `HarnessTraceClient`; other languages can POST the same
strict descriptor JSON. The Harness may report tool I/O, sandbox commands,
decisions, context selection/compaction, retries and termination decisions. It
cannot self-assign identity/sequence/timestamp/event IDs, and cannot emit model
responses, verifier verdicts, sandbox lifecycle or the terminal Episode fact.

The Docker-to-proxy smoke uses a fake controlled model and no GPU:

```bash
python3 -m scripts.local_model_proxy_smoke \
  --image alpine \
  --image-digest <sha256> \
  --output-dir artifacts/local-model-proxy-smoke
```

The full no-GPU transaction smoke is:

```bash
python3 -m scripts.local_execution_smoke \
  --image alpine \
  --image-digest <sha256> \
  --output-dir artifacts/local-execution-smoke
```

This has also been validated end to end on the A6000 with a real Pi Harness,
Qwen2.5-Coder-14B served by vLLM, per-Episode Docker sandboxes, host-owned
verification, model evidence, finalized Episode and checksum-bound
ExecutionBundle. See `artifacts/vllm-migration-20260830/fair-comparison-report.md`
for the controlled Local Data Plane versus Polar comparison.

"""Pi agent loop for verl's trainer (framework-owned rollout).

verl owns the training loop, the advantage/update math, the checkpoint and the
weight sync. This module owns only the *rollout* shape the framework asks for:
an ``AgentLoopBase`` whose ``run`` returns one certified Pi episode as an
``AgentLoopOutput``.

The token contract is the important part. verl's
``AgentLoopWorker._agent_loop_postprocess`` never re-tokenizes: it pads
``response_ids`` verbatim and trusts ``response_mask`` (1 = LLM-generated, 0 =
tool observation) and ``response_logprobs``. Those are exactly our
``AssembledSequence`` semantics, so an episode certified by the existing data
plane can be handed to the trainer without loss.

Traffic path (closeout §3.1): Pi -> our evidence-capturing model proxy ->
verl's vLLM OpenAI endpoint. The proxy is started by
``PiHostExecutionOrchestrator``; this module only tells it where the engine is.

Imports verl, so it is deliberately NOT re-exported from
``src/integrations/verl/__init__.py`` (that package must import without verl).
verl loads this class by fully-qualified name from the agent-loop YAML.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from src.contracts._json import sha256_json
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EvaluatorManifest, HarnessManifest
from src.errors import ContractValidationError
from src.integrations.verl.admission import AdmittedVerlSequence
from src.integrations.verl.bridge import BridgeCallRecord, build_per_call_segments
from src.integrations.verl.sequence import assemble_episode_sequence
from src.integrations.verl.native_transport import NativeTokenTransport
from src.orchestration.pi_host_execution import (
    PiHostExecutionOrchestrator,
    PiHostExecutionSpec,
)

PI_LOOP_VERSION = "verl-pi-agent-loop/v1"
FIXED_TOOLS = ("read", "bash", "write", "edit", "ls")

logger = logging.getLogger(__name__)


def _served_model_id(engine_base: str) -> str:
    """Ask the framework's engine which model id it answers to.

    vLLM registers the policy under its own name -- verl passes the model path
    through unless a served_model_name is configured -- and replies to any
    other name with 404 "The model ... does not exist". Stage E's own server
    accepted an arbitrary name, so this cannot be carried over as a constant;
    it is discovered from the engine instead of assumed.
    """
    url = f"{engine_base}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except OSError as exc:
        raise ContractValidationError(f"engine at {engine_base} is not answering: {exc}") from exc
    entries = payload.get("data")
    if not isinstance(entries, list) or not entries:
        raise ContractValidationError(f"engine at {engine_base} serves no model")
    model_id = entries[0].get("id")
    if not model_id:
        raise ContractValidationError(
            f"engine at {engine_base} returned a model entry without an id"
        )
    return str(model_id)


def episode_workspace(output_dir: Any, episode_id: str) -> Path:
    """The task workspace, which must live outside the episode directories.

    Two constraints, both learned the hard way:

    * ``PiHostExecutionOrchestrator.run`` creates the episode root itself and
      fails closed when it already exists -- it owns raw-events.jsonl,
      model-evidence.jsonl, summary.json and objects/. Creating anything *under*
      the root beforehand creates the root as a side effect of ``parents=True``,
      and then every episode dies on the orchestrator's mkdir. The workspace
      also has to exist before the run, because the orchestrator snapshots it
      and uses it as the subprocess cwd.
    * The batch gate enumerates an attempt directory and treats every
      subdirectory as one episode, so a workspace placed beside the episode
      roots would be read as an episode missing its evidence.

    Hence a sibling tree keyed by attempt and episode, which no scan walks.
    """
    attempt_root = Path(output_dir).parent
    return attempt_root.parent / "workspaces" / f"{attempt_root.name}--{episode_id}"


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars (the batch columns are numpy arrays) for JSON."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _record_dispatch(output_dir: Path, episode_id: str, kwargs: dict[str, Any]) -> None:
    """Append one line per episode dispatch, next to the batch it came from.

    The episode writer fails closed on an existing directory, so any duplicate
    (episode_id, output_dir) within a generation is a hard error with no clue
    about who produced the second copy. This trace records the executing
    process and the identity the framework handed each task, which is what
    distinguishes a repeated dispatch from a mis-stamped batch. It is written
    with O_APPEND so concurrent workers cannot clobber each other, and it is
    never allowed to break the rollout.
    """
    root = output_dir.parent.parent
    try:
        root.mkdir(parents=True, exist_ok=True)
        record = {
            "pid": os.getpid(),
            "episode_id": episode_id,
            "output_dir": str(output_dir),
            "output_dir_existed": output_dir.exists(),
            "attempt": _jsonable(kwargs.get("pi_attempt")),
            "sample_token": _jsonable(kwargs.get("pi_sample_token")),
            "batch_position": _jsonable(kwargs.get("pi_batch_position")),
            "index": _jsonable(kwargs.get("index")),
        }
        with open(root / "dispatch-trace.jsonl", "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception:
        # Diagnostics must never be the reason a rollout fails.
        pass


def _engine_base_url(server_manager: Any) -> str:
    """Resolve the framework vLLM OpenAI base URL (``http://host:port``).

    ``PiServerManager`` exposes ``http_addresses``; the stock manager keeps the
    same strings as the keys of ``_server_id_to_handle``. With one replica
    (TP=2, DP=1) there is exactly one engine, so the choice is unambiguous.
    """
    addresses = getattr(server_manager, "http_addresses", None)
    if not addresses:
        addresses = list(getattr(server_manager, "_server_id_to_handle", {}) or {})
    if not addresses:
        raise ContractValidationError(
            "no vLLM server address is reachable from the agent loop; "
            "the rollout engine must be launched in hybrid mode"
        )
    return f"http://{addresses[0]}"


class PiAgentLoop(AgentLoopBase):
    """Run one real Pi episode and return it as a certified verl output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Our settings ride in the agent-loop YAML entry and arrive via **kwargs;
        # pull them out before the base class consumes the standard arguments.
        self.pi_binary = str(kwargs.pop("pi_binary", ""))
        self.episode_root = str(kwargs.pop("episode_root", ""))
        self.run_id = str(kwargs.pop("run_id", ""))
        self.policy_fingerprint = str(kwargs.pop("policy_fingerprint", ""))
        self.tool_schema_checksum = str(kwargs.pop("tool_schema_checksum", ""))
        self.verifier_script = str(kwargs.pop("verifier_script", ""))
        self.tools = tuple(kwargs.pop("tools", FIXED_TOOLS))
        self.episode_timeout_seconds = float(kwargs.pop("episode_timeout_seconds", 480.0))
        self.max_tokens_per_generation = int(kwargs.pop("max_tokens_per_generation", 1024))
        self.max_model_requests = int(kwargs.pop("max_model_requests_per_episode", 8))
        self.sampling_temperature = float(kwargs.pop("sampling_temperature", 1.0))
        self.sampling_top_p = float(kwargs.pop("sampling_top_p", 1.0))
        super().__init__(*args, **kwargs)
        self._validate_settings()

    def _validate_settings(self) -> None:
        if not self.pi_binary:
            raise ContractValidationError("pi_agent requires 'pi_binary'")
        if not self.episode_root:
            raise ContractValidationError("pi_agent requires 'episode_root'")
        if not self.run_id:
            raise ContractValidationError("pi_agent requires 'run_id'")
        if not self.policy_fingerprint:
            raise ContractValidationError("pi_agent requires 'policy_fingerprint'")
        if not self.verifier_script:
            raise ContractValidationError("pi_agent requires 'verifier_script'")
        if sorted(self.tools) != sorted(FIXED_TOOLS):
            raise ContractValidationError(
                "pi_agent tool set is fixed to read/bash/write/edit/ls"
            )

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info") or {}
        if not isinstance(extra_info, dict):
            raise ContractValidationError("dataset extra_info must be an object")
        if not kwargs.get("uid"):
            raise ContractValidationError("native verl uid is required for GRPO grouping")
        extra_info = dict(extra_info, group_id=str(kwargs["uid"]))
        self.policy_fingerprint = str(kwargs.get("pi_policy_fingerprint", self.policy_fingerprint))
        index = int(kwargs.get("index", 0))
        # The manager stamps the attempt number (for resampling) and a unique
        # per-sample token into the batch; `rollout.n` duplicates make the rows
        # of one group otherwise indistinguishable from inside the loop.
        attempt = int(kwargs.get("pi_attempt", 0))
        sample_token = str(kwargs.get("pi_sample_token", f"a{attempt}s{index}"))
        episode_id = f"{self.run_id}-{extra_info.get('task_slug', 'task')}-{sample_token}"
        # The manager stamps the output root for this specific generation call,
        # because it calls generate_sequences more than once per run (validation,
        # then every step) with the same tasks and the same sample tokens.
        call_root = kwargs.get("pi_episode_root")
        output_dir = (
            Path(str(call_root)) / f"attempt-{attempt}" / episode_id
            if call_root
            else Path(self.episode_root) / f"attempt-{attempt}" / episode_id
        )
        engine_base = _engine_base_url(self.server_manager)
        # A protocol label only: actual generation uses the framework's token
        # RPC, which selects its synchronized LoRA, not /v1/models[0].
        model_id = "verl-native-policy"
        from verl.experimental.agent_loop.tool_parser import ToolParser
        rollout = self.config.actor_rollout_ref.rollout
        transport = NativeTokenTransport(
            loop=asyncio.get_running_loop(), server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            parser=ToolParser.get_tool_parser("hermes", self.tokenizer),
            episode_id=episode_id, policy_revision=self.policy_fingerprint,
            sampling_params=sampling_params,
            max_prompt=int(rollout.prompt_length), max_response=int(rollout.response_length),
            max_tokens=self.max_tokens_per_generation, max_requests=self.max_model_requests,
            timeout=self.episode_timeout_seconds,
            expected_engine_step=kwargs.get("pi_expected_engine_step"),
            expected_tool_schema=self.tool_schema_checksum,
            # Siblings of the attempt directory, never inside it: the batch gate
            # enumerates episode directories, and an unexpected file there has
            # broken a run before.
            step_note_path=(
                Path(str(call_root)) / "engine-notes" / f"{episode_id}.step.jsonl"
                if call_root
                else None
            ),
            turn_note_path=(
                Path(str(call_root)) / "engine-notes" / f"{episode_id}.turns.jsonl"
                if call_root
                else None
            ),
        )
        started = time.monotonic()
        _record_dispatch(output_dir, episode_id, kwargs)

        summary, sequence, policy_artifact_checksum = await asyncio.to_thread(
            self._execute_episode,
            extra_info=extra_info,
            index=index,
            episode_id=episode_id,
            output_dir=output_dir,
            engine_base=engine_base,
            model_id=model_id,
            transport=transport,
        )
        if transport.error:
            raise ContractValidationError(transport.error)
        # A turn that carried no tool call is a sample, not a fault: the episode
        # still has an action (the policy's own tokens), a task outcome and a
        # reward, which is all PPO-style training needs. This used to raise
        # ("real tool observation and subsequent model call required"), and the
        # measured cost was severe -- on smoke23 the ON_POLICY_RL profile
        # certified the no-action episode ELIGIBLE with score 0.0, while the
        # raise aborted the whole generate_sequences gather and threw away the
        # two episodes of the same batch that were already issuing their second
        # model request. The gate was stricter than the certification contract
        # and amplified one weak sample into total loss. A batch of such episodes
        # has no reward variance and is still rejected by the batch gate, which
        # is where "no learning signal" belongs. `num_tool_rounds` keeps the fact
        # visible in the evidence.
        elapsed = time.monotonic() - started

        verifier_status = summary["verifier_status"]
        if verifier_status == "PASSED":
            reward = 1.0
        elif verifier_status == "FAILED":
            reward = 0.0
        else:
            # Infrastructure failures are not a valid zero reward; they must
            # never reach the trainer as a training signal.
            raise ContractValidationError(
                f"episode {episode_id} has no valid verifier verdict "
                f"({verifier_status}); refusing to invent a reward"
            )

        admitted = AdmittedVerlSequence(
            episode_id=episode_id,
            group_id=str(extra_info.get("group_id", self.run_id)),
            policy_fingerprint=self.policy_fingerprint,
            prompt_ids=tuple(sequence.prompt_ids),
            response_ids=tuple(sequence.response_ids),
            response_mask=tuple(sequence.response_mask),
            response_logprobs=tuple(sequence.response_logprobs),
            reward=reward,
            num_model_calls=sequence.num_model_calls,
            num_tool_rounds=sequence.num_tool_rounds,
            member_id=episode_id,
            execution_bundle_checksum=summary["execution_bundle_checksum"],
            policy_artifact_checksum=policy_artifact_checksum,
        )
        # The manager certifies the whole batch before the trainer may consume
        # it, so each episode leaves its admitted sequence on disk next to the
        # rest of its evidence.
        (output_dir / "admitted-sequence.json").write_text(
            json.dumps(admitted.to_dict(), indent=2) + "\n"
        )
        metrics = AgentLoopMetrics(
            generate_sequences=float(elapsed),
            tool_calls=float(sequence.num_tool_rounds),
            num_preempted=-1,
        )
        return AgentLoopOutput(
            prompt_ids=list(admitted.prompt_ids),
            response_ids=list(admitted.response_ids),
            response_mask=list(admitted.response_mask),
            response_logprobs=list(admitted.response_logprobs),
            reward_score=reward,
            num_turns=sequence.num_model_calls,
            metrics=metrics,
            extra_fields={
                "episode_id": episode_id,
                "group_id": admitted.group_id,
                "policy_fingerprint": self.policy_fingerprint,
                "policy_artifact_checksum": policy_artifact_checksum,
                "execution_bundle_checksum": summary["execution_bundle_checksum"],
                "num_model_calls": sequence.num_model_calls,
                "num_tool_rounds": sequence.num_tool_rounds,
                "pi_loop_version": PI_LOOP_VERSION,
                "verifier_status": verifier_status,
                "min_global_steps": transport.global_steps,
                "max_global_steps": transport.global_steps,
                # How much of this episode was cut off by a budget rather than
                # ended by the policy. Truncated turns are trained on -- that is
                # what verl's own agent loop does -- so the count is what lets a
                # reader see how much of the batch was cut off.
                "truncated_turns": transport.truncated_turns,
                "num_model_requests": transport.calls,
                "unconfirmed_engine_steps": transport.unconfirmed_steps,
                "engine_closeout": _read_closeout(output_dir),
            },
        )

    def _execute_episode(
        self,
        *,
        extra_info: dict[str, Any],
        index: int,
        episode_id: str,
        output_dir: Path,
        engine_base: str,
        model_id: str,
        transport: NativeTokenTransport,
    ) -> tuple[dict[str, Any], Any, str]:
        """Blocking body: run Pi through our orchestrator, then assemble.

        Runs in a worker thread because the orchestrator owns blocking
        subprocess I/O; ``AgentLoopBase.run`` is async only because the stock
        loops await the engine over Ray.
        """
        task_id = str(extra_info.get("task_id", ""))
        if not task_id:
            raise ContractValidationError("dataset extra_info lacks task_id")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        workspace = episode_workspace(output_dir, episode_id)
        workspace.mkdir(parents=True, exist_ok=False)
        (workspace / "solution.py").write_text('"""stub"""\n')
        task_prompt_path = workspace / "task-prompt.txt"
        task_prompt_path.write_text(str(extra_info.get("task_prompt", "")))
        group_id = str(extra_info.get("group_id", self.run_id))

        identity = ExecutionIdentity(
            run_id=self.run_id,
            task_id=task_id,
            episode_id=episode_id,
            attempt_id=index + 1,
            producer_id="pi-real",
            producer_version="0.84.2",
            group_id=group_id,
            policy_fingerprint=self.policy_fingerprint,
            sampling_fingerprint=sha256_json(
                {"temperature": self.sampling_temperature, "top_p": self.sampling_top_p}
            ),
        )
        verifier_output = output_dir / "verifier-output.json"
        spec = PiHostExecutionSpec(
            identity=identity,
            pi=self.pi_binary,
            workspace=str(workspace),
            prompt=_agent_prompt(task_id),
            tools=self.tools,
            timeout_seconds=self.episode_timeout_seconds,
            upstream_url=f"{engine_base}/v1/chat/completions",
            provider="local-qwen-proxy",
            provider_api="openai-completions",
            model=model_id,
            model_revision=f"verl-rollout/{self.policy_fingerprint[:16]}",
            # Pi's own default asks for 8192 tokens, which cannot fit the
            # engine's context once the prompt is included and is far longer
            # than the rollout's response budget. The loop's configured cap is
            # what actually bounds a generation.
            max_tokens_per_generation=self.max_tokens_per_generation,
            # The engine serves one model under its own id; recording that id
            # is what lets the evidence claim a controlled policy identity.
            backend_model_revision=model_id,
            upstream_transport=transport,
            isolate_pi=True,
            training_sequence_builder=_training_sequence_from_evidence,
            harness_manifest=HarnessManifest(
                name="pi",
                version="0.84.2",
                revision="pi-json/v0.84.2",
                config_digest=sha256_json(
                    {"benchmark": "mbpp", "tools": ",".join(self.tools)}
                ),
                policy_flags={"benchmark": "mbpp", "policy": PI_LOOP_VERSION},
            ),
            evaluator_manifest=EvaluatorManifest(
                name="mbpp-execution-verifier",
                revision="mbpp-verifier/v1",
                config_digest=sha256_json({"task": task_id, "policy": "prompt-asserts"}),
            ),
            verifier_command=(
                _verifier_python(),
                self.verifier_script,
                task_id,
                str(workspace),
                str(verifier_output),
            ),
            task_snapshot=str(task_prompt_path),
            experiment_manifest_ref=self.run_id,
            sampling_config={
                "temperature": self.sampling_temperature,
                "top_p": self.sampling_top_p,
            },
            require_rl_evidence=True,
        )
        summary = PiHostExecutionOrchestrator().run(spec, output_dir=output_dir)
        sequence = _assemble(output_dir, episode_id)
        # The finalizer writes the eligibility decision the batch gate later
        # certifies from. Its absence must fail this episode now rather than
        # surface as a batch-level rejection with no local cause.
        eligibility_path = output_dir / "finalized" / "on-policy-rl-eligibility.json"
        if not eligibility_path.is_file():
            raise ContractValidationError(
                f"episode {episode_id} has no finalizer eligibility decision"
            )
        policy_artifact_checksum = summary["policy_artifact_checksum"]
        return summary, sequence, str(policy_artifact_checksum)


def _training_sequence_from_evidence(evidence):
    records = [BridgeCallRecord(
        request_id=item.request.request_id,
        prompt_token_ids=tuple(item.backend.prompt_token_ids),
        response_token_ids=tuple(item.backend.response_token_ids),
        response_logprobs=tuple(item.backend.response_logprobs),
    ) for item in evidence]
    segments, prompt = build_per_call_segments(records)
    sequence = assemble_episode_sequence(
        episode_id=evidence[0].request.identity.episode_id,
        prompt_ids=prompt, per_call_segments=segments,
    )
    return {"prompt_ids": list(sequence.prompt_ids),
            "response_ids": list(sequence.response_ids),
            "loss_mask": list(sequence.response_mask),
            "response_logprobs": list(sequence.response_logprobs)}


def _assemble(output_dir: Path, episode_id: str):
    """Rebuild the certified single sequence from captured native evidence."""
    evidence_path = output_dir / "model-evidence.jsonl"
    rows = [
        json.loads(line)
        for line in evidence_path.read_text().splitlines()
        if line.strip()
    ]
    records = [
        BridgeCallRecord(
            request_id=row["request"]["request_id"],
            prompt_token_ids=tuple(row["backend"]["prompt_token_ids"]),
            response_token_ids=tuple(row["backend"]["response_token_ids"]),
            response_logprobs=tuple(row["backend"]["response_logprobs"]),
        )
        for row in rows
    ]
    segments, frozen_prompt = build_per_call_segments(records)
    return assemble_episode_sequence(
        episode_id=episode_id, prompt_ids=frozen_prompt, per_call_segments=segments
    )


def _read_closeout(output_dir: Path) -> str | None:
    """How a budget ended this episode's model turns, if one did.

    Written by the proxy just before it answers a request with its closeout
    reply. That reply is deliberately not model evidence, so this note is the
    only record that the episode was cut off rather than concluded -- and an
    episode cut off by budget must not be readable as one the policy finished.
    """
    path = output_dir / "engine-closeout.jsonl"
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            return str(json.loads(line).get("reason") or "unknown")
        except (json.JSONDecodeError, AttributeError):
            return "unparseable"
    return None


def _agent_prompt(task_id: str) -> str:
    return (
        "You are in a workspace with solution.py. First read task-prompt.txt "
        "and wait for the result. Then implement the requested function in "
        "solution.py and test it with bash. Use tool results to decide the next "
        "step; finish with a brief summary. Keep reasoning brief. Emit every "
        "tool call by itself, wrapped exactly as "
        '<tool_call>{"name": <tool-name>, "arguments": <json-object>}</tool_call>'
        ". Task %s: implement the function described in task-prompt.txt." % task_id
    )


# The verifier runs under the same interpreter as the trainer unless the
# deployment pins a different one (the trainer venv carries evalplus).
def _verifier_python() -> str:
    return os.environ.get("AGENT_VERIFIER_PYTHON") or sys.executable


__all__ = ["PI_LOOP_VERSION", "FIXED_TOOLS", "PiAgentLoop"]

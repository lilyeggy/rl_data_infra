"""verl agent-loop manager subclass that gates every batch before the update.

verl gives a recipe three sanctioned extension points, and this module uses
exactly those (no patching of the installed framework):

* ``rollout.agent.agent_loop_manager_class`` selects this manager;
* ``AgentLoopManager.agent_loop_workers_class`` and
  ``AgentLoopWorker.server_manager`` are both guarded in the base class with
  "for recipe to change" checks, so a subclass can substitute its own worker
  and server manager.

The one job the framework cannot do for us is the batch gate. Our certification
contract is *batch-level* (one behaviour policy, unique episodes, minimum group
size, non-zero intra-group reward variance, per-episode task binding) while
``AgentLoopBase.run`` is *per-sample*. So certification runs here, after
generation and before the trainer can consume the batch, and it fails closed.

A batch with no reward variance carries no GRPO signal, so instead of feeding a
degenerate batch to the optimizer we redraw it: generation is re-run (fresh
episodes, nothing was consumed) up to ``max_attempts`` times. The attempt number
is stamped into the prompt batch, which is how the per-sample loop learns to
write new episode directories, and each refused attempt is written next to the
certificate so a redraw is visible after the fact. Only a degenerate batch is
redrawn -- see :mod:`src.integrations.verl.resampling` for why every other
violation fails on its first occurrence.
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import ray
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
    AsyncLLMServerManager,
)
from verl.utils.ray_utils import auto_await

from src.contracts._json import sha256_json
from src.errors import ContractValidationError
from src.integrations.verl.admission import AdmittedVerlSequence
from src.integrations.verl.sequence import (
    describe_sequence_difference,
    training_sequence_matches,
)
from src.integrations.verl.manager import CertifiedAgentLoopManager, CertifiedBatch
from src.integrations.verl.resampling import certify_with_resampling
from src.training.policy_fingerprint import PolicyFingerprint
from src.certification import ConsumerProfile, certify_for
from src.contracts.agent_episode import AgentEpisode
from src.contracts.execution_bundle import ExecutionBundle
from src.producers.base import ProducerArtifact

VERL_PI_MANAGER_VERSION = "verl-pi-manager/v1"
PI_ATTEMPT_KEY = "pi_attempt"
PI_SAMPLE_KEY = "pi_sample_token"
PI_BATCH_POSITION_KEY = "pi_batch_position"
PI_ROOT_KEY = "pi_episode_root"
PI_CERTIFICATION_KEY = "pi_certification"


def utcnow_stamp() -> str:
    """Sortable, filesystem-safe UTC timestamp for a session directory."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def generation_root(episode_root: Any, session_id: str, generation: int) -> Path:
    """Directory holding exactly one generate_sequences call's episodes."""
    return Path(episode_root) / session_id / f"gen-{generation:03d}"


def _read(node: Any, key: str, default: Any = None) -> Any:
    """Read a config node that may be an OmegaConf object or a plain mapping."""
    getter = getattr(node, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(node, key, default)

logger = logging.getLogger(__name__)


class PiServerManager(AsyncLLMServerManager):
    """Expose the engine HTTP addresses the stock manager keeps private.

    ``AsyncLLMServerManager`` stores ``(address, handle)`` pairs keyed by
    address. The address is what a subprocess harness needs, so surface it.
    """

    def __init__(self, config: Any, servers: Any, load_balancer_handle: Any) -> None:
        super().__init__(config, servers, load_balancer_handle=load_balancer_handle)
        self.http_addresses = [address for address, _handle in servers]


class PiAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker whose server manager publishes the HTTP addresses."""

    def __init__(
        self,
        config: Any,
        servers: Any,
        load_balancer_handle: Any,
        reward_loop_worker_handles: Any = None,
    ) -> None:
        # Set before super(): the base class only builds its own manager when
        # `server_manager` is absent, which is the documented recipe hook.
        self.server_manager = PiServerManager(config, servers, load_balancer_handle)
        super().__init__(config, servers, load_balancer_handle, reward_loop_worker_handles)


class CertifiedVerlAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that certifies each generated batch before the update."""

    agent_loop_workers_class = ray.remote(PiAgentLoopWorker)

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        settings = _manager_settings(config)
        self.run_id = settings["run_id"]
        self.episode_root = Path(settings["episode_root"])
        self.minimum_group_size = int(settings["minimum_group_size"])
        self.max_attempts = int(settings["max_attempts"])
        policy_path = Path(settings["policy_fingerprint_json"])
        if not policy_path.is_file():
            raise ContractValidationError(
                f"round policy fingerprint file is missing: {policy_path}"
            )
        self.policy = PolicyFingerprint.from_dict(json.loads(policy_path.read_text()))
        self.initial_policy = self.policy
        self.checkpoint_root = Path(config.trainer.default_local_dir)
        self.last_certified_batch: CertifiedBatch | None = None
        # generate_sequences is not called once per run: the trainer validates
        # before training (ray_trainer.py:1259) and then calls it again for
        # every step, each time with the same tasks and the same per-sample
        # tokens. Episodes therefore cannot be named from the sample alone --
        # a second call would rewrite the first call's directories, which the
        # episode writer refuses to do (it fails closed on an existing root).
        # Each call gets its own generation directory, and the session id keeps
        # a re-run of the same round from colliding with the previous one.
        self.session_id = f"{self.run_id}-{utcnow_stamp()}-{os.getpid()}"
        self._generation = 0
        logger.info(
            "%s: episode root for this session is %s",
            self.run_id,
            self.episode_root / self.session_id,
        )

    # The trainer drives the whole fit loop synchronously (ray_trainer.py:546
    # and :1321 call this without awaiting), so this override has to carry the
    # same @auto_await bridge the base method does. Without it the caller gets
    # back a coroutine object and fails on the first attribute access.
    @auto_await
    async def generate_sequences(self, prompts: Any) -> Any:
        generation = self._generation
        self._generation += 1
        call_root = generation_root(self.episode_root, self.session_id, generation)
        engine_step = int(prompts.meta_info["global_steps"]) - 1
        if engine_step < 0:
            raise ContractValidationError("training step must identify the preceding synchronized policy")
        if engine_step:
            checkpoint = self.checkpoint_root / f"global_step_{engine_step}" / "actor"
            files = sorted(p for p in checkpoint.rglob("*") if p.is_file())
            if not files:
                raise ContractValidationError(f"synchronous checkpoint missing: {checkpoint}")
            digest = hashlib.sha256()
            for path in files:
                digest.update(str(path.relative_to(checkpoint)).encode() + b"\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            self.policy = replace(self.initial_policy, adapter_revision=digest.hexdigest(),
                                  policy_generation=f"P{engine_step}")
        call_root.mkdir(parents=True, exist_ok=False)
        (call_root / "round-policy.json").write_text(json.dumps(self.policy.to_dict(), indent=2))
        prompts.non_tensor_batch["pi_policy_fingerprint"] = np.array([self.policy.checksum()] * len(prompts), dtype=object)
        prompts.non_tensor_batch["pi_expected_engine_step"] = np.array([engine_step] * len(prompts), dtype=np.int64)
        # print, not logger: the worker/driver loggers are filtered here and the
        # banner has to survive into the aggregated run log.
        print(
            f"[pi-manager] {self.run_id} generation={generation} "
            f"prompt_rows={len(prompts)} root={call_root}",
            flush=True,
        )
        result = await certify_with_resampling(
            max_attempts=self.max_attempts,
            run_id=self.run_id,
            generate=lambda attempt: self._generate_attempt(prompts, call_root, attempt),
            certify=lambda batch, attempt: self._gate_attempt(batch, prompts, call_root, attempt),
            on_retry=lambda attempt, exc: self._record_rejection(call_root, attempt, exc),
        )
        self.last_certified_batch = result.certificate
        self._record_certificate(result.certificate, result.attempt, call_root)
        logger.info(
            "%s: attempt %d certified batch %s over %d sequences",
            self.run_id,
            result.attempt,
            result.certificate.batch_id,
            len(result.certificate.sequence_checksums),
        )
        return result.batch

    async def _generate_attempt(self, prompts: Any, call_root: Path, attempt: int) -> Any:
        """Stamp the attempt, then let the framework roll the batch out."""
        self._stamp_attempt(prompts, attempt, call_root)
        return await super().generate_sequences(prompts)

    def _gate_attempt(
        self, batch: Any, prompts: Any, call_root: Path, attempt: int
    ) -> CertifiedBatch:
        """Certify one attempt from persisted evidence, then check the batch."""
        certified = self._certify_from_evidence(call_root, attempt)
        self._verify_returned_batch(batch, prompts, call_root, attempt)
        return certified

    def _record_rejection(self, call_root: Path, attempt: int, exc: Exception) -> None:
        """Leave the redraw on disk, so a resample cannot look like a first try."""
        payload = {
            "run_id": self.run_id,
            "attempt": attempt,
            "reason": type(exc).__name__,
            "message": str(exc),
            "manager_version": VERL_PI_MANAGER_VERSION,
        }
        (call_root / f"rejected-attempt{attempt}.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        print(
            f"[pi-manager] {self.run_id} attempt {attempt} carries no learning "
            f"signal, redrawing: {exc}",
            flush=True,
        )

    @staticmethod
    def _stamp_attempt(prompts: Any, attempt: int, call_root: Path) -> None:
        non_tensor = getattr(prompts, "non_tensor_batch", None)
        if non_tensor is None:
            raise ContractValidationError("prompt batch has no non_tensor_batch")
        size = len(prompts)
        non_tensor[PI_ATTEMPT_KEY] = np.array([attempt] * size, dtype=np.int64)
        # The per-sample loop runs in a different process from this manager, so
        # the output root travels with the batch rather than being recomputed.
        non_tensor[PI_ROOT_KEY] = np.array([str(call_root)] * size, dtype=object)
        # The trainer expands one dataset row into `rollout.n` identical rows
        # (sharing one uid), so nothing else in the batch distinguishes the n
        # samples of a group. Stamp a unique token per sample so every episode
        # gets its own id, output directory and raw_events file.
        non_tensor[PI_SAMPLE_KEY] = np.array(
            [f"a{attempt}s{position}" for position in range(size)], dtype=object
        )
        non_tensor[PI_BATCH_POSITION_KEY] = np.arange(size, dtype=np.int64)

    def _certify_from_evidence(self, call_root: Path, attempt: int) -> CertifiedBatch:
        """Certify the whole attempt from the episodes' persisted evidence."""
        attempt_root = call_root / f"attempt-{attempt}"
        if not attempt_root.is_dir():
            raise ContractValidationError(f"attempt root is missing: {attempt_root}")
        episode_dirs = sorted(path for path in attempt_root.iterdir() if path.is_dir())
        if not episode_dirs:
            raise ContractValidationError(f"attempt {attempt} produced no episodes")
        sequences: list[AdmittedVerlSequence] = []
        task_ids: dict[str, str] = {}
        for episode_dir in episode_dirs:
            sequence_path = episode_dir / "admitted-sequence.json"
            summary_path = episode_dir / "summary.json"
            if not sequence_path.is_file() or not summary_path.is_file():
                raise ContractValidationError(
                    f"episode {episode_dir.name} is missing sequence or summary evidence"
                )
            summary = json.loads(summary_path.read_text())
            if summary.get("verifier_status") not in ("PASSED", "FAILED"):
                raise ContractValidationError(
                    f"episode {episode_dir.name} has no terminating verifier verdict "
                    f"({summary.get('verifier_status')!r})"
                )
            if summary.get("on_policy_rl_verdict") != "ELIGIBLE":
                raise ContractValidationError(
                    f"episode {episode_dir.name} is not ON_POLICY_RL eligible "
                    f"({summary.get('on_policy_rl_verdict')!r})"
                )
            sequence = AdmittedVerlSequence.from_dict(json.loads(sequence_path.read_text()))
            episode = AgentEpisode.from_dict(json.loads((episode_dir / "finalized/episode.json").read_text()))
            bundle = ExecutionBundle.from_dict(json.loads((episode_dir / "finalized/execution-bundle.json").read_text()))
            artifact = ProducerArtifact.from_dict(json.loads((episode_dir / "policy-artifact.json").read_text()))
            decision = certify_for(episode, ConsumerProfile.ON_POLICY_RL, execution_bundle=bundle,
                                   policy_artifact=artifact, target_policy_fingerprint=self.policy.checksum())
            if decision.verdict.value != "ELIGIBLE":
                raise ContractValidationError("raw episode failed fresh certification")
            if artifact.checksum != sequence.policy_artifact_checksum or bundle.checksum != sequence.execution_bundle_checksum:
                raise ContractValidationError("admitted sequence references different immutable evidence")
            from src.integrations.verl.pi_loop import _assemble
            rebuilt = _assemble(episode_dir, sequence.episode_id)
            training = {"prompt_ids": list(rebuilt.prompt_ids), "response_ids": list(rebuilt.response_ids),
                        "loss_mask": list(rebuilt.response_mask), "response_logprobs": list(rebuilt.response_logprobs)}
            # print, not logger: the worker/driver loggers are filtered here, and
            # a gate that rejects 16 episodes must say which one and why in the
            # run log the operator actually reads.
            print(
                f"[pi-manager] certify {episode_dir.name} calls={sequence.num_model_calls} "
                f"tool_rounds={sequence.num_tool_rounds} seq_len={len(sequence.response_ids)} "
                f"verifier={summary.get('verifier_status')}",
                flush=True,
            )
            if not training_sequence_matches(artifact.payload.get("training_sequence"), training):
                raise ContractValidationError(
                    f"policy artifact does not bind actual inference context "
                    f"({episode_dir.name}): "
                    f"{describe_sequence_difference(artifact.payload.get('training_sequence'), training)}"
                )
            if (sequence.prompt_ids != tuple(rebuilt.prompt_ids) or sequence.response_ids != tuple(rebuilt.response_ids)
                    or sequence.response_mask != tuple(rebuilt.response_mask)
                    or sequence.response_logprobs != tuple(rebuilt.response_logprobs)
                    or sequence.reward != (1.0 if summary["verifier_status"] == "PASSED" else 0.0)):
                raise ContractValidationError("admitted tensors or reward differ from captured evidence")
            sequences.append(sequence)
            task_id = summary.get("task_id")
            if not task_id:
                raise ContractValidationError(
                    f"episode {episode_dir.name} has no task binding in its summary"
                )
            task_ids[sequence.episode_id] = str(task_id)
        gate = CertifiedAgentLoopManager(
            policy=self.policy, minimum_group_size=self.minimum_group_size
        )
        return gate.certify_batch(
            sequences,
            batch_id=f"{self.run_id}-attempt{attempt}",
            task_ids_by_episode=task_ids,
        )

    @staticmethod
    def _verify_returned_batch(batch, prompts, call_root, attempt):
        if len(batch) != len(prompts):
            raise ContractValidationError("framework changed batch cardinality")
        seen = set()
        for i in range(len(batch)):
            episode_id = str(batch.non_tensor_batch["episode_id"][i])
            if episode_id in seen or Path(episode_id).name != episode_id:
                raise ContractValidationError("duplicate or invalid returned episode ID")
            seen.add(episode_id)
            seq = AdmittedVerlSequence.from_dict(json.loads(
                (call_root / f"attempt-{attempt}" / episode_id / "admitted-sequence.json").read_text()))
            if seq.group_id != str(prompts.non_tensor_batch["uid"][i]):
                raise ContractValidationError("returned episode changed native uid group")
            tensors = batch.batch
            prompt_width = tensors["prompts"].shape[1]
            attention = tensors["attention_mask"][i].bool()
            if tensors["prompts"][i][attention[:prompt_width]].tolist() != list(seq.prompt_ids):
                raise ContractValidationError("framework prompt differs from certified tokens")
            valid = attention[prompt_width:]
            if tensors["responses"][i][valid].tolist() != list(seq.response_ids):
                raise ContractValidationError("framework response differs from certified tokens")
            mask = tensors["response_mask"][i]
            if mask[valid].tolist() != list(seq.response_mask) or mask[~valid].any():
                raise ContractValidationError("framework altered loss mask or trains padding")
            import torch
            expected = torch.tensor(seq.response_logprobs, dtype=tensors["rollout_log_probs"].dtype)
            if not torch.equal(tensors["rollout_log_probs"][i][valid].cpu(), expected):
                raise ContractValidationError("framework altered native logprobs")
            reward = tensors["rm_scores"][i]
            if float(reward.sum()) != seq.reward or reward[~valid].any():
                raise ContractValidationError("framework altered verifier reward")
        evidence_count = sum(p.is_dir() for p in (call_root / f"attempt-{attempt}").iterdir())
        if evidence_count != len(batch):
            raise ContractValidationError("unconsumed or missing episode evidence")

    def _record_certificate(self, certified: CertifiedBatch, attempt: int, call_root: Path) -> None:
        payload = certified.to_dict() | {
            "run_id": self.run_id,
            "attempt": attempt,
            "manager_version": VERL_PI_MANAGER_VERSION,
        }
        payload["batch_checksum"] = sha256_json(certified.to_dict())
        (call_root / f"certified-batch-attempt{attempt}.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


def _manager_settings(config: Any) -> dict[str, Any]:
    # Read from the top level: `actor_rollout_ref.rollout.agent` is converted to
    # the typed AgentLoopConfig dataclass, which rejects unknown keys, while the
    # root config stays a plain DictConfig that tolerates recipe additions.
    settings = _read(config, PI_CERTIFICATION_KEY)
    if settings is None:
        raise ContractValidationError(
            f"{PI_CERTIFICATION_KEY} is required by CertifiedVerlAgentLoopManager"
        )
    required = (
        "run_id",
        "episode_root",
        "minimum_group_size",
        "max_attempts",
        "policy_fingerprint_json",
    )
    missing = [name for name in required if name not in settings]
    if missing:
        raise ContractValidationError(f"{PI_CERTIFICATION_KEY} is missing: {missing}")
    return {name: settings[name] for name in required}


__all__ = [
    "VERL_PI_MANAGER_VERSION",
    "PI_ATTEMPT_KEY",
    "PI_SAMPLE_KEY",
    "PI_CERTIFICATION_KEY",
    "PiServerManager",
    "PiAgentLoopWorker",
    "CertifiedVerlAgentLoopManager",
]

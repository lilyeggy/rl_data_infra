"""Phase B contract tests for the verl integration (CPU only, no GPU).

Simulated inference responses are used, but assertions run against the real
pinned verl v0.7.1 ``AgentLoopOutput``/``AgentLoopMetrics`` types loaded from
the downloaded sdist. Simulated artifacts are marked and never reused as GPU
acceptance artifacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

VERL_SDIST_ROOT = Path("/tmp/verl-closeout/phase-a/verl-0.7.1")
if str(VERL_SDIST_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_SDIST_ROOT))

try:
    from verl.experimental.agent_loop.agent_loop import (  # noqa: E402  (pinned real type)
        AgentLoopMetrics,
        AgentLoopOutput,
    )
    from verl import __version__ as _VERL_VERSION  # noqa: E402

    if _VERL_VERSION != "0.7.1":
        raise ImportError(f"pinned verl 0.7.1 required, found {_VERL_VERSION}")
    HAS_REAL_VERL_TYPES = True
except Exception:  # pragma: no cover - documents missing-verl behavior
    HAS_REAL_VERL_TYPES = False
    AgentLoopMetrics = None  # type: ignore[assignment]
    AgentLoopOutput = None  # type: ignore[assignment]

from src.assembly import assemble_execution_bundle  # noqa: E402
from src.assembly.episode_assembler import EpisodeAssembler  # noqa: E402
from src.certification import ConsumerProfile, certify_for  # noqa: E402
from src.contracts.dataset import DatasetPurpose, DatasetRole, DatasetSplit  # noqa: E402
from src.contracts.execution_identity import ExecutionIdentity  # noqa: E402
from src.errors import ContractValidationError  # noqa: E402
from src.integrations.verl import (  # noqa: E402
    AdmittedVerlSequence,
    BridgeCallRecord,
    CertifiedAgentLoopManager,
    PiAgentLoopConfig,
    RunCleanup,
    admit_on_policy_manifest,
    assemble_episode_sequence,
    build_per_call_segments,
    to_agent_loop_output_dict,
)
from src.integrations.verl.sequence import SEQUENCE_ASSEMBLER_VERSION  # noqa: E402
from src.learning import CertifiedArtifact, compile_dataset  # noqa: E402
from src.producers.base import (  # noqa: E402
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)
from src.training.policy_fingerprint import PolicyFingerprint  # noqa: E402
from tests.execution_fixtures import (  # noqa: E402
    make_complete_event_stream,
    make_episode_context,
)

SIMULATED_MARKER = "simulated-cpu-contract-only"


def _policy(generation: str = "P0") -> PolicyFingerprint:
    return PolicyFingerprint(
        provider="verl-vllm",
        model_id="qwen2.5-coder-14b",
        base_model_revision="base-rev",
        adapter_revision="adapter-rev-p0",
        tokenizer_revision="tok-rev",
        chat_template_checksum="c" * 64,
        tool_schema_checksum="t" * 64,
        sampling_config={"temperature": 1.0, "top_p": 1.0},
        temperature=1.0,
        top_p=1.0,
        policy_generation=generation,
    )


def _two_turn_sequence(episode_id: str):
    return assemble_episode_sequence(
        episode_id=episode_id,
        prompt_ids=[100, 101, 102],
        per_call_segments=[
            {
                "native_response_ids": [10, 11, 12],
                "native_response_logprobs": [-0.1, -0.2, -0.3],
                "context_suffix_ids": [200, 201],
            },
            {
                "native_response_ids": [13, 14],
                "native_response_logprobs": [-0.4, -0.5],
            },
        ],
    )


def _member(index: int, *, reward: float, policy: PolicyFingerprint):
    episode_id = f"episode-verl-{index}"
    episode = EpisodeAssembler().assemble(
        make_complete_event_stream(run_id="run-verl", episode_id=episode_id),
        contexts={episode_id: make_episode_context(task_id="task-verl", attempt=index)},
    ).episodes[0]
    identity = ExecutionIdentity(
        run_id=episode.run_id,
        task_id=episode.task_id,
        episode_id=episode.episode_id,
        attempt_id=episode.attempt,
        producer_id="pi-real",
        producer_version="0.84.2",
        group_id="group-verl",
        policy_fingerprint=policy.checksum(),
        sampling_fingerprint="b" * 64,
    )
    sequence = _two_turn_sequence(episode_id)
    # Real GPU artifacts carry BOTH the raw per-call traces (audit, existing
    # format consumed by certification) AND the assembled episode-level
    # verl_sequence (training, consumed by verl admission).
    artifact = ProducerArtifact(
        identity=identity,
        status=ProducerExecutionStatus.COMPLETED,
        capabilities=frozenset(
            {
                ProducerCapability.TOKEN_IDS,
                ProducerCapability.ACTION_MASK,
                ProducerCapability.BEHAVIOR_LOGPROBS,
                ProducerCapability.POLICY_VERSION,
                ProducerCapability.VERIFIER_EVIDENCE,
            }
        ),
        payload={
            "trajectory": {
                "traces": [
                    {
                        "request_id": f"req-{episode_id}-1",
                        "response_ids": [10, 11, 12],
                        "action_mask": [1, 1, 1],
                        "response_logprobs": [-0.1, -0.2, -0.3],
                        "reward": reward,
                        "backend_model_revision": "adapter-rev-p0",
                    },
                    {
                        "request_id": f"req-{episode_id}-2",
                        "response_ids": [13, 14],
                        "action_mask": [1, 1],
                        "response_logprobs": [-0.4, -0.5],
                        "reward": reward,
                        "backend_model_revision": "adapter-rev-p0",
                    },
                ],
                "verl_sequence": sequence.to_dict(),
                "reward": reward,
            },
        },
    )
    bundle = assemble_execution_bundle(
        identity=identity,
        episode=episode,
        producer_artifacts=(artifact,),
        verifier_report_checksum=(str(index) * 64),
    )
    decision = certify_for(
        episode,
        ConsumerProfile.ON_POLICY_RL,
        execution_bundle=bundle,
        policy_artifact=artifact,
        target_policy_fingerprint=policy.checksum(),
    )
    assert decision.verdict.value == "ELIGIBLE", decision.reasons
    return identity, artifact, bundle, decision


def _manifest(entries):
    return compile_dataset(
        dataset_id="dataset-verl",
        revision="r1",
        purpose=DatasetPurpose.ON_POLICY_RL,
        selection_policy_version="selection/v1",
        artifacts=tuple(
            CertifiedArtifact(
                identity=identity,
                decision=decision,
                artifact_checksum=artifact.checksum,
                split=DatasetSplit.TRAIN,
                role=DatasetRole.TRAJECTORY,
            )
            for identity, artifact, _, decision in entries
        ),
    )


class SequenceContractTest(unittest.TestCase):
    def test_two_turn_tool_call_keeps_native_tokens_and_masks_observation(self) -> None:
        sequence = _two_turn_sequence("episode-seq-1")
        self.assertEqual(sequence.response_ids, (10, 11, 12, 200, 201, 13, 14))
        self.assertEqual(sequence.response_mask, (1, 1, 1, 0, 0, 1, 1))
        self.assertEqual(len(sequence.response_logprobs), 7)
        # Observation slots carry an explicit non-probability placeholder.
        self.assertEqual(sequence.response_logprobs[3], 0.0)
        self.assertEqual(sequence.num_tool_rounds, 1)

    def test_single_turn_without_a_tool_round_is_still_a_sample(self) -> None:
        """A turn with no tool call is a sample with reward 0, not a fault.

        Measured on smoke23: six of eight first turns carried no tool call at
        all (the extractor refused only prose placeholders such as `{}` and
        `{...}`, and rejected no valid call). The ON_POLICY_RL profile certified
        such an episode ELIGIBLE with score 0.0, while our own stricter gate
        aborted the entire batch gather -- which also discarded the two episodes
        of the same batch that were already issuing their second model request.
        The action is the policy's own generation and the reward is the observed
        outcome, which is all PPO-style training needs; a batch made only of such
        episodes has no variance and is still rejected by the batch gate.

        `admit_on_policy_manifest` (the CPU certifier path, which the live loop
        does not use) keeps requiring one tool round: that is a deliberate
        definition of a qualifying trajectory, and the two paths may differ.
        """
        sequence = assemble_episode_sequence(
            episode_id="episode-single",
            prompt_ids=[1, 2],
            per_call_segments=[
                {
                    "native_response_ids": [10, 11, 12],
                    "native_response_logprobs": [-0.1, -0.2, -0.3],
                    "context_suffix_ids": [],
                    "history_rewritten": False,
                }
            ],
        )
        self.assertEqual(sequence.num_model_calls, 1)
        self.assertEqual(sequence.num_tool_rounds, 0)
        self.assertEqual(sequence.response_ids, (10, 11, 12))
        self.assertEqual(sequence.response_mask, (1, 1, 1))
        admitted = AdmittedVerlSequence(
            episode_id="episode-single",
            group_id="run",
            policy_fingerprint="f" * 64,
            prompt_ids=sequence.prompt_ids,
            response_ids=sequence.response_ids,
            response_mask=sequence.response_mask,
            response_logprobs=sequence.response_logprobs,
            reward=0.0,
            num_model_calls=1,
            num_tool_rounds=0,
            member_id="episode-single",
            execution_bundle_checksum="a" * 64,
            policy_artifact_checksum="b" * 64,
        )
        rebuilt = AdmittedVerlSequence.from_dict(json.loads(json.dumps(admitted.to_dict())))
        self.assertEqual(rebuilt.num_tool_rounds, 0)
        self.assertEqual(rebuilt.reward, 0.0)

    def test_decode_drift_does_not_rewrite_native_tokens(self) -> None:
        # A decoder might re-encode [10, 11] as [10, 11, 99]; the assembler
        # must keep the native evidence and reject the rewrite.
        with self.assertRaises(ContractValidationError):
            assemble_episode_sequence(
                episode_id="episode-seq-drift",
                prompt_ids=[1],
                per_call_segments=[
                    {
                        "native_response_ids": [10, 11],
                        "native_response_logprobs": [-0.1, -0.2],
                        "context_suffix_ids": [10, 11, 99],
                        "history_rewritten": True,
                    }
                ],
            )

    def test_observation_only_in_context_never_in_loss(self) -> None:
        sequence = _two_turn_sequence("episode-seq-2")
        trainable = [
            token
            for token, mask in zip(sequence.response_ids, sequence.response_mask)
            if mask == 1
        ]
        self.assertEqual(trainable, [10, 11, 12, 13, 14])
        self.assertNotIn(200, trainable)

    def test_rejects_missing_mismatched_or_nonfinite(self) -> None:
        good = {
            "native_response_ids": [10],
            "native_response_logprobs": [-0.1],
        }
        with self.assertRaises(ContractValidationError):
            assemble_episode_sequence(
                episode_id="e", prompt_ids=[1], per_call_segments=[{**good, "native_response_ids": []}]
            )
        with self.assertRaises(ContractValidationError):
            assemble_episode_sequence(
                episode_id="e",
                prompt_ids=[1],
                per_call_segments=[{**good, "native_response_logprobs": [-0.1, -0.2]}],
            )
        with self.assertRaises(ContractValidationError):
            assemble_episode_sequence(
                episode_id="e",
                prompt_ids=[1],
                per_call_segments=[{**good, "native_response_logprobs": [float("nan")]}],
            )


class BridgeContractTest(unittest.TestCase):
    def test_contiguity_derives_suffix_arithmetically(self) -> None:
        records = [
            BridgeCallRecord(
                request_id="r1",
                prompt_token_ids=(1, 2),
                response_token_ids=(10, 11),
                response_logprobs=(-0.1, -0.2),
            ),
            BridgeCallRecord(
                request_id="r2",
                prompt_token_ids=(1, 2, 10, 11, 200),
                response_token_ids=(12,),
                response_logprobs=(-0.3,),
            ),
        ]
        segments, prompt = build_per_call_segments(records)
        self.assertEqual(tuple(prompt), (1, 2))
        self.assertEqual(segments[0]["context_suffix_ids"], [200])
        self.assertEqual(segments[1]["context_suffix_ids"], [])

    def test_history_rewrite_rejects(self) -> None:
        records = [
            BridgeCallRecord(
                request_id="r1",
                prompt_token_ids=(1, 2),
                response_token_ids=(10, 11),
                response_logprobs=(-0.1, -0.2),
            ),
            BridgeCallRecord(
                request_id="r2",
                prompt_token_ids=(1, 2, 999, 11, 200),
                response_token_ids=(12,),
                response_logprobs=(-0.3,),
            ),
        ]
        with self.assertRaises(ContractValidationError):
            build_per_call_segments(records)

    @unittest.skipUnless(HAS_REAL_VERL_TYPES, "requires pinned verl 0.7.1 sdist + deps")
    def test_real_verl_type_accepts_bridge_output(self) -> None:
        """Simulated tokens through the REAL pinned verl type + postprocess math."""
        policy = _policy()
        entries = [_member(1, reward=1.0, policy=policy), _member(2, reward=0.0, policy=policy)]
        manifest = _manifest(entries)
        batch = admit_on_policy_manifest(
            manifest,
            decisions_by_checksum={d.checksum: d for _, _, _, d in entries},
            bundles_by_checksum={b.checksum: b for _, _, b, _ in entries},
            artifacts_by_checksum={a.checksum: a for _, a, _, _ in entries},
            minimum_group_size=2,
        )
        first = batch.sequences[0]
        payload = to_agent_loop_output_dict(first, num_turns=3)
        payload["extra_fields"][SIMULATED_MARKER] = True
        output = AgentLoopOutput(
            prompt_ids=payload["prompt_ids"],
            response_ids=payload["response_ids"],
            response_mask=payload["response_mask"],
            response_logprobs=payload["response_logprobs"],
            reward_score=payload["reward_score"],
            num_turns=payload["num_turns"],
            metrics=AgentLoopMetrics(**payload["metrics"]),
            extra_fields=payload["extra_fields"],
        )
        # Mirror verl's _agent_loop_postprocess mask semantics on CPU:
        # response_mask is multiplied by the response attention mask so that
        # padding never contributes to the loss.
        import torch

        response_length = 16
        pad = response_length - len(output.response_mask)
        padded_mask = torch.tensor(output.response_mask + [0] * pad)
        attention = torch.tensor([1] * len(output.response_mask) + [0] * pad)
        effective = padded_mask * attention
        self.assertEqual(int(effective.sum()), sum(output.response_mask))
        self.assertEqual(output.extra_fields["episode_id"], first.episode_id)
        self.assertTrue(output.extra_fields[SIMULATED_MARKER])


class AdmissionContractTest(unittest.TestCase):
    def test_group_counts_episodes_not_calls(self) -> None:
        policy = _policy()
        entries = [_member(1, reward=1.0, policy=policy), _member(2, reward=0.0, policy=policy)]
        manifest = _manifest(entries)
        batch = admit_on_policy_manifest(
            manifest,
            decisions_by_checksum={d.checksum: d for _, _, _, d in entries},
            bundles_by_checksum={b.checksum: b for _, _, b, _ in entries},
            artifacts_by_checksum={a.checksum: a for _, a, _, _ in entries},
            minimum_group_size=2,
        )
        self.assertEqual(batch.trajectory_count, 2)
        self.assertEqual(batch.group_sizes, {"group-verl": 2})
        # One sequence per episode even though each episode has 2 model calls.
        self.assertEqual(len(batch.sequences), 2)
        self.assertTrue(all(s.num_model_calls == 2 for s in batch.sequences))

    def test_rejects_per_call_trace_without_sequence(self) -> None:
        policy = _policy()
        identity, artifact, bundle, decision = _member(1, reward=1.0, policy=policy)
        legacy = ProducerArtifact(
            identity=artifact.identity,
            status=artifact.status,
            capabilities=artifact.capabilities,
            payload={
                "trajectory": {
                    "traces": [
                        {
                            "prompt_ids": [1],
                            "response_ids": [2],
                            "loss_mask": [1],
                            "response_logprobs": [-0.1],
                            "reward": 1.0,
                        }
                    ]
                }
            },
        )
        entries = [(identity, legacy, bundle, decision)]
        manifest = _manifest([(identity, artifact, bundle, decision)])
        with self.assertRaises(ContractValidationError):
            admit_on_policy_manifest(
                manifest,
                decisions_by_checksum={decision.checksum: decision},
                bundles_by_checksum={bundle.checksum: bundle},
                artifacts_by_checksum={legacy.checksum: legacy},
                minimum_group_size=2,
            )

    def test_rejects_mixed_policy_and_duplicate_episode(self) -> None:
        policy = _policy()
        other = _policy(generation="P1")
        entries = [_member(1, reward=1.0, policy=policy), _member(2, reward=0.0, policy=other)]
        manifest = _manifest(entries)
        with self.assertRaises(ContractValidationError):
            admit_on_policy_manifest(
                manifest,
                decisions_by_checksum={d.checksum: d for _, _, _, d in entries},
                bundles_by_checksum={b.checksum: b for _, _, b, _ in entries},
                artifacts_by_checksum={a.checksum: a for _, a, _, _ in entries},
                minimum_group_size=2,
            )

    def test_rejects_zero_variance_batch(self) -> None:
        policy = _policy()
        entries = [_member(1, reward=0.0, policy=policy), _member(2, reward=0.0, policy=policy)]
        manifest = _manifest(entries)
        with self.assertRaises(ContractValidationError):
            admit_on_policy_manifest(
                manifest,
                decisions_by_checksum={d.checksum: d for _, _, _, d in entries},
                bundles_by_checksum={b.checksum: b for _, _, b, _ in entries},
                artifacts_by_checksum={a.checksum: a for _, a, _, _ in entries},
                minimum_group_size=2,
            )

    def test_rejects_tool_less_sequence(self) -> None:
        policy = _policy()
        identity, artifact, bundle, decision = _member(1, reward=1.0, policy=policy)
        sequence = assemble_episode_sequence(
            episode_id=identity.episode_id,
            prompt_ids=[1, 2],
            per_call_segments=[
                {"native_response_ids": [10], "native_response_logprobs": [-0.1]},
            ],
        )
        no_tool = ProducerArtifact(
            identity=artifact.identity,
            status=artifact.status,
            capabilities=artifact.capabilities,
            payload={"trajectory": {"verl_sequence": sequence.to_dict(), "reward": 1.0}},
        )
        manifest = _manifest([(identity, artifact, bundle, decision)])
        with self.assertRaises(ContractValidationError):
            admit_on_policy_manifest(
                manifest,
                decisions_by_checksum={decision.checksum: decision},
                bundles_by_checksum={bundle.checksum: bundle},
                artifacts_by_checksum={no_tool.checksum: no_tool},
                minimum_group_size=2,
            )


class ManagerContractTest(unittest.TestCase):
    def test_between_group_difference_is_not_intra_group_signal(self) -> None:
        policy = _policy()
        admitted, _ = self._batch(policy)
        first = admitted.sequences[0]
        sequences = tuple(replace(first, episode_id=f"e{i}", group_id=f"g{i // 2}",
                                  reward=float(i // 2)) for i in range(4))
        with self.assertRaisesRegex(ContractValidationError, "intra-group"):
            CertifiedAgentLoopManager(policy=policy, minimum_group_size=2).certify_batch(
                sequences, batch_id="b", task_ids_by_episode={s.episode_id: "t" for s in sequences})

    def test_group_cannot_mix_tasks(self) -> None:
        policy = _policy()
        admitted, _ = self._batch(policy)
        with self.assertRaisesRegex(ContractValidationError, "mixes tasks"):
            CertifiedAgentLoopManager(policy=policy, minimum_group_size=2).certify_batch(
                admitted.sequences, batch_id="b",
                task_ids_by_episode={s.episode_id: f"t{i}" for i, s in enumerate(admitted.sequences)})

    def _batch(self, policy):
        entries = [_member(1, reward=1.0, policy=policy), _member(2, reward=0.0, policy=policy)]
        manifest = _manifest(entries)
        admitted = admit_on_policy_manifest(
            manifest,
            decisions_by_checksum={d.checksum: d for _, _, _, d in entries},
            bundles_by_checksum={b.checksum: b for _, _, b, _ in entries},
            artifacts_by_checksum={a.checksum: a for _, a, _, _ in entries},
            minimum_group_size=2,
        )
        return admitted, entries

    def test_certifies_complete_batch_with_lineage(self) -> None:
        policy = _policy()
        admitted, _ = self._batch(policy)
        manager = CertifiedAgentLoopManager(policy=policy, minimum_group_size=2)
        certified = manager.certify_batch(
            admitted.sequences,
            batch_id="batch-test-1",
            task_ids_by_episode={s.episode_id: "task-verl" for s in admitted.sequences},
        )
        self.assertEqual(certified.policy_generation, "P0")
        self.assertEqual(len(certified.sequence_checksums), 2)

    def test_rejects_foreign_policy_and_duplicates(self) -> None:
        policy = _policy()
        admitted, _ = self._batch(policy)
        manager = CertifiedAgentLoopManager(policy=_policy(generation="P1"), minimum_group_size=2)
        with self.assertRaises(ContractValidationError):
            manager.certify_batch(
                admitted.sequences,
                batch_id="b",
                task_ids_by_episode={s.episode_id: "task-verl" for s in admitted.sequences},
            )
        manager_ok = CertifiedAgentLoopManager(policy=policy, minimum_group_size=2)
        with self.assertRaises(ContractValidationError):
            manager_ok.certify_batch(
                (admitted.sequences[0], admitted.sequences[0]),
                batch_id="b",
                task_ids_by_episode={admitted.sequences[0].episode_id: "task-verl"},
            )

    def test_config_rejects_bad_toolset_and_context(self) -> None:
        policy = _policy()
        with self.assertRaises(ContractValidationError):
            PiAgentLoopConfig(run_id="r", policy=policy, tools=("read", "bash"))
        with self.assertRaises(ContractValidationError):
            PiAgentLoopConfig(
                run_id="r", policy=policy, max_prompt_tokens=4096, max_response_tokens=8192
            )


class LifecycleContractTest(unittest.TestCase):
    def test_scoped_cleanup_only_registered(self) -> None:
        cleanup = RunCleanup()
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        cleanup.register("test-sleeper", proc)
        outcomes = cleanup.terminate_all(grace_seconds=2.0)
        self.assertIn("test-sleeper", outcomes)
        self.assertEqual(proc.poll() is not None, True)

    def test_request_timeout_surfaces_as_infra_not_reward_zero(self) -> None:
        # A model-request timeout is an infrastructure error: it must never be
        # recorded as a valid 0-reward rollout.
        from src.capture.model_proxy import (
            ModelBackendResponse,
            ModelEndpointKind,
            ModelProxyRequest,
            capture_model_call,
        )

        identity = ExecutionIdentity(
            run_id="run-t",
            task_id="task-t",
            episode_id="episode-t",
            attempt_id=1,
            producer_id="pi",
            producer_version="0.84.2",
            group_id="g",
            policy_fingerprint="a" * 64,
            sampling_fingerprint="b" * 64,
        )
        request = ModelProxyRequest(
            identity=identity,
            request_id="req-timeout",
            endpoint_kind=ModelEndpointKind.CONTROLLED,
            model_id="qwen",
            messages=({"role": "user", "content": "hi"},),
        )
        backend = ModelBackendResponse(
            response={"error": "timeout"},
            latency_ms=1000.0,
            status_code=504,
            backend_model_revision="rev",
        )
        evidence = capture_model_call(request, backend)
        self.assertFalse(evidence.rl_usable_call)


class ProjectImportContractTest(unittest.TestCase):
    def test_core_modules_import_without_verl(self) -> None:
        # Isolated subprocess without the sdist path: core modules must not
        # require verl.
        code = (
            "import sys; "
            "sys.path = [p for p in sys.path if 'verl-closeout' not in p]; "
            "assert 'verl' not in sys.modules; "
            "import src.cli, src.orchestration.pi_host_execution, "
            "src.integrations.verl; "
            "print('core-import-ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("core-import-ok", proc.stdout)

    def test_orchestrator_defaults_preserved(self) -> None:
        from src.orchestration.pi_host_execution import PiHostExecutionSpec

        fields = PiHostExecutionSpec.__dataclass_fields__
        for name in (
            "model_bridge_url",
            "verified_policy_fingerprint",
            "injected_training_sequence",
        ):
            self.assertIsNone(fields[name].default)


if __name__ == "__main__":
    unittest.main()

"""Contract tests for the framework-owned (RayPPOTrainer) Pi rollout path.

Three layers are covered without a GPU:

* ``AdmittedVerlSequence.from_dict`` — the batch gate rebuilds sequences from
  persisted evidence, so a tampered or truncated file must fail closed.
* the model proxy's ability to recover native token ids and logprobs from a
  stock vLLM OpenAI response (verl's ``vLLMHttpServer`` has no bespoke
  evidence extension, unlike our own servers).
* the agent-loop / manager settings validation (skipped when verl is absent).
"""

from __future__ import annotations

import inspect
import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.capture import (
    EventWriter,
    ModelEndpointKind,
    ModelEvidenceJsonlWriter,
    ModelProxyService,
    TraceRecorder,
)
from src.capture.model_proxy_http import (
    _standard_prompt_token_ids,
    _standard_response_token_ids,
)
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError
from src.integrations.verl import AdmittedVerlSequence, SEQUENCE_ASSEMBLER_VERSION

try:  # pragma: no cover - exercised only where the framework is installed
    import ray  # noqa: F401
    import verl  # noqa: F401

    HAS_VERL = True
except Exception:  # noqa: BLE001
    HAS_VERL = False


def _admitted_payload() -> dict:
    return {
        "episode_id": "run-a0s0",
        "group_id": "run",
        "policy_fingerprint": "f" * 64,
        "prompt_ids": [1, 2, 3],
        "response_ids": [4, 5, 6, 7],
        "response_mask": [1, 1, 0, 0],
        "response_logprobs": [-0.1, -0.2, 0.0, 0.0],
        "reward": 1.0,
        "num_model_calls": 2,
        "num_tool_rounds": 1,
        "member_id": "run-a0s0",
        "execution_bundle_checksum": "b" * 64,
        "policy_artifact_checksum": "c" * 64,
        "sequence_schema_version": SEQUENCE_ASSEMBLER_VERSION,
    }


class AdmittedSequenceFromDictTest(unittest.TestCase):
    def test_round_trip_preserves_every_field(self) -> None:
        payload = _admitted_payload()
        sequence = AdmittedVerlSequence.from_dict(payload)
        self.assertEqual(sequence.to_dict(), payload)

    def test_rejects_missing_and_unknown_fields(self) -> None:
        payload = _admitted_payload()
        del payload["reward"]
        with self.assertRaises(ContractValidationError):
            AdmittedVerlSequence.from_dict(payload)
        payload = _admitted_payload()
        payload["surprise"] = 1
        with self.assertRaises(ContractValidationError):
            AdmittedVerlSequence.from_dict(payload)

    def test_rejects_misaligned_arrays(self) -> None:
        payload = _admitted_payload()
        payload["response_logprobs"] = [-0.1, -0.2]
        with self.assertRaises(ContractValidationError):
            AdmittedVerlSequence.from_dict(payload)

    def test_rejects_bad_mask_and_non_finite_logprobs(self) -> None:
        payload = _admitted_payload()
        payload["response_mask"] = [1, 2, 0, 0]
        with self.assertRaises(ContractValidationError):
            AdmittedVerlSequence.from_dict(payload)
        payload = _admitted_payload()
        payload["response_mask"] = [0, 0, 0, 0]
        with self.assertRaises(ContractValidationError):
            AdmittedVerlSequence.from_dict(payload)
        payload = _admitted_payload()
        payload["response_logprobs"] = [-0.1, float("nan"), 0.0, 0.0]
        with self.assertRaises(ContractValidationError):
            AdmittedVerlSequence.from_dict(payload)

    def test_rejects_wrong_schema_version(self) -> None:
        payload = _admitted_payload()
        payload["sequence_schema_version"] = "verl-sequence-assembler/v0"
        with self.assertRaises(ContractValidationError):
            AdmittedVerlSequence.from_dict(payload)


class FakeVllmUpstream:
    """Stand-in for verl's vLLM OpenAI server (no evidence extension)."""

    def __init__(self, payload: dict) -> None:
        requests: list[dict] = []
        self.requests = requests

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers["Content-Length"])
                requests.append(json.loads(self.rfile.read(length)))
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _vllm_style_payload(*, token_ids: bool) -> dict:
    logprobs = [
        {"token": "token_id:20", "logprob": -0.5, "top_logprobs": []},
        {"token": "token_id:21", "logprob": -0.25, "top_logprobs": []},
    ]
    choice = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "done"},
        "logprobs": {"content": logprobs},
    }
    if token_ids:
        choice["token_ids"] = [20, 21]
    payload = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "model": "pi-rollout",
        "choices": [choice],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
    }
    if token_ids:
        payload["prompt_token_ids"] = [10, 11]
    else:
        payload["prompt_logprobs"] = [{"10": 0.0}, {"11": -0.3}]
    return payload


class ProxyNativeEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, payload: dict, tools: list | None = None):
        upstream = FakeVllmUpstream(payload)
        self.addCleanup(upstream.close)
        evidence_path = self.root / "model-evidence.jsonl"
        recorder = TraceRecorder(
            EventWriter(self.root / "raw-events.jsonl"),
            run_id="run",
            episode_id="episode",
            trace_id="trace-run",
        )
        service = ModelProxyService(
            identity=ExecutionIdentity(
                run_id="run",
                task_id="Mbpp/118",
                episode_id="episode",
                attempt_id=1,
                producer_id="pi-real",
                producer_version="0.84.2",
                policy_fingerprint="f" * 64,
            ),
            endpoint_kind=ModelEndpointKind.CONTROLLED,
            upstream_chat_completions_url=upstream.url,
            evidence_writer=ModelEvidenceJsonlWriter(evidence_path),
            recorder=recorder,
            access_token="token",
        )
        status, _ = service.forward_chat_completions({
            "model": "pi-rollout",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            **({"tools": tools} if tools is not None else {}),
        })
        self.assertEqual(status, 200)
        row = json.loads(evidence_path.read_text().splitlines()[0])
        return upstream.requests[0], row["backend"]

    def test_requests_native_ids_and_logprobs(self) -> None:
        request, _ = self._run(_vllm_style_payload(token_ids=True))
        self.assertTrue(request["return_token_ids"])
        self.assertTrue(request["return_tokens_as_token_ids"])
        self.assertTrue(request["logprobs"])
        # The proxy makes a non-streaming upstream call so it can persist one
        # complete response before replaying SSE to Pi.
        self.assertFalse(request["stream"])

    def test_recovers_tool_calls_the_engine_parse_missed(self) -> None:
        # The policy emits bare tool-call JSON; vLLM's parsers need
        # <tool_call> tags, so the engine returns content and an empty
        # tool_calls list. Pi only acts on a tool-call turn, so the data plane
        # has to recover the call or every episode dies on its first turn.
        payload = _vllm_style_payload(token_ids=True)
        payload["choices"][0]["message"]["content"] = (
            "I will read it.\n"
            '{"name": "read", "arguments": {"path": "solution.py"}}\n'
        )
        payload["choices"][0]["message"]["tool_calls"] = []
        payload["choices"][0]["finish_reason"] = "length"

        request, backend = self._run(
            payload,
            tools=[
                {"type": "function", "function": {"name": "read", "parameters": {}}},
                {"type": "function", "function": {"name": "bash", "parameters": {}}},
            ],
        )
        self.assertEqual(len(request["tools"]), 2)
        choice = backend["response"]["choices"][0]
        recovered = choice["message"]["tool_calls"]
        self.assertEqual([c["function"]["name"] for c in recovered], ["read"])
        self.assertEqual(choice["finish_reason"], "tool_calls")
        # The native token stream is untouched by the protocol conversion.
        self.assertEqual(backend["response_token_ids"], [20, 21])

    def test_engine_parsed_tool_calls_are_not_second_guessed(self) -> None:
        payload = _vllm_style_payload(token_ids=True)
        payload["choices"][0]["message"]["content"] = (
            '{"name": "bash", "arguments": {"command": "ls"}}'
        )
        payload["choices"][0]["message"]["tool_calls"] = [
            {
                "id": "engine-call",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }
        ]
        payload["choices"][0]["finish_reason"] = "tool_calls"

        _, backend = self._run(
            payload,
            tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
        )
        recovered = backend["response"]["choices"][0]["message"]["tool_calls"]
        self.assertEqual([c["id"] for c in recovered], ["engine-call"])

    def test_captures_ids_from_return_token_ids(self) -> None:
        _, backend = self._run(_vllm_style_payload(token_ids=True))
        self.assertEqual(backend["prompt_token_ids"], [10, 11])
        self.assertEqual(backend["response_token_ids"], [20, 21])
        self.assertEqual(backend["response_logprobs"], [-0.5, -0.25])

    def test_falls_back_to_token_id_logprob_keys(self) -> None:
        _, backend = self._run(_vllm_style_payload(token_ids=False))
        self.assertEqual(backend["response_token_ids"], [20, 21])
        self.assertEqual(backend["prompt_token_ids"], [10, 11])

    def test_standard_helpers_reject_inconsistent_payloads(self) -> None:
        self.assertIsNone(_standard_response_token_ids({"choices": [{"token_ids": ["x"]}]}))
        self.assertIsNone(_standard_prompt_token_ids({"prompt_token_ids": ["x"]}))
        self.assertIsNone(_standard_prompt_token_ids({}))


@unittest.skipUnless(HAS_VERL, "verl/ray are not installed in this environment")
class PiLoopSettingsTest(unittest.TestCase):
    def test_agent_loop_requires_its_settings(self) -> None:
        from src.integrations.verl.pi_loop import PiAgentLoop

        loop = PiAgentLoop.__new__(PiAgentLoop)
        loop.pi_binary = ""
        loop.episode_root = "/tmp"
        loop.run_id = "run"
        loop.policy_fingerprint = "f" * 64
        loop.verifier_script = "/tmp/verify.py"
        loop.tools = ("read", "bash", "write", "edit", "ls")
        with self.assertRaises(ContractValidationError):
            loop._validate_settings()
        loop.pi_binary = "/usr/bin/pi"
        loop.tools = ("read",)
        with self.assertRaises(ContractValidationError):
            loop._validate_settings()

    def test_manager_settings_are_required(self) -> None:
        from types import SimpleNamespace

        from src.integrations.verl.verl_manager import _manager_settings

        with self.assertRaises(ContractValidationError):
            _manager_settings(SimpleNamespace())
        with self.assertRaises(ContractValidationError):
            _manager_settings(SimpleNamespace(pi_certification={"run_id": "r"}))
        settings = _manager_settings(SimpleNamespace(pi_certification={
            "run_id": "r",
            "episode_root": "/tmp/e",
            "minimum_group_size": 4,
            "max_attempts": 3,
            "policy_fingerprint_json": "/tmp/p.json",
        }))
        self.assertEqual(settings["run_id"], "r")


@unittest.skipUnless(HAS_VERL, "verl/ray are not installed in this environment")
class AutoAwaitBridgeTest(unittest.TestCase):
    """Overrides must keep the bridge that lets a sync trainer drive them.

    ``RayPPOTrainer.fit`` is synchronous and calls
    ``agent_loop_manager.generate_sequences`` without awaiting
    (ray_trainer.py:546 and :1321). The base class makes that legal by wrapping
    the coroutine function in ``verl.utils.ray_utils.auto_await``. An override
    that drops the decorator hands the caller a bare coroutine, which fails on
    the first subscript of the result.
    """

    def test_generate_sequences_keeps_the_auto_await_bridge(self) -> None:
        from verl.experimental.agent_loop.agent_loop import AgentLoopManager

        from src.integrations.verl.verl_manager import CertifiedVerlAgentLoopManager

        self.assertFalse(
            inspect.iscoroutinefunction(AgentLoopManager.generate_sequences),
            "upstream no longer uses auto_await here; revisit this override",
        )
        self.assertFalse(
            inspect.iscoroutinefunction(
                CertifiedVerlAgentLoopManager.generate_sequences
            ),
            "the override lost @auto_await, so a synchronous caller would "
            "receive an un-awaited coroutine",
        )

    def test_each_generation_call_gets_its_own_episode_root(self) -> None:
        # generate_sequences is called more than once per run (validation, then
        # one call per step) with the same tasks and the same sample tokens, so
        # two calls must never resolve to the same episode directory.
        from src.integrations.verl.verl_manager import generation_root

        roots = {generation_root("/episodes", "sess-a", n) for n in range(6)}
        self.assertEqual(len(roots), 6)
        self.assertNotEqual(
            generation_root("/episodes", "sess-a", 0),
            generation_root("/episodes", "sess-b", 0),
            "a re-run of the same round must not reuse the previous session",
        )

    def test_stamp_attempt_carries_the_call_root_and_unique_tokens(self) -> None:
        from src.integrations.verl.verl_manager import (
            PI_ROOT_KEY,
            PI_SAMPLE_KEY,
            CertifiedVerlAgentLoopManager,
        )

        class _Batch:
            def __init__(self, rows: int) -> None:
                # Padding a one-row batch to the worker count repeats that row,
                # so the tokens alone would collide without this stamping.
                self.non_tensor_batch: dict = {}
                self._rows = rows

            def __len__(self) -> int:
                return self._rows

        batch = _Batch(4)
        CertifiedVerlAgentLoopManager._stamp_attempt(batch, 0, Path("/episodes/gen-000"))
        self.assertEqual(
            list(batch.non_tensor_batch[PI_ROOT_KEY]),
            ["/episodes/gen-000"] * 4,
        )
        tokens = list(batch.non_tensor_batch[PI_SAMPLE_KEY])
        self.assertEqual(len(set(tokens)), 4)


@unittest.skipUnless(HAS_VERL, "verl/ray are not installed in this environment")
class LoopModuleScopeTest(unittest.TestCase):
    """Names the loop uses at run time must exist at module scope.

    ``run`` logged its episode target through a ``logger`` the module never
    defined. Nothing importable failed, so only a real rollout exposed it.
    """

    def test_run_references_only_defined_module_names(self) -> None:
        from src.integrations.verl import pi_loop

        self.assertTrue(hasattr(pi_loop, "logger"))
        self.assertTrue(callable(getattr(pi_loop.logger, "info", None)))

    def test_workspace_is_outside_every_episode_scan(self) -> None:
        from src.integrations.verl.pi_loop import episode_workspace

        attempt_root = Path("/episodes/gen-000/attempt-0")
        episode_root = attempt_root / "ep-1"
        workspace = episode_workspace(episode_root, "ep-1")

        # Inside the episode root: creating it would also create the episode
        # root, which the orchestrator refuses to reuse.
        self.assertNotEqual(workspace.parent, episode_root)
        self.assertNotIn(str(episode_root) + "/", str(workspace) + "/")
        # Directly inside the attempt root: the batch gate enumerates that
        # directory and would read the workspace as an episode.
        self.assertNotEqual(workspace.parent, attempt_root)
        self.assertNotEqual(episode_workspace(episode_root, "ep-1"),
                            episode_workspace(attempt_root / "ep-2", "ep-2"))

    def test_served_model_id_is_discovered_not_assumed(self) -> None:
        # verl's engine registers the policy under its own name and answers any
        # other name with 404 "The model ... does not exist", so the id the loop
        # sends has to come from the engine.
        from src.errors import ContractValidationError
        from src.integrations.verl.pi_loop import _served_model_id

        class _ModelsHandler(BaseHTTPRequestHandler):
            payload: dict = {}

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                body = json.dumps(self.payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            _ModelsHandler.payload = {"data": [{"id": "/models/qwen2.5-coder-14b-base"}]}
            self.assertEqual(_served_model_id(base), "/models/qwen2.5-coder-14b-base")

            for payload in ({"data": []}, {}, {"data": [{"id": ""}]}):
                _ModelsHandler.payload = payload
                with self.assertRaises(ContractValidationError):
                    _served_model_id(base)
        finally:
            server.shutdown()
            server.server_close()
            _ModelsHandler.payload = {}


class LauncherOverrideTest(unittest.TestCase):
    """Guards the launcher's Hydra override list.

    A shell comment placed after a trailing backslash is swallowed into the
    same logical line, which silently drops every override below it. That cost
    a debugging cycle once; this keeps it from happening again.
    """

    def setUp(self) -> None:
        self.path = (
            Path(__file__).resolve().parents[2] / "scripts" / "run_verl_pi_train.sh"
        )
        self.lines = self.path.read_text().splitlines()

    def test_no_comment_inside_a_continued_command(self) -> None:
        for index, line in enumerate(self.lines[:-1]):
            if line.rstrip().endswith("\\"):
                following = self.lines[index + 1].strip()
                self.assertFalse(
                    following.startswith("#"),
                    f"{self.path.name}:{index + 2}: comment follows a trailing "
                    "backslash and would swallow the rest of the command",
                )

    def test_required_overrides_are_present(self) -> None:
        joined = " ".join(self.lines)
        for key in (
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.n=4",
            "actor_rollout_ref.actor.ppo_mini_batch_size=1",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
            # FSDPEngineConfig defaults model_dtype to "fp32", and
            # fsdp_workers.py:387 feeds it straight to from_pretrained. Left at
            # the default, each FSDP rank materialises the whole checkpoint as
            # anonymous host memory (~55.8 GiB measured for this 14B model), and
            # two ranks plus the colocated vLLM cross Ray's 95% node-memory kill
            # threshold before the first rollout.
            "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
            # fsdp_workers.py:742 derives base_sync_done from load_format, and
            # the "dummy" default makes the pre-rollout sync (ray_trainer.py:1252)
            # all-gather the whole 14B base to CPU in every rank. Rendering a
            # real checkpoint lets vLLM load the base and the sync become a
            # LoRA delta; layered_summon additionally keeps that collection
            # per-layer and pins vLLM sleep_level to 1 so the weights are
            # offloaded rather than discarded on each resume.
            "actor_rollout_ref.rollout.load_format=safetensors",
            "actor_rollout_ref.rollout.layered_summon=true",
            # Validation runs the same tasks again before training, with
            # do_sample=False, which contradicts the frozen sampling config and
            # costs a second full set of Pi episodes.
            "trainer.val_before_train=false",
            # Pi asks for tool_choice="auto"; without both of these vLLM
            # rejects the request with 400 before the model is ever invoked.
            "actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true",
            "actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes",
            "algorithm.adv_estimator=grpo",
            "data.train_batch_size=1",
            "actor_rollout_ref.rollout.calculate_log_probs=true",
            "actor_rollout_ref.model.override_config.attn_implementation=sdpa",
            "VLLM_BATCH_INVARIANT",
            "critic.model.path=",
            "+pi_certification.policy_fingerprint_json=",
        ):
            self.assertIn(key, joined, f"launcher lost the override {key}")


if __name__ == "__main__":
    unittest.main()

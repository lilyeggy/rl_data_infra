import json
import tempfile
import unittest
from pathlib import Path

from src.framework import (
    CONSUMERS,
    DataPlane,
    ExecutionResult,
    ModelResponse,
    RunSpec,
    TaskRef,
    Verdict,
)


class DummyModel:
    def complete(self, request):
        return ModelResponse(
            text="ok",
            prompt_token_ids=(1, 2),
            response_token_ids=(3,),
            response_logprobs=(-0.1,),
        )


class DummyVerifier:
    def verify(self, result, workspace):
        return Verdict(status="PASSED", score=1.0, detail={"workspace": str(workspace)})


class DummyHarness:
    def run(self, spec):
        evidence = Path(spec.output_dir) / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        verdict = spec.verifier.verify(None, evidence)
        return ExecutionResult(
            episode_id="ep-1",
            task_id=spec.task.task_id,
            verdict=verdict,
            evidence_dir=evidence,
            model_calls=1,
            tool_rounds=0,
        )


class DummyCompiler:
    name = "SFT"

    def compile(self, result, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "sft.jsonl"
        path.write_text(json.dumps({"episode_id": result.episode_id}) + "\n")
        return {"rows": 1, "path": str(path)}


class FrameworkInterfaceTests(unittest.TestCase):
    def test_consumer_profiles_are_declared(self):
        self.assertIn("SFT", CONSUMERS)
        self.assertIn("ON_POLICY_RL", CONSUMERS)

    def test_data_plane_run_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = RunSpec(
                run_id="run-1",
                task=TaskRef(task_id="t1", statement="do it"),
                harness=DummyHarness(),
                model=DummyModel(),
                verifier=DummyVerifier(),
                consumer="SFT",
                output_dir=Path(tmp),
            )
            result = DataPlane().run(spec)
            self.assertEqual(result.verdict.status, "PASSED")
            manifest = json.loads((Path(tmp) / "run-manifest.json").read_text())
            self.assertEqual(manifest["consumer"], "SFT")
            self.assertEqual(manifest["verdict"], "PASSED")
            self.assertEqual(len(manifest["checksum"]), 64)

    def test_consumer_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            compiler = DummyCompiler()
            result = ExecutionResult(
                episode_id="ep-1",
                task_id="t1",
                verdict=Verdict(status="PASSED", score=1.0),
                evidence_dir=Path(tmp),
            )
            out = DataPlane([compiler]).compile(result, "SFT", Path(tmp) / "out")
            self.assertEqual(out["rows"], 1)
            self.assertTrue((Path(tmp) / "out" / "sft.jsonl").is_file())

    def test_unknown_consumer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = RunSpec(
                run_id="run-1",
                task=TaskRef(task_id="t1", statement="do it"),
                harness=DummyHarness(),
                model=DummyModel(),
                verifier=DummyVerifier(),
                consumer="NOPE",
                output_dir=Path(tmp),
            )
            with self.assertRaises(ValueError):
                DataPlane().run(spec)
            result = ExecutionResult(
                episode_id="ep-1",
                task_id="t1",
                verdict=Verdict(status="PASSED", score=1.0),
                evidence_dir=Path(tmp),
            )
            with self.assertRaises(KeyError):
                DataPlane().compile(result, "SFT", Path(tmp))


if __name__ == "__main__":
    unittest.main()

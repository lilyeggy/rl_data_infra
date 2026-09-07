from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from src.cli import build_parser
from tests.orchestration.test_local_execution import _spec


class LocalLauncherCliTest(unittest.TestCase):
    def test_plan_local_is_non_executing_and_requires_explicit_command(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            args = build_parser().parse_args(
                [
                    "plan-local",
                    "--run-id",
                    "run-cli-local",
                    "--task-id",
                    "task-cli-local",
                    "--episode-id",
                    "episode-cli-local",
                    "--image",
                    "example/harness",
                    "--image-digest",
                    "a" * 64,
                    "--workspace",
                    workspace,
                    "--",
                    "python",
                    "-m",
                    "harness",
                ]
            )
            with redirect_stdout(StringIO()) as output:
                result = args.func(args)
        self.assertEqual(result, 0)
        self.assertIn('"network_policy": "NONE"', output.getvalue())
        self.assertIn('"AGENT_EPISODE_ID": "episode-cli-local"', output.getvalue())

    def test_prepare_local_writes_manifest_and_plan_from_secret_free_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(_spec(str(workspace)).to_dict()))
            output_dir = root / "prepared"
            args = build_parser().parse_args(
                [
                    "prepare-local",
                    "--spec",
                    str(spec_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            with redirect_stdout(StringIO()):
                result = args.func(args)
            plan = (output_dir / "launch-plan.json").read_text()
            manifest_exists = (output_dir / "execution-run-manifest.json").is_file()
        self.assertEqual(result, 0)
        self.assertTrue(manifest_exists)
        self.assertNotIn("prepare-only-secret", plan)


if __name__ == "__main__":
    unittest.main()

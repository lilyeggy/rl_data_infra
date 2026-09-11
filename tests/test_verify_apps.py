from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.verify_apps import _stdio_text


class StdioTextTest(unittest.TestCase):
    def test_stdio_text_preserves_top_level_lines(self) -> None:
        assert _stdio_text(["1", "3", "4 7 2 9", "5 6 4 7"]) == (
            "1\n3\n4 7 2 9\n5 6 4 7"
        )

    def test_stdio_text_joins_nested_values_with_spaces(self) -> None:
        assert _stdio_text([[1, 2], [3, 4]]) == "1 2\n3 4"


class VerifyAppsSubprocessTest(unittest.TestCase):
    def test_verifier_handles_multiline_stdin_and_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_id = "apps-train-test"
            manifest = tmp_path / "manifest.json"
            workspace = tmp_path / "workspace"
            output = tmp_path / "verifier-output.json"
            workspace.mkdir()
            manifest.write_text(
                json.dumps(
                    {
                        "tasks": {
                            task_id: {
                                "input_output": {
                                    "inputs": [["2", "hello", "world"]],
                                    "outputs": [["hello", "world"]],
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (workspace / "solution.py").write_text(
                "import sys\nlines = sys.stdin.read().splitlines()\nprint(*lines[1:], sep='\\n')\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parents[1] / "scripts" / "verify_apps.py"),
                    "--manifest",
                    str(manifest),
                    "--task-id",
                    task_id,
                    "--source-worktree",
                    str(workspace),
                    "--python",
                    sys.executable,
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, result.stderr
            assert json.loads(output.read_text(encoding="utf-8"))["resolved"] is True

    def test_verifier_rejects_call_based_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_id = "apps-train-call-based"
            manifest = tmp_path / "manifest.json"
            workspace = tmp_path / "workspace"
            output = tmp_path / "verifier-output.json"
            workspace.mkdir()
            manifest.write_text(
                json.dumps(
                    {
                        "tasks": {
                            task_id: {
                                "input_output": {
                                    "fn_name": "solve",
                                    "inputs": [[[1, 2]]],
                                    "outputs": [3],
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (workspace / "solution.py").write_text(
                "def solve(values): return sum(values)\n", encoding="utf-8"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parents[1] / "scripts" / "verify_apps.py"),
                    "--manifest",
                    str(manifest),
                    "--task-id",
                    task_id,
                    "--source-worktree",
                    str(workspace),
                    "--python",
                    sys.executable,
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            assert result.returncode == 1
            assert report["resolved"] is False
            assert report["reason"] == "call-based APPS task is unsupported by the stdin/stdout verifier"


if __name__ == "__main__":
    unittest.main()

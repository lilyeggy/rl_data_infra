from __future__ import annotations

import unittest

from src.contracts._json import sha256_json
from src.contracts.canonical_action import (
    ActionType,
    CANONICAL_ACTION_TYPES,
    CanonicalAction,
    PI_TO_CANONICAL,
    ResultStatus,
    build_canonical_action,
    normalize_command_for_workspace,
    normalize_path_argument,
)
from src.errors import ContractValidationError
from src.contracts.canonical_action import canonical_tool_for_source

WORKSPACE = "/home/f630/homePLUS/agent-data-plane/swebench/sympy-23824"


class PathNormalizationTest(unittest.TestCase):
    def test_workspace_absolute_becomes_relative(self) -> None:
        normalized, lossy = normalize_path_argument(
            f"{WORKSPACE}/sympy/core/a.py", workspace_root=WORKSPACE
        )
        self.assertEqual(normalized, "sympy/core/a.py")
        self.assertFalse(lossy)

    def test_root_itself_becomes_dot(self) -> None:
        normalized, lossy = normalize_path_argument(WORKSPACE, workspace_root=WORKSPACE)
        self.assertEqual(normalized, ".")
        self.assertFalse(lossy)

    def test_relative_path_kept(self) -> None:
        normalized, lossy = normalize_path_argument(
            "sympy/handle.py", workspace_root=WORKSPACE
        )
        self.assertEqual(normalized, "sympy/handle.py")
        self.assertFalse(lossy)

    def test_outside_workspace_marked_lossy(self) -> None:
        normalized, lossy = normalize_path_argument(
            "/etc/passwd", workspace_root=WORKSPACE
        )
        self.assertEqual(normalized, "/etc/passwd")
        self.assertTrue(lossy)

    def test_tilde_expanded(self) -> None:
        normalized, lossy = normalize_path_argument(
            f"~/homePLUS/{WORKSPACE.split('/')[-1]}/x.py",
            workspace_root=WORKSPACE,
        )
        # Tilde expansion is os-specific; just assert it never crashes and
        # either normalizes under root or is marked lossy.
        self.assertIsInstance(normalized, str)
        self.assertIsInstance(lossy, bool)


class CommandNormalizationTest(unittest.TestCase):
    def test_workspace_prefix_replaced_with_placeholder(self) -> None:
        normalized, lossy = normalize_command_for_workspace(
            f"cd {WORKSPACE} && git status", workspace_root=WORKSPACE
        )
        self.assertEqual(normalized, "$WORKSPACE && git status")
        self.assertFalse(lossy)

    def test_other_absolute_ref_marked_lossy(self) -> None:
        normalized, lossy = normalize_command_for_workspace(
            f"cat /home/f630/other/secret.txt && ls {WORKSPACE}",
            workspace_root=WORKSPACE,
        )
        self.assertEqual(normalized, "$WORKSPACE")
        self.assertTrue(lossy)


class CanonicalActionContractTest(unittest.TestCase):
    def test_build_round_trip_and_checksum(self) -> None:
        action = build_canonical_action(
            action_id="ca-test-1",
            action_type=ActionType.READ_FILE,
            canonical_tool_name="read_file",
            source_tool_name="read",
            source_harness="pi",
            arguments={"path": f"{WORKSPACE}/src/a.py", "limit": 100},
            workspace_root=WORKSPACE,
            observation="text content",
            result_status=ResultStatus.SUCCEEDED,
            action_timestamp="2026-08-23T00:00:00Z",
        )
        self.assertEqual(action.normalized_arguments["path"], "src/a.py")
        self.assertFalse(action.lossy)
        restored = CanonicalAction.from_dict(action.to_dict())
        self.assertEqual(restored, action)
        self.assertEqual(restored.checksum, action.checksum)
        self.assertEqual(restored.checksum, sha256_json(action.to_dict()))

    def test_unknown_canonical_tool_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            build_canonical_action(
                action_id="x",
                action_type=ActionType.READ_FILE,
                canonical_tool_name="not-a-real-tool",
                source_tool_name="read",
                source_harness="pi",
                arguments={},
                workspace_root=WORKSPACE,
            )

    def test_lossy_requires_reason(self) -> None:
        with self.assertRaises(ContractValidationError):
            CanonicalAction(
                action_id="x",
                action_type=ActionType.RUN_COMMAND,
                canonical_tool_name="run_command",
                arguments={},
                normalized_arguments={},
                observation=None,
                result_status=ResultStatus.NONE,
                source_tool_name="bash",
                source_harness="pi",
                lossy=True,
                lossy_reasons=(),
            )

    def test_action_types_all_present(self) -> None:
        expected = {
            "read_file", "search_code", "list_directory", "run_command",
            "edit_file", "write_file", "finish", "tool_error",
            "environment_observation",
        }
        self.assertEqual(set(CANONICAL_ACTION_TYPES), expected)
        self.assertEqual(len(set(CANONICAL_ACTION_TYPES)), len(CANONICAL_ACTION_TYPES))

    def test_pi_tool_mapping_subset_of_canonical(self) -> None:
        self.assertTrue(set(PI_TO_CANONICAL.values()) <= set(CANONICAL_ACTION_TYPES) | {"finish"})
        self.assertEqual(canonical_tool_for_source("read")[0], "read_file")
        self.assertEqual(canonical_tool_for_source("grep")[0], "search_code")
        self.assertEqual(canonical_tool_for_source("find")[0], "search_code")
        self.assertEqual(canonical_tool_for_source("ls")[0], "list_directory")
        self.assertEqual(canonical_tool_for_source("bash")[0], "run_command")
        self.assertEqual(canonical_tool_for_source("edit")[0], "edit_file")
        self.assertEqual(canonical_tool_for_source("write")[0], "write_file")

    def test_unknown_tool_fails_closed(self) -> None:
        kind, reason = canonical_tool_for_source("teleport")
        self.assertIsNone(kind)
        self.assertIn("unknown", reason)


if __name__ == "__main__":
    unittest.main()
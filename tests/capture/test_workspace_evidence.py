from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.capture import (
    ArtifactStore,
    snapshot_workspace,
    store_workspace_change_evidence,
)


class WorkspaceEvidenceTest(unittest.TestCase):
    def test_stores_machine_diff_and_text_patch_without_git_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "app.py"
            source.write_text("value = 1\n")
            before = snapshot_workspace(workspace)
            source.write_text("value = 2\n")
            (workspace / "new.txt").write_text("new\n")
            refs = store_workspace_change_evidence(
                before,
                snapshot_workspace(workspace),
                store=ArtifactStore(root / "objects"),
                created_at="2026-08-22T00:00:00Z",
            )
            by_kind = {item.kind: item for item in refs}
            report = json.loads(Path(by_kind["workspace-diff"].uri).read_text())
            patch = Path(by_kind["workspace-patch"].uri).read_text()

        self.assertEqual(report["modified"][0]["path"], "app.py")
        self.assertEqual(report["added"][0]["path"], "new.txt")
        self.assertIn("-value = 1", patch)
        self.assertIn("+value = 2", patch)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from scripts.verify_swebench_selected import _resolve_selectors


class SwebenchSelectorResolutionTest(unittest.TestCase):
    def test_bare_function_selector_is_resolved_by_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tests = tmp_path / "pkg" / "tests"
            tests.mkdir(parents=True)
            (tests / "test_feature.py").write_text(
                "def test_bare_target():\n    assert True\n"
            )
            resolved, diagnostic = _resolve_selectors(
                python=sys.executable, target=tmp_path, selectors=["test_bare_target"]
            )
            assert resolved == ["pkg/tests/test_feature.py::test_bare_target"]
            assert diagnostic["bare_selector_files"] == {
                "test_bare_target": ["pkg/tests/test_feature.py"]
            }

    def test_bare_selector_expands_duplicate_definitions_conservatively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for package in ("one", "two"):
                tests = tmp_path / package
                tests.mkdir()
                (tests / "test_feature.py").write_text(
                    "def test_shared_name():\n    assert True\n"
                )
            resolved, _ = _resolve_selectors(
                python=sys.executable, target=tmp_path, selectors=["test_shared_name"]
            )
            assert resolved == [
                "one/test_feature.py::test_shared_name",
                "two/test_feature.py::test_shared_name",
            ]

    def test_unknown_bare_selector_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            resolved, diagnostic = _resolve_selectors(
                python=sys.executable, target=tmp_path, selectors=["test_missing"]
            )
            assert resolved is None
            assert diagnostic["unresolved"] == {"test_missing": []}


if __name__ == "__main__":
    unittest.main()

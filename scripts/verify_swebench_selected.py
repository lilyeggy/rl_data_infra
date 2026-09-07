#!/usr/bin/env python3
"""Fail-closed isolated verifier for one selected SWE-bench instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


_SITECUSTOMIZE = """\
import collections
import collections.abc
for _name in ('Callable', 'Iterable', 'Mapping', 'MutableMapping', 'Sequence', 'MutableSequence'):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
"""


def _test_env(target: Path, *, temporary_root: Path | None = None) -> dict[str, str]:
    compatibility = target / ".swebench-python-compat"
    compatibility.mkdir(exist_ok=True)
    (compatibility / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    additions = [str(compatibility), str(target), str(target / "src")]
    env = os.environ | {"PYTHONPATH": os.pathsep.join(additions)}
    if temporary_root is not None:
        env["TMPDIR"] = str(temporary_root)
    return env


def _materialize_environment_files(source: Path, target: Path) -> list[dict[str, str]]:
    """Copy only allowlisted install-generated files missing from a clean clone."""

    result = []
    for relative in (Path("src/_pytest/_version.py"),):
        origin = source / relative
        destination = target / relative
        if not origin.is_file() or destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
        result.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    return result


def _run_group(
    *, python: str, target: Path, selectors: list[str], name: str
) -> dict[str, object]:
    report = target / f".swebench-{name}.xml"
    temporary_root = target / f".swebench-tmp-{name}"
    temporary_root.mkdir(exist_ok=True)
    test = subprocess.run(
        [
            python,
            "-m",
            "pytest",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            f"--junitxml={report}",
            *selectors,
        ],
        cwd=target,
        env=_test_env(target, temporary_root=temporary_root),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    counts: dict[str, int] = {}
    parse_error: str | None = None
    try:
        root = ET.parse(report).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            raise ValueError("JUnit report contains no testsuite")
        counts = {
            key: int(suite.attrib.get(key, "0"))
            for key in ("tests", "failures", "errors", "skipped")
        }
    except (OSError, ET.ParseError, TypeError, ValueError) as exc:
        parse_error = str(exc)
    return {
        "name": name,
        "selectors": selectors,
        "returncode": test.returncode,
        "counts": counts,
        "report_parse_error": parse_error,
        "output_tail": (test.stdout + test.stderr)[-8000:],
    }


def _group_is_infra_invalid(group: dict[str, object]) -> bool:
    counts = group["counts"]
    assert isinstance(counts, dict)
    return bool(
        group["report_parse_error"]
        or group["returncode"] not in {0, 1}
        or counts.get("errors", 0)
        or counts.get("tests", 0) == 0
    )


def _resolve_selectors(
    *, python: str, target: Path, selectors: list[str]
) -> tuple[list[str] | None, dict[str, object]]:
    """Resolve uniquely truncated benchmark node ids against pytest collection.

    Some SWE-bench metadata node ids end at the first space in a parametrized
    display value.  Prefix recovery is safe only when collection produces
    exactly one candidate; zero or multiple matches fail closed.
    """

    bare_selector_files: dict[str, list[str]] = {}
    files: set[str] = set()
    for selector in selectors:
        if "::" in selector:
            files.add(selector.split("::", 1)[0])
            continue
        definition = re.compile(
            rf"^\s*(?:async\s+)?def\s+{re.escape(selector)}\s*\(", re.MULTILINE
        )
        candidates = []
        for path in target.rglob("test*.py"):
            if ".git" in path.parts:
                continue
            try:
                content = path.read_text(errors="replace")
            except OSError:
                continue
            if definition.search(content):
                candidates.append(path.relative_to(target).as_posix())
        bare_selector_files[selector] = sorted(candidates)
        files.update(candidates)
    collect_files = sorted(files)
    if not collect_files:
        return None, {
            "returncode": None,
            "collected_count": 0,
            "resolution": {},
            "unresolved": {selector: [] for selector in selectors},
            "bare_selector_files": bare_selector_files,
            "output_tail": "no candidate test files found",
        }
    collection_runs = [
        subprocess.run(
            [python, "-m", "pytest", "--collect-only", "-q", "--no-header", path],
            cwd=target,
            env=_test_env(target),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        for path in collect_files
    ]
    collected = tuple(dict.fromkeys(
        line.strip()
        for collect in collection_runs
        for line in collect.stdout.splitlines()
        if "::" in line and not line.startswith(("=", "ERROR"))
    ))
    collect_returncode = next(
        (collect.returncode for collect in collection_runs if collect.returncode != 0), 0
    )
    collect_output = "\n".join(
        collect.stdout + collect.stderr for collect in collection_runs
    )
    resolution: dict[str, list[str]] = {}
    unresolved: dict[str, list[str]] = {}
    for selector in selectors:
        if "::" not in selector:
            matches = [
                item for item in collected
                if item.rsplit("::", 1)[-1].split("[", 1)[0] == selector
            ]
        else:
            matches = [item for item in collected if item == selector]
            if not matches:
                matches = [item for item in collected if item.startswith(selector)]
            if not matches and "[" in selector:
                function_prefix = selector.split("[", 1)[0] + "["
                matches = [item for item in collected if item.startswith(function_prefix)]
        if matches:
            # Expanding every match is conservative: lossy benchmark metadata
            # can cause false rejection, but can never hide a regression.
            resolution[selector] = matches
        else:
            unresolved[selector] = matches
    diagnostic: dict[str, object] = {
        "returncode": collect_returncode,
        "collected_count": len(collected),
        "resolution": resolution,
        "unresolved": unresolved,
        "bare_selector_files": bare_selector_files,
        "output_tail": collect_output[-4000:],
    }
    if collect_returncode != 0 or unresolved:
        return None, diagnostic
    flattened: list[str] = []
    for selector in selectors:
        for item in resolution[selector]:
            if item not in flattened:
                flattened.append(item)
    return flattened, diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.selected.read_text())
    task = selected[args.instance_id]
    source = args.source_worktree.resolve()
    base = task["base_commit"]
    full_patch = subprocess.run(
        ["git", "-C", str(source), "diff", "--binary", base],
        capture_output=True,
        text=True,
        check=False,
    )
    evaluator_test_files = sorted(
        line[6:].split("\t", 1)[0]
        for line in task["test_patch"].splitlines()
        if line.startswith("+++ b/")
    )
    evaluation_patch = subprocess.run(
        [
            "git", "-C", str(source), "diff", "--binary", base, "--", ".",
            *(f":(exclude){path}" for path in evaluator_test_files),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    changed = subprocess.run(
        ["git", "-C", str(source), "diff", "--name-only", base],
        capture_output=True,
        text=True,
        check=False,
    )
    changed_files = tuple(line for line in changed.stdout.splitlines() if line)
    excluded_agent_files = sorted(set(changed_files) & set(evaluator_test_files))
    fail_to_pass = list(task.get("FAIL_TO_PASS", ()))
    pass_to_pass = list(task.get("PASS_TO_PASS", ()))
    result: dict[str, object] = {
        "instance_id": args.instance_id,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "evaluator_test_files": evaluator_test_files,
        "excluded_agent_files": excluded_agent_files,
        "full_patch_sha256": hashlib.sha256(full_patch.stdout.encode()).hexdigest(),
        "evaluation_patch_sha256": hashlib.sha256(
            evaluation_patch.stdout.encode()
        ).hexdigest(),
    }
    if not fail_to_pass:
        result |= {
            "execution_validity": "INFRA_INVALID",
            "resolved": None,
            "reason": "missing FAIL_TO_PASS selectors",
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 2
    with tempfile.TemporaryDirectory(prefix="swebench-verify-") as temporary:
        target = Path(temporary) / "repo"
        subprocess.run(
            ["git", "clone", "--quiet", "--no-checkout", str(source), str(target)], check=True
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--quiet", "--detach", base],
            check=True,
        )
        result["environment_materializations"] = _materialize_environment_files(
            source, target
        )
        if evaluation_patch.stdout:
            apply = subprocess.run(
                ["git", "-C", str(target), "apply", "--whitespace=nowarn", "-"],
                input=evaluation_patch.stdout,
                capture_output=True,
                text=True,
                check=False,
            )
            if apply.returncode:
                result["patch_apply_error"] = apply.stderr[-4000:]
                result["execution_validity"] = "INFRA_INVALID"
                result["resolved"] = None
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                return 2
        test_patch = target / ".swebench-test.patch"
        test_patch.write_text(task["test_patch"])
        test_patch_apply = subprocess.run(
            ["git", "-C", str(target), "apply", str(test_patch)],
            capture_output=True,
            text=True,
            check=False,
        )
        if test_patch_apply.returncode:
            result |= {
                "test_patch_apply_error": test_patch_apply.stderr[-4000:],
                "execution_validity": "INFRA_INVALID",
                "resolved": None,
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            return 2
        resolved_ftp, ftp_resolution = _resolve_selectors(
            python=args.python, target=target, selectors=fail_to_pass
        )
        resolved_ptp, ptp_resolution = _resolve_selectors(
            python=args.python, target=target, selectors=pass_to_pass
        ) if pass_to_pass else ([], {"resolution": {}, "unresolved": {}})
        result["selector_resolution"] = {
            "fail_to_pass": ftp_resolution,
            "pass_to_pass": ptp_resolution,
        }
        if resolved_ftp is None or resolved_ptp is None:
            result |= {
                "execution_validity": "INFRA_INVALID",
                "resolved": None,
                "reason": "test selector resolution was not unique",
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            return 2
        groups = [
            _run_group(
                python=args.python,
                target=target,
                selectors=resolved_ftp,
                name="fail-to-pass",
            )
        ]
        if pass_to_pass:
            groups.append(
                _run_group(
                    python=args.python,
                    target=target,
                    selectors=resolved_ptp,
                    name="pass-to-pass",
                )
            )
        infra_invalid = any(_group_is_infra_invalid(group) for group in groups)
        resolved = None if infra_invalid else all(group["returncode"] == 0 for group in groups)
        result |= {
            "patch_present": bool(evaluation_patch.stdout.strip()),
            "full_workspace_patch_present": bool(full_patch.stdout.strip()),
            "groups": groups,
            "execution_validity": "INFRA_INVALID" if infra_invalid else "VALID",
            "resolved": resolved,
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 2 if result["resolved"] is None else 0 if result["resolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

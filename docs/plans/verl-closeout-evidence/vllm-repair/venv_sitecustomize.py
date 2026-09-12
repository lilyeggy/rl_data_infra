"""Venv-local fix for the shared conda env's broken editable vLLM install.

The shared env's metadata records an editable install whose target
(/home/cxr1/vllm) was deleted, and its .pth installs a meta-path finder that
fails before any path-based lookup. A built source tree survives at
/home/cxr/ds4-deploy/vllm. site.py imports this module after all .pth files are
processed, so this is the earliest point where the finder can be neutralized —
and it applies to every process using this venv, including vLLM's subprocesses.

Scope: this venv only. The shared conda environment is not modified.
"""

import sys

_TREE = "/home/cxr/ds4-deploy/vllm"


def _disable_broken_vllm_finder():
    kept = []
    for finder in sys.meta_path:
        module = (getattr(finder, "__module__", "") or "").lower()
        if "editable" in module and "vllm" in module:
            continue  # points at the deleted /home/cxr1/vllm
        kept.append(finder)
    sys.meta_path = kept


def _prefer_built_tree():
    if _TREE not in sys.path:
        sys.path.insert(0, _TREE)


_disable_broken_vllm_finder()
_prefer_built_tree()

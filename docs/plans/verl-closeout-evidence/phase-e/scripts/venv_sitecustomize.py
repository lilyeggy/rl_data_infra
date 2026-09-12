"""Venv-local fixes for the shared conda env's broken editable vLLM install.

Three problems, all repaired without touching the shared conda environment:

1. Broken editable install. The shared env's metadata records an editable
   install whose target (/home/cxr1/vllm) was deleted, and its .pth installs a
   meta-path finder that raises before any path-based lookup. A built source
   tree survives at /home/cxr/ds4-deploy/vllm.
2. Stale version metadata. That surviving tree was built from an untagged repo,
   so setuptools_scm baked ``0.1.dev17320+geac9e008a`` into ``_version.py``.
   Consumers (verl v0.7.1) gate behaviour on ``vllm.__version__``; parsed as
   0.1.dev, every gate selects pre-0.11 code paths that do not exist in this
   tree (e.g. ``from vllm.utils import FlexibleArgumentParser``), so the import
   fails outright. The tree's actual module layout matches vLLM >= 0.13.0
   (``vllm.utils.argparse_utils`` and
   ``vllm.entrypoints.openai.parser.harmony_utils`` both exist, while the
   0.12-only ``vllm.entrypoints.harmony_utils`` does not).
3. The fix must reach vLLM's own helper subprocesses, which re-exec Python.

site.py imports this module after all .pth files are processed, so this is the
earliest point where the finder can be neutralized and the version corrected.

Scope: this venv only. The shared conda environment is not modified.
"""

import sys

_TREE = "/home/cxr/ds4-deploy/vllm"

# Truthful series for the deployed tree, inferred from its module layout.
_VLLM_VERSION_OVERRIDE = "0.13.0"


def _disable_broken_vllm_finder():
    kept = []
    for finder in sys.meta_path:
        module = (getattr(finder, "__module__", "") or "").lower()
        if "editable" in module and "vllm" in module:
            continue  # points at the deleted /home/cxr1/vllm
        kept.append(finder)
    sys.meta_path = kept


def _prefer_built_tree():
    # Insert the tree *after* this venv's site-packages, not at the front: the
    # venv carries a vllm-0.13.0.dist-info, and importlib.metadata scans
    # sys.path in order. At the front, the tree's own vllm.egg-info (built from
    # the same untagged repo) would win and report 0.1.dev1 to verl's
    # third_party gate.
    if _TREE in sys.path:
        sys.path.remove(_TREE)
    import sysconfig

    purelib = sysconfig.get_paths()["purelib"]
    try:
        idx = sys.path.index(purelib)
    except ValueError:
        idx = max(len(sys.path) - 1, 0)
    sys.path.insert(idx + 1, _TREE)


def _rewrite_version_metadata(module):
    # Only the dunder attributes: on the vllm package `version` is also the
    # name bound to the vllm.version submodule, and rewriting it would shadow
    # the module with a string.
    if hasattr(module, "__version__"):
        try:
            setattr(module, "__version__", _VLLM_VERSION_OVERRIDE)
        except Exception:
            pass
    for attr in ("__version_tuple__", "version_tuple"):
        if hasattr(module, attr):
            try:
                setattr(module, attr, (0, 13, 0))
            except Exception:
                pass


class _VersionLoader:
    """Wrap vLLM's loader so the stale scm version is corrected after exec."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        for name in ("vllm", "vllm.version", "vllm._version"):
            loaded = sys.modules.get(name)
            if loaded is not None:
                _rewrite_version_metadata(loaded)


class _VersionFinder:
    """Post-import hook for the vllm package, delegating the real lookup."""

    def find_spec(self, name, path=None, target=None):
        if name != "vllm":
            return None
        import importlib.machinery

        sys.meta_path.remove(self)
        try:
            spec = importlib.machinery.PathFinder.find_spec(name, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _VersionLoader(spec.loader)
        return spec


_disable_broken_vllm_finder()
_prefer_built_tree()
sys.meta_path.insert(0, _VersionFinder())

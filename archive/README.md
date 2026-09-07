# Archive index

This directory preserves retired implementation experiments. Archived files are
not supported runtime entrypoints and must not be used as evidence that the
current project provides a certified training path.

## Archived in this refactor

| Path | Why it was archived | Current replacement |
|---|---|---|
| `archive/experiments/local_model/` | Ad-hoc model server, SFT, and custom GRPO scripts bypassed the canonical certification/data path. The GRPO path did not capture exact behavior-policy log probabilities. | `src/training/`, the new producer boundary, and future Polar + Slime integration |
| `archive/experiments/swebench/` | Host-specific benchmark runners mixed environment setup, rollout, verification, and training concerns. | `src/evaluation/swebench.py`, future task/environment adapters, and versioned workflows |
| `archive/scripts/local-model/` | Hardware- and host-specific launch/migration scripts referenced the retired local-model experiments. | Future versioned deployment profiles under `configs/` and `orchestration/` |
| `archive/scripts/host-specific/` | Scripts contained hard-coded remote hosts/paths and were unsafe outside the retired lab setup. Embedded credentials were removed during archival. | Explicit environment-specific deployment tooling, created only when a live target is in scope |
| `archive/scripts/two-blackwell/` | Megatron/SGLang smoke and evidence tools were tied to a retired 2×96G Blackwell machine. | A new A6000 evidence capture and upstream lock after live validation |
| `archive/configs/two-blackwell/` | Project and dependency locks described the retired two-GPU machine and contradicted the current A6000 constraint. | `configs/project.yaml`; a future evidence-backed `configs/upstream-lock.yaml` |
| `docs/archive/project-history/` | Old root docs, release notes, and runbooks describe completed prototypes rather than the current recommended architecture. | `README.md`, `PROJECT_SCOPE.md`, and current ADRs |
| `archive/experiments/v1-v7-pi-tool-protocol.md` | v1–v7 Qwen/Pi tool-protocol SFT experiments. Harness-specific; not a general agent capability. | The canonical contract (`docs/canonical-agent-contract.md`), multi-view exporters (`src/exporters/`), and layered evaluation |

## Rules

1. Archive content is read-only history. Bug fixes belong in the current implementation.
2. No production module or test may import from `archive/`.
3. Historical datasets are not certified datasets unless a current dataset manifest explicitly includes them.
4. Restoring an archived component requires a new ADR and a current end-to-end test.

Historical executable scenarios that are still imported by regression tests live
under `examples/legacy_scenarios/`; they are intentionally outside `src/` and the
production CLI.

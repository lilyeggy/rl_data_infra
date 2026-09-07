"""HISTORICAL EXAMPLE: offline training views from V2.1 live fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.assembly.episode_assembler import EpisodeAssembler, EpisodeContext
from src.capture.pi_adapter import (
    PiJsonAdapter,
    PiRunConfig,
    read_pi_ndjson,
)
from src.contracts._json import sha256_json
from src.contracts.agent_episode import EpisodeVerifierStatus, ExecutionValidity, TaskStatus
from src.exporters.training_view import (
    TRAINING_VIEW_PROJECTION_VERSION,
    build_preference_pairs_with_reasons,
    project_episode_to_rollout,
)
from examples.legacy_scenarios.real_pi_v2 import verify_reference_answer
from examples.legacy_scenarios.real_pi_v21_live import (
    LIVE_RUN_IDS,
    build_context,
)
from src.contracts.experiment import ExperimentManifest


FIXTURES = (
    Path(__file__).parents[2] / "tests" / "fixtures" / "pi" / "v2.1-live"
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def generate_v24_offline_training_view(output_dir: str | Path) -> dict[str, Any]:
    """Project real live traces into SFT rollouts and preference pairs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentManifest(
        experiment_id="experiment-v24-offline-training-view",
        revision="r1",
        task_dataset_revision="pi-live-error-recovery-suite/r1",
        control_run_id=LIVE_RUN_IDS["control"],
        candidate_run_id=LIVE_RUN_IDS["candidate"],
        target_policy_flag="OBSERVED_TOOL_ERROR_LOOP",
        minimum_pairs=3,
        metadata={"source": "tests/fixtures/pi/v2.1-live", "model": "deepseek-v4-flash"},
    )

    adapters: dict[str, Any] = {}
    contexts: dict[str, EpisodeContext] = {}
    for task_index in (1, 2, 3):
        for variant in ("control", "candidate"):
            episode_id = f"episode-v21-live-{variant}-{task_index}"
            path = FIXTURES / f"{variant}-{task_index}.ndjson"
            records, issues = read_pi_ndjson(path.read_text())
            if issues:
                raise RuntimeError(f"fixture has issues: {path}: {issues}")
            declaration, _ = verify_reference_answer(records, task_index=task_index)
            adapter = PiJsonAdapter().convert(
                records,
                run_id=LIVE_RUN_IDS[variant],
                episode_id=episode_id,
                trace_id=f"trace-v21-live-{variant}-{task_index}",
                config=PiRunConfig(model="deepseek-v4-flash"),
                declared_outcome=declaration,
                normalize_tool_errors=True,
            )
            adapters[episode_id] = adapter
            contexts[episode_id] = build_context(
                task_index=task_index,
                variant=variant,
                experiment=experiment,
                adapter=adapter,
            )

    all_events = tuple(event for adapter in adapters.values() for event in adapter.events)
    assembly = EpisodeAssembler().assemble(all_events, contexts=contexts)
    if assembly.warnings or assembly.quarantined_events:
        raise RuntimeError(f"assembly failed: {assembly.warnings}")
    episodes = tuple(assembly.episodes)
    from src.validation.episode_semantics import certify_episode

    certifications = {e.episode_id: certify_episode(e) for e in episodes}
    _write_json(
        output / "episode-certifications.json",
        [c.to_dict() for c in certifications.values()],
    )
    rollouts = tuple(project_episode_to_rollout(episode) for episode in episodes)
    # Preference pairs require identical task/revision/model/policy/env/evaluator/
    # tool-schema identity.  The V2.1 control/candidate arms are a HARNESS A/B
    # (different recovery policy => different tool schema), so they must NOT be
    # paired as DPO chosen/rejected.  We report stable rejection reasons instead
    # of the legacy unverified "same task and model" claim.
    pairs, pair_rejections = build_preference_pairs_with_reasons(
        episodes, pair_key_of=lambda episode: episode.task_id
    )

    _write_json(
        output / "rollout-records.jsonl",
        [record.to_dict() for record in rollouts],
    )
    _write_json(
        output / "preference-pairs.json", [pair.to_dict() for pair in pairs]
    )
    _write_json(
        output / "preference-pair-rejections.json",
        [item.to_dict() for item in pair_rejections],
    )
    summary = {
        "release": "v2.4-offline-training-view",
        "scope": "offline SFT/preference projections from real live traces",
        "projection_version": TRAINING_VIEW_PROJECTION_VERSION,
        "episode_count": len(episodes),
        "rollout_count": len(rollouts),
        "sft_eligible_rollouts": sum(
            record.rollout_status.value == "completed" for record in rollouts
        ),
        "preference_pair_count": len(pairs),
        "preference_pair_rejections": [item.to_dict() for item in pair_rejections],
        "preference_pair_note": (
            "control/candidate arms differ in harness policy (tool schema), so no "
            "DPO pair is formed; rejecting the legacy 'same task and model' claim"
        ),
        "on_policy_rl_eligible": 0,
        "missing_for_on_policy": [
            "target-policy sampled token ids",
            "target-policy logprobs",
            "trainable mask",
        ],
        "capability_note": (
            "token_ids/logprobs/mask are None; consumers requiring them must use "
            "the capability gate and will fail explicitly"
        ),
        "certification_note": (
            "SFT rollouts require verifier-certified episodes; see "
            "episode-certifications.json"
        ),
        "evidence_checksums": {
            "episodes": sha256_json([episode.to_dict() for episode in episodes]),
            "rollouts": sha256_json([record.to_dict() for record in rollouts]),
            "pairs": sha256_json([pair.to_dict() for pair in pairs]),
        },
        "claim_boundary": (
            "offline teacher candidates and preference pairs; not on-policy RL rollouts"
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary

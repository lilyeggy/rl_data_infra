"""Certified admission boundary in front of the verl AgentLoop bridge."""

from src.integrations.verl.admission import (
    VERL_ADMISSION_VERSION,
    AdmittedVerlSequence,
    VerlAdmissionBatch,
    admit_on_policy_manifest,
)
from src.integrations.verl.bridge import (
    BRIDGE_VERSION,
    BridgeCallRecord,
    build_per_call_segments,
    to_agent_loop_output_dict,
)
from src.integrations.verl.lifecycle import (
    RUN_CLEANUP_VERSION,
    RunCleanup,
    popen_process_group,
)
from src.integrations.verl.manager import (
    CERTIFIED_LOOP_MANAGER_VERSION,
    CertifiedAgentLoopManager,
    CertifiedBatch,
    PiAgentLoopConfig,
)
from src.integrations.verl.sequence import (
    SEQUENCE_ASSEMBLER_VERSION,
    AssembledSequence,
    assemble_episode_sequence,
)

__all__ = [
    "VERL_ADMISSION_VERSION",
    "AdmittedVerlSequence",
    "VerlAdmissionBatch",
    "admit_on_policy_manifest",
    "BRIDGE_VERSION",
    "BridgeCallRecord",
    "build_per_call_segments",
    "to_agent_loop_output_dict",
    "RUN_CLEANUP_VERSION",
    "RunCleanup",
    "popen_process_group",
    "CERTIFIED_LOOP_MANAGER_VERSION",
    "CertifiedAgentLoopManager",
    "CertifiedBatch",
    "PiAgentLoopConfig",
    "SEQUENCE_ASSEMBLER_VERSION",
    "AssembledSequence",
    "assemble_episode_sequence",
]

"""Certified admission boundary in front of the upstream Polar–Slime bridge."""

from src.integrations.slime.admission import (
    SLIME_ADMISSION_VERSION,
    AdmittedPolicyTrace,
    SlimeAdmissionBatch,
    admit_on_policy_manifest,
)

__all__ = [
    "SLIME_ADMISSION_VERSION",
    "AdmittedPolicyTrace",
    "SlimeAdmissionBatch",
    "admit_on_policy_manifest",
]

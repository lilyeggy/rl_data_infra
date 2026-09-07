"""Training trust-boundary primitives: policy identity and unified eligibility."""

from src.training.eligibility import (
    ELIGIBILITY_VERSION,
    TrainingEligibility,
    TrainingEligibilityStatus,
    evaluate_training_eligibility,
)
from src.training.policy_fingerprint import (
    POLICY_FINGERPRINT_VERSION,
    PolicyFingerprint,
)

__all__ = [
    "ELIGIBILITY_VERSION",
    "POLICY_FINGERPRINT_VERSION",
    "PolicyFingerprint",
    "TrainingEligibility",
    "TrainingEligibilityStatus",
    "evaluate_training_eligibility",
]

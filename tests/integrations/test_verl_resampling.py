"""Contract tests for the resampling policy behind the framework-side gate.

The bug this guards against is silent. A retry loop that re-raises inside its
own ``except`` block is indistinguishable from a single-shot run at the call
site: ``max_attempts`` is read, echoed into the config, and never consulted. It
shipped that way through five GPU rounds, and it is what turned one
variance-free batch (smoke26, 0/16 solved) into a dead run.
"""

from __future__ import annotations

import unittest

from src.errors import ContractValidationError, DegenerateBatchError
from src.integrations.verl.resampling import certify_with_resampling


class ResamplingTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_degenerate_batch_is_redrawn(self) -> None:
        drawn: list[int] = []
        refusals: list[tuple[int, str]] = []

        async def generate(attempt: int) -> str:
            drawn.append(attempt)
            return f"batch-{attempt}"

        def certify(batch: str, attempt: int) -> str:
            if attempt == 0:
                raise DegenerateBatchError("group 'g0' has no intra-group reward variance")
            return f"certificate-{batch}"

        result = await certify_with_resampling(
            max_attempts=3,
            run_id="run",
            generate=generate,
            certify=certify,
            on_retry=lambda attempt, exc: refusals.append((attempt, str(exc))),
        )

        self.assertEqual(drawn, [0, 1], "the refused draw must be replaced, not reused")
        self.assertEqual(result.attempt, 1)
        self.assertEqual(result.batch, "batch-1")
        self.assertEqual(result.certificate, "certificate-batch-1")
        self.assertEqual(refusals, [(0, "group 'g0' has no intra-group reward variance")])

    async def test_a_deterministic_violation_is_not_retried(self) -> None:
        drawn: list[int] = []

        async def generate(attempt: int) -> str:
            drawn.append(attempt)
            return "batch"

        def certify(batch: str, attempt: int) -> str:
            raise ContractValidationError("policy artifact does not bind actual inference context")

        with self.assertRaisesRegex(
            ContractValidationError, "run: batch rejected: policy artifact does not bind"
        ):
            await certify_with_resampling(
                max_attempts=3, run_id="run", generate=generate, certify=certify
            )

        # A deterministic violation reproduces on every draw, so a second draw
        # would buy nothing and would report the same failure three times.
        self.assertEqual(drawn, [0], "a deterministic violation must fail on its first attempt")

    async def test_a_verdictless_episode_is_not_retried(self) -> None:
        # The retryable class is deliberately narrow: only a flat reward draws
        # again. Anything else about the evidence is a defect, not luck.
        def certify(batch: str, attempt: int) -> str:
            raise ContractValidationError("episode has no terminating verifier verdict")

        async def generate(attempt: int) -> str:
            return "batch"

        with self.assertRaisesRegex(ContractValidationError, "batch rejected"):
            await certify_with_resampling(
                max_attempts=3, run_id="run", generate=generate, certify=certify
            )

    async def test_exhausting_attempts_names_the_last_error(self) -> None:
        drawn: list[int] = []

        async def generate(attempt: int) -> str:
            drawn.append(attempt)
            return "batch"

        def certify(batch: str, attempt: int) -> str:
            raise DegenerateBatchError(f"flat group on draw {attempt}")

        with self.assertRaisesRegex(
            ContractValidationError,
            "no attempt produced a certifiable batch after 2 attempts; "
            "last error: flat group on draw 1",
        ):
            await certify_with_resampling(
                max_attempts=2, run_id="run", generate=generate, certify=certify
            )

        self.assertEqual(drawn, [0, 1], "every configured attempt must actually be spent")

    async def test_a_single_attempt_never_retries(self) -> None:
        drawn: list[int] = []

        async def generate(attempt: int) -> str:
            drawn.append(attempt)
            return "batch"

        def certify(batch: str, attempt: int) -> str:
            raise DegenerateBatchError("flat group")

        with self.assertRaises(ContractValidationError):
            await certify_with_resampling(
                max_attempts=1, run_id="run", generate=generate, certify=certify
            )

        self.assertEqual(drawn, [0])

    async def test_attempt_budget_must_be_positive(self) -> None:
        async def generate(attempt: int) -> str:
            raise AssertionError("must not roll out with a zero attempt budget")

        def certify(batch: str, attempt: int) -> str:
            raise AssertionError("must not certify with a zero attempt budget")

        with self.assertRaisesRegex(ContractValidationError, "max_attempts must be at least 1"):
            await certify_with_resampling(
                max_attempts=0, run_id="run", generate=generate, certify=certify
            )

    async def test_a_certified_first_attempt_is_not_redrawn(self) -> None:
        drawn: list[int] = []

        async def generate(attempt: int) -> str:
            drawn.append(attempt)
            return "batch"

        def certify(batch: str, attempt: int) -> tuple[str, int]:
            return (batch, attempt)

        result = await certify_with_resampling(
            max_attempts=3, run_id="run", generate=generate, certify=certify
        )

        self.assertEqual(drawn, [0])
        self.assertEqual(result.certificate, ("batch", 0))


class RetryableErrorTypeTest(unittest.TestCase):
    """The gate must raise the retryable type, or the policy above never fires."""

    def test_degenerate_batch_error_stays_a_contract_violation(self) -> None:
        # Callers that only guard ContractValidationError must keep working.
        self.assertTrue(issubclass(DegenerateBatchError, ContractValidationError))

    def test_variance_gates_raise_the_retryable_type(self) -> None:
        from src.integrations.verl.admission import _require_reward_variance

        class _Sequence:
            def __init__(self, reward: float) -> None:
                self.reward = reward

        with self.assertRaises(DegenerateBatchError):
            _require_reward_variance([_Sequence(0.0), _Sequence(0.0)])
        _require_reward_variance([_Sequence(0.0), _Sequence(1.0)])


if __name__ == "__main__":
    unittest.main()

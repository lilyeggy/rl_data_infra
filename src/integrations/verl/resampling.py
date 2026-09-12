"""Retry policy for a rollout batch that sampled no learning signal.

The gate in ``manager.py`` refuses a group whose samples all earned the same
reward, because GRPO's advantage is the group-relative spread and a flat group
has none. Refusing it is correct; *aborting the run* for it is not. Nothing was
consumed, so the honest response is a fresh draw.

This module holds that policy on its own, away from Ray and verl, for two
reasons. It is the only part of the framework-side manager that has a decision
worth testing without a GPU, and the failure it guards against is silent: an
attempt loop that re-raises inside its own ``except`` looks like a retry loop
and behaves like a single shot, which is exactly what shipped before this --
``max_attempts`` was configuration that could not take effect, and a
variance-free batch ended the run.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from src.errors import (
    ContractValidationError,
    DegenerateBatchError,
    RetryableRolloutError,
)


@dataclass(frozen=True, slots=True)
class CertifiedAttempt:
    """The attempt that certified, with the batch it certified."""

    attempt: int
    batch: Any
    certificate: Any


async def certify_with_resampling(
    *,
    max_attempts: int,
    run_id: str,
    generate: Callable[[int], Awaitable[Any]],
    certify: Callable[[Any, int], Any],
    on_retry: Callable[[int, Exception], None] | None = None,
) -> CertifiedAttempt:
    """Generate and certify one batch, redrawing a degenerate one.

    ``generate(attempt)`` must stamp the attempt number before it rolls out, so
    the redraw writes to its own episode directory instead of colliding with
    the refused attempt's; the attempt index is also what makes the redraw
    auditable after the fact.

    Only :class:`DegenerateBatchError` and :class:`RetryableRolloutError` are retried. A deterministic violation
    raises on its first occurrence, because retrying it would spend a full
    rollout to reproduce the same failure and would report the symptom three
    times instead of once.
    """
    if max_attempts < 1:
        raise ContractValidationError("max_attempts must be at least 1")
    last_error: ContractValidationError | None = None
    for attempt in range(max_attempts):
        try:
            batch = await generate(attempt)
            certificate = certify(batch, attempt)
        except (DegenerateBatchError, RetryableRolloutError) as exc:
            last_error = exc
            if on_retry is not None:
                on_retry(attempt, exc)
            continue
        except ContractValidationError as exc:
            raise ContractValidationError(f"{run_id}: batch rejected: {exc}") from exc
        return CertifiedAttempt(attempt=attempt, batch=batch, certificate=certificate)
    raise ContractValidationError(
        f"{run_id}: no attempt produced a certifiable batch after {max_attempts} "
        f"attempts; last error: {last_error}"
    )


__all__ = ["CertifiedAttempt", "certify_with_resampling"]

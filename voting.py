"""Small, reusable quorum voting over normalized immutable values."""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Generic, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Hashable)


class NoQuorumError(RuntimeError):
    """Raised when no single normalized value reaches the configured quorum."""


@dataclass(frozen=True, slots=True)
class VotingResult(Generic[T]):
    """The outcome and small set of metrics from one sequential vote."""

    value: T | None
    attempts: int
    quorum: int
    valid_votes: int
    rejected_votes: int
    distinct_values: int
    elapsed_seconds: float
    last_error: Exception | None = field(default=None, repr=False, compare=False)

    @property
    def disagreement(self) -> bool:
        return self.distinct_values > 1


def validate_voting_config(*, attempts: int, quorum: int) -> None:
    """Require explicit positive integers and a reachable quorum."""

    if type(attempts) is not int or attempts <= 0:
        raise ValueError("attempts must be a positive integer")
    if type(quorum) is not int or quorum <= 0:
        raise ValueError("quorum must be a positive integer")
    if quorum > attempts:
        raise ValueError("quorum must not exceed attempts")


def quorum_vote(
    cast_vote: Callable[[], T],
    *,
    attempts: int,
    quorum: int,
    label: str,
) -> VotingResult[T]:
    """Run independent votes sequentially and compare whole normalized values.

    Failed calls, invalid outputs, and unhashable values are rejected rather
    than added to the count.  A result is accepted only when exactly one value
    reaches quorum; competing quorum winners are not resolved heuristically.
    """

    validate_voting_config(attempts=attempts, quorum=quorum)
    counts: Counter[T] = Counter()
    rejected_votes = 0
    last_error: Exception | None = None
    started = time.perf_counter()

    for vote_number in range(1, attempts + 1):
        try:
            value = cast_vote()
            hash(value)
        except Exception as error:
            rejected_votes += 1
            last_error = error
            logger.warning(
                "%s vote rejected: vote=%d/%d error_type=%s",
                label,
                vote_number,
                attempts,
                type(error).__name__,
            )
            continue
        counts[value] += 1

    elapsed_seconds = time.perf_counter() - started
    winners = [value for value, count in counts.items() if count >= quorum]
    accepted = winners[0] if len(winners) == 1 else None
    distinct_values = len(counts)

    if distinct_values > 1:
        logger.warning(
            "%s vote disagreement: distinct_values=%d valid_votes=%d "
            "quorum=%d accepted=%s",
            label,
            distinct_values,
            sum(counts.values()),
            quorum,
            accepted is not None,
        )
    if accepted is None:
        logger.error(
            "%s vote rejected: outcome=no_quorum valid_votes=%d "
            "rejected_votes=%d distinct_values=%d quorum=%d attempts=%d",
            label,
            sum(counts.values()),
            rejected_votes,
            distinct_values,
            quorum,
            attempts,
        )

    return VotingResult(
        value=accepted,
        attempts=attempts,
        quorum=quorum,
        valid_votes=sum(counts.values()),
        rejected_votes=rejected_votes,
        distinct_values=distinct_values,
        elapsed_seconds=elapsed_seconds,
        last_error=last_error,
    )


__all__ = [
    "NoQuorumError",
    "VotingResult",
    "quorum_vote",
    "validate_voting_config",
]

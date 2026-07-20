"""Rank fusion utilities.

Reciprocal Rank Fusion (RRF) combines several ranked result lists into one,
using only rank positions — no fragile per-signal score weighting. This
replaces the old ad-hoc ``0.8*clip + 0.2*text`` blend in the visual retriever
(pattern borrowed from the ashleyashok dual-path search reference).

RRF score for an item::

    score(item) = sum_over_lists( 1 / (k + rank_in_list) )

where ``rank_in_list`` is 1-based and ``k`` (default 60) damps the influence
of top ranks so many-list agreement matters more than any single #1.

Pure-Python and dependency-free so it is trivially unit-testable.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[T]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[T, float]]:
    """Fuses multiple ranked lists into one via RRF.

    Args:
        ranked_lists: Each inner sequence is a list of item ids ordered best
            → worst. Items may appear in any subset of the lists.
        k: RRF damping constant (higher = flatter contribution from top ranks).
        top_k: If given, truncate the fused result to this many items.

    Returns:
        A list of ``(item, fused_score)`` sorted by descending score. Ties
        are broken deterministically by first appearance order.

    Raises:
        ValueError: If ``k`` is negative.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    scores: dict[T, float] = {}
    first_seen: dict[T, int] = {}
    order: int = 0

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1

    fused: list[tuple[T, float]] = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], first_seen[kv[0]]),
    )
    if top_k is not None:
        fused = fused[:top_k]
    return fused

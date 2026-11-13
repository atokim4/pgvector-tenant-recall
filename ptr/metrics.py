"""Recall math for filtered top-k search.

The ground truth is exact search restricted to the tenant. A tenant with fewer
rows than k can only ever return all of its rows, so the expected count is
min(k, tenant_size). Ties at the k-th distance are common with real embeddings
(duplicate chunks), so any returned row whose true distance is within `eps` of
the k-th exact distance counts as a hit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class QueryScore:
    expected: int  # min(k, tenant_size)
    returned: int  # rows the query actually returned
    hits: int  # returned rows that belong to the exact top-k (tie aware)
    wrong_tenant: int  # returned rows that belong to another tenant

    @property
    def recall(self) -> float:
        return 1.0 if self.expected == 0 else self.hits / self.expected

    @property
    def short(self) -> bool:
        """True when the query returned fewer rows than the tenant could supply."""
        return self.returned < self.expected


def score_query(
    returned_ids: Sequence[int],
    truth: Sequence[tuple[int, float]],
    k: int,
    true_distance: Mapping[int, float],
    tenant_ids: set[int] | None = None,
    eps: float = 1e-5,
) -> QueryScore:
    """Score one query.

    returned_ids: ids in the order the database returned them.
    truth: exact (id, distance) pairs for the tenant, sorted by distance;
        at least min(k, tenant_size) entries.
    true_distance: exact distance for every id the database may return.
    tenant_ids: ids owned by the tenant; anything else counts as a leak.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    expected = min(k, len(truth))
    if expected == 0:
        return QueryScore(0, len(returned_ids), 0, len(returned_ids))
    kth = truth[expected - 1][1]
    top = {i for i, _ in truth[:expected]}
    seen: set[int] = set()
    hits = 0
    wrong = 0
    for rid in returned_ids[:k]:
        if rid in seen:
            continue
        seen.add(rid)
        if tenant_ids is not None and rid not in tenant_ids:
            wrong += 1
            continue
        if rid in top or true_distance.get(rid, float("inf")) <= kth + eps:
            hits += 1
    return QueryScore(expected, len(returned_ids), min(hits, expected), wrong)


@dataclass(frozen=True)
class BucketSummary:
    queries: int
    mean_recall: float
    min_recall: float
    short_share: float  # share of queries that returned fewer rows than expected
    zero_share: float  # share of queries that returned nothing at all
    mean_returned: float
    expected: float
    wrong_tenant: int


def summarize(scores: Sequence[QueryScore]) -> BucketSummary:
    if not scores:
        raise ValueError("no scores to summarize")
    return BucketSummary(
        queries=len(scores),
        mean_recall=mean(s.recall for s in scores),
        min_recall=min(s.recall for s in scores),
        short_share=sum(s.short for s in scores) / len(scores),
        zero_share=sum(s.returned == 0 and s.expected > 0 for s in scores) / len(scores),
        mean_returned=mean(s.returned for s in scores),
        expected=mean(s.expected for s in scores),
        wrong_tenant=sum(s.wrong_tenant for s in scores),
    )


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 100]."""
    if not values:
        raise ValueError("no values")
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * p // 100))  # ceil without floats drifting
    return ordered[int(rank) - 1]

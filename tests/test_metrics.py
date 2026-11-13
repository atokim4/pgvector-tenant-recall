import pytest

from ptr.metrics import QueryScore, percentile, score_query, summarize


def truth_of(*pairs):
    return list(pairs), dict(pairs)


def test_perfect_result():
    truth, dist = truth_of((1, 0.1), (2, 0.2), (3, 0.3))
    s = score_query([1, 2], truth, k=2, true_distance=dist)
    assert s == QueryScore(expected=2, returned=2, hits=2, wrong_tenant=0)
    assert s.recall == 1.0 and not s.short


def test_small_tenant_expects_only_its_rows():
    truth, dist = truth_of((7, 0.4))
    full = score_query([7], truth, k=10, true_distance=dist)
    empty = score_query([], truth, k=10, true_distance=dist)
    assert full.expected == 1 and full.recall == 1.0 and not full.short
    assert empty.recall == 0.0 and empty.short


def test_short_result_counts_missing_rows():
    truth, dist = truth_of(*[(i, i / 100) for i in range(1, 21)])
    s = score_query([1, 2, 3], truth, k=10, true_distance=dist)
    assert s.expected == 10 and s.hits == 3 and s.recall == pytest.approx(0.3) and s.short


def test_wrong_neighbour_is_a_miss():
    truth, dist = truth_of((1, 0.1), (2, 0.2), (3, 0.3))
    s = score_query([1, 3], truth, k=2, true_distance=dist)
    assert s.hits == 1 and s.recall == 0.5 and not s.short


def test_tie_at_kth_distance_counts_as_hit():
    truth, dist = truth_of((1, 0.1), (2, 0.2), (3, 0.2))
    s = score_query([1, 3], truth, k=2, true_distance=dist)
    assert s.recall == 1.0


def test_other_tenant_rows_are_leaks_not_hits():
    truth, dist = truth_of((1, 0.1), (2, 0.2))
    s = score_query([1, 99], truth, k=2, true_distance=dist, tenant_ids={1, 2})
    assert s.wrong_tenant == 1 and s.hits == 1


def test_duplicates_and_extra_rows_do_not_inflate_recall():
    truth, dist = truth_of((1, 0.1), (2, 0.2), (3, 0.3))
    s = score_query([1, 1, 1], truth, k=2, true_distance=dist)
    assert s.hits == 1
    s = score_query([1, 2, 3], truth, k=2, true_distance=dist)
    assert s.hits == 2


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        score_query([], [(1, 0.1)], k=0, true_distance={1: 0.1})


def test_summarize():
    scores = [
        QueryScore(10, 10, 10, 0),
        QueryScore(10, 4, 4, 0),
        QueryScore(1, 0, 0, 0),
    ]
    b = summarize(scores)
    assert b.queries == 3
    assert b.mean_recall == pytest.approx((1 + 0.4 + 0) / 3)
    assert b.min_recall == 0
    assert b.short_share == pytest.approx(2 / 3)
    assert b.zero_share == pytest.approx(1 / 3)
    assert b.mean_returned == pytest.approx(14 / 3)


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile(values, 100) == 100
    assert percentile([5.0], 95) == 5.0
    with pytest.raises(ValueError):
        percentile([], 50)

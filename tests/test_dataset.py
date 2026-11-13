import numpy as np
import pytest

from ptr.dataset import DatasetConfig, exact_topk, generate

SMALL = DatasetConfig(
    latent_dim=8,
    rows=3_000,
    dim=16,
    topics=20,
    background_tenants=50,
    queries_per_bucket=4,
    probe_buckets={1: 3, 10: 3, 100: 2},
)


def test_probe_tenants_have_exact_sizes_and_rows_add_up():
    ds = generate(SMALL)
    assert len(ds.tenant_of_row) == SMALL.rows
    assert sum(ds.tenant_size.values()) == SMALL.rows
    for bucket, ids in ds.probe_tenants.items():
        assert len(ids) == SMALL.probe_buckets[bucket]
        for t in ids:
            assert ds.tenant_size[t] == bucket
            assert int((ds.tenant_of_row == t).sum()) == bucket


def test_vectors_are_unit_length_and_deterministic():
    a, b = generate(SMALL), generate(SMALL)
    assert np.allclose(np.linalg.norm(a.vectors, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a.vectors, b.vectors)
    assert all(qa[1] == qb[1] for qa, qb in zip(a.queries, b.queries, strict=True))


def test_queries_cover_every_bucket():
    ds = generate(SMALL)
    buckets = [b for b, _, _ in ds.queries]
    assert {b: buckets.count(b) for b in set(buckets)} == {1: 4, 10: 4, 100: 4}


def test_exact_topk_matches_brute_force():
    ds = generate(SMALL)
    bucket, tenant, q = ds.queries[-1]
    ranked = exact_topk(ds, tenant, q)
    assert len(ranked) == bucket
    rows = np.flatnonzero(ds.tenant_of_row == tenant)
    best = rows[np.argmax(ds.vectors[rows] @ q)] + 1
    assert ranked[0][0] == best
    assert [d for _, d in ranked] == sorted(d for _, d in ranked)


def test_latent_dim_must_fit():
    with pytest.raises(ValueError):
        generate(DatasetConfig(rows=3_000, dim=16, latent_dim=32, background_tenants=50, probe_buckets={1: 1}))

"""Seeded synthetic multi-tenant embedding set.

This is not a public embedding dataset. It is deterministic random data shaped
like a shared RAG table:

* Probe tenants of exactly 1, 10, 100, 1,000 and 10,000 rows, which is where we
  measure. The rest of the table belongs to background tenants whose sizes
  follow a Zipf curve, so a few big customers own most rows.
* Vectors come from shared topics. Topic popularity is Zipf too, so many
  tenants write about the same popular topics (think "billing", "login").
  Each tenant has a small private offset, so its chunks differ from other
  tenants' chunks on the same topic.
* Real text embeddings use far fewer directions than they have dimensions, so
  everything is drawn in a `latent_dim` space and mapped into `dim` with a
  random orthonormal projection, plus a little full-dimensional noise. Without
  this, HNSW recall on the unfiltered table is much lower than it is on real
  embeddings, and every result would look worse than it should.
* Queries are written by a tenant about one of its own topics: they land near
  the tenant's data, and also near every other tenant who shares that topic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PROBE_BUCKETS: dict[int, int] = {1: 30, 10: 30, 100: 30, 1_000: 20, 10_000: 5}


@dataclass(frozen=True)
class DatasetConfig:
    rows: int = 200_000
    dim: int = 128
    latent_dim: int = 32
    ambient_noise: float = 0.05
    topics: int = 200
    background_tenants: int = 2_000
    tenant_zipf: float = 1.07
    topic_zipf: float = 1.0
    topic_noise: float = 0.9  # spread of chunks around their topic centre
    tenant_offset: float = 0.45  # how far a tenant's style shifts its chunks
    queries_per_bucket: int = 60
    seed: int = 20261113
    probe_buckets: dict[int, int] = field(default_factory=lambda: dict(PROBE_BUCKETS))


@dataclass
class Dataset:
    config: DatasetConfig
    tenant_of_row: np.ndarray  # int32 [rows]
    vectors: np.ndarray  # float32 [rows, dim], unit length
    tenant_size: dict[int, int]
    probe_tenants: dict[int, list[int]]  # bucket size -> tenant ids
    queries: list[tuple[int, int, np.ndarray]]  # (bucket, tenant, vector)


def tenant_sizes(cfg: DatasetConfig, rng: np.random.Generator) -> tuple[list[int], dict[int, list[int]]]:
    """Return per-tenant row counts and the probe tenants by bucket.

    Tenant ids are 1-based. Probe tenants come first so their ids are stable
    when only the table size changes.
    """
    sizes: list[int] = []
    probes: dict[int, list[int]] = {}
    for bucket, count in sorted(cfg.probe_buckets.items()):
        ids = []
        for _ in range(count):
            sizes.append(bucket)
            ids.append(len(sizes))
        probes[bucket] = ids
    remaining = cfg.rows - sum(sizes)
    if remaining < cfg.background_tenants:
        raise ValueError(f"{cfg.rows} rows cannot hold the probe tenants plus background tenants")
    ranks = np.arange(1, cfg.background_tenants + 1, dtype=np.float64)
    weights = ranks**-cfg.tenant_zipf
    # every background tenant gets at least one row, the rest is Zipf
    extra = rng.multinomial(remaining - cfg.background_tenants, weights / weights.sum())
    sizes.extend(int(x) + 1 for x in extra)
    return sizes, probes


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def generate(cfg: DatasetConfig) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    sizes, probes = tenant_sizes(cfg, rng)
    n_tenants = len(sizes)

    ld = cfg.latent_dim
    if not 0 < ld <= cfg.dim:
        raise ValueError("latent_dim must be in (0, dim]")
    projection, _ = np.linalg.qr(rng.standard_normal((cfg.dim, ld)))  # dim x ld, orthonormal columns
    projection = projection.T  # ld x dim

    def embed(latent: np.ndarray) -> np.ndarray:
        ambient = rng.standard_normal((*latent.shape[:-1], cfg.dim)) * cfg.ambient_noise / np.sqrt(cfg.dim)
        return _unit(_unit(latent) @ projection + ambient)

    centres = _unit(rng.standard_normal((cfg.topics, ld)))
    topic_p = np.arange(1, cfg.topics + 1, dtype=np.float64) ** -cfg.topic_zipf
    topic_p /= topic_p.sum()
    noise_scale = 1.0 / np.sqrt(ld)

    tenant_topics: list[np.ndarray] = []
    offsets = rng.standard_normal((n_tenants, ld)) * noise_scale * cfg.tenant_offset
    tenant_of_row = np.empty(cfg.rows, dtype=np.int32)
    vectors = np.empty((cfg.rows, cfg.dim), dtype=np.float32)
    pos = 0
    for t, size in enumerate(sizes):
        n_topics = int(rng.integers(1, 5))
        topics = rng.choice(cfg.topics, size=n_topics, replace=False, p=topic_p)
        tenant_topics.append(topics)
        picked = topics[rng.integers(0, n_topics, size=size)]
        noise = rng.standard_normal((size, ld)) * noise_scale * cfg.topic_noise
        vectors[pos : pos + size] = embed(centres[picked] + offsets[t] + noise)
        tenant_of_row[pos : pos + size] = t + 1
        pos += size

    queries: list[tuple[int, int, np.ndarray]] = []
    for bucket, ids in sorted(probes.items()):
        for q in range(cfg.queries_per_bucket):
            tenant = ids[q % len(ids)]
            topics = tenant_topics[tenant - 1]
            topic = topics[int(rng.integers(0, len(topics)))]
            noise = rng.standard_normal(ld) * noise_scale * cfg.topic_noise
            queries.append((bucket, tenant, embed(centres[topic] + offsets[tenant - 1] + noise)))

    return Dataset(
        config=cfg,
        tenant_of_row=tenant_of_row,
        vectors=vectors,
        tenant_size={i + 1: s for i, s in enumerate(sizes)},
        probe_tenants=probes,
        queries=queries,
    )


def exact_topk(ds: Dataset, tenant: int, query: np.ndarray) -> list[tuple[int, float]]:
    """Exact cosine distance ranking of every row of a tenant (row id = index + 1)."""
    rows = np.flatnonzero(ds.tenant_of_row == tenant)
    sims = ds.vectors[rows].astype(np.float64) @ query.astype(np.float64)
    norms = np.linalg.norm(ds.vectors[rows].astype(np.float64), axis=1) * np.linalg.norm(query.astype(np.float64))
    dist = 1.0 - sims / norms
    order = np.argsort(dist, kind="stable")
    return [(int(rows[i]) + 1, float(dist[i])) for i in order]

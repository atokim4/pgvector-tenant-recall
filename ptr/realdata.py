"""A multi-tenant table built from real text embeddings.

Source: the Simple English Wikipedia dump on Hugging Face
(`wikimedia/wikipedia`, config `20231101.simple`, CC BY-SA). Every paragraph of
80+ characters is one chunk, cut to `max_chars`, embedded with
`sentence-transformers/all-MiniLM-L6-v2` (384 dims) through fastembed (ONNX, CPU).

Tenants are customers whose knowledge base is a set of whole articles:

* 1-row and 10-row probe tenants each own one article that has exactly that many
  paragraphs, so they are natural documents, not cut-outs.
* 100, 1,000 and 10,000-row probe tenants own a random mix of articles (the
  last one is cut to hit the exact size).
* Background tenants get Zipf-distributed sizes, filled with the next
  articles in a seeded shuffle.

A query is the title of one of the tenant's own articles, phrased as a question
("What is <title>?"). That is a realistic retrieval query, and unlike the
synthetic set nobody tuned how close it lands to the tenant's rows.

The embeddings are cached as .npy next to the parquet file, so the slow step
(about 45 minutes for 200k chunks on 2 vCPU) runs once.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataset import PROBE_BUCKETS, Dataset, DatasetConfig, tenant_sizes

PARQUET_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.simple/train-00000-of-00001.parquet"
)
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384


@dataclass(frozen=True)
class RealConfig:
    rows: int = 200_000
    min_chars: int = 80
    max_chars: int = 300
    background_tenants: int = 2_000
    tenant_zipf: float = 1.07
    queries_per_bucket: int = 60
    seed: int = 20261113
    probe_buckets: dict[int, int] = field(default_factory=lambda: dict(PROBE_BUCKETS))


def paragraphs(text: str, min_chars: int, max_chars: int) -> list[str]:
    """Split an article into chunks: one per paragraph long enough to carry meaning."""
    return [p.strip()[:max_chars] for p in text.split("\n") if len(p.strip()) >= min_chars]


@dataclass
class Layout:
    """Which chunks go to which tenant, before anything is embedded."""

    chunks: list[str]
    tenant_of_row: np.ndarray
    tenant_size: dict[int, int]
    probe_tenants: dict[int, list[int]]
    queries: list[tuple[int, int, str]]  # (bucket, tenant, query text)


def plan_layout(articles: list[tuple[str, list[str]]], cfg: RealConfig) -> Layout:
    """Assign whole articles to tenants. `articles` is (title, chunks) in source order."""
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(articles))
    by_len: dict[int, list[int]] = {}
    for i in order:
        by_len.setdefault(len(articles[i][1]), []).append(int(i))
    used: set[int] = set()

    chunks: list[str] = []
    owner: list[int] = []
    titles: dict[int, list[str]] = {}

    def give(tenant: int, art: int, take: int | None = None) -> int:
        title, parts = articles[art]
        parts = parts[:take] if take is not None else parts
        chunks.extend(parts)
        owner.extend([tenant] * len(parts))
        titles.setdefault(tenant, []).append(title)
        used.add(art)
        return len(parts)

    stream = (int(i) for i in order)

    def next_free() -> int:
        for i in stream:
            if i not in used and articles[i][1]:
                return i
        raise ValueError("ran out of articles")

    def fill(tenant: int, size: int) -> None:
        left = size
        while left > 0:
            art = next_free()
            left -= give(tenant, art, take=left)

    probes: dict[int, list[int]] = {}
    tenant = 0
    for bucket, count in sorted(cfg.probe_buckets.items()):
        probes[bucket] = []
        for _ in range(count):
            tenant += 1
            exact = [a for a in by_len.get(bucket, []) if a not in used]
            if bucket <= 10 and exact:
                give(tenant, exact[0])
            else:
                fill(tenant, bucket)
            probes[bucket].append(tenant)

    probe_rows = len(chunks)
    if cfg.rows - probe_rows < cfg.background_tenants:
        raise ValueError(f"{cfg.rows} rows cannot hold the probe tenants plus background tenants")
    shape = DatasetConfig(
        rows=cfg.rows - probe_rows,
        background_tenants=cfg.background_tenants,
        tenant_zipf=cfg.tenant_zipf,
        probe_buckets={},
    )
    background, _ = tenant_sizes(shape, rng)
    for size in background:
        tenant += 1
        fill(tenant, size)

    tenant_of_row = np.asarray(owner, dtype=np.int32)
    sizes = {t: int(n) for t, n in zip(*np.unique(tenant_of_row, return_counts=True), strict=True)}
    queries: list[tuple[int, int, str]] = []
    for bucket, ids in sorted(probes.items()):
        for q in range(cfg.queries_per_bucket):
            t = ids[q % len(ids)]
            title = titles[t][int(rng.integers(0, len(titles[t])))]
            queries.append((bucket, t, f"What is {title}?"))
    return Layout(chunks, tenant_of_row, sizes, probes, queries)


def load_articles(parquet: Path, cfg: RealConfig) -> list[tuple[str, list[str]]]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["title", "text"])
    titles = table.column("title").to_pylist()
    texts = table.column("text").to_pylist()
    return [(t, paragraphs(x, cfg.min_chars, cfg.max_chars)) for t, x in zip(titles, texts, strict=True)]


def _embed(texts: list[str], cache_dir: Path) -> np.ndarray:
    from fastembed import TextEmbedding

    model = TextEmbedding(MODEL, cache_dir=str(cache_dir / "models"))
    out = np.empty((len(texts), DIM), dtype=np.float32)
    for i, v in enumerate(model.embed(texts, batch_size=256)):
        out[i] = v
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / norms


def build(cfg: RealConfig, cache_dir: Path) -> Dataset:
    """Return the real-embedding Dataset, embedding (and caching) what is missing."""
    parquet = cache_dir / "simplewiki.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"download {PARQUET_URL} to {parquet} first")
    layout = plan_layout(load_articles(parquet, cfg), cfg)
    key = hashlib.sha256(json.dumps([MODEL, cfg.__dict__], sort_keys=True).encode()).hexdigest()[:12]
    rows_file = cache_dir / f"rows-{key}.npy"
    queries_file = cache_dir / f"queries-{key}.npy"
    if not rows_file.exists():
        np.save(rows_file, _embed(layout.chunks, cache_dir))
    if not queries_file.exists():
        np.save(queries_file, _embed([q for _, _, q in layout.queries], cache_dir))
    vectors = np.load(rows_file)
    qvecs = np.load(queries_file)
    if len(vectors) != len(layout.chunks):
        raise RuntimeError(f"stale cache {rows_file}: {len(vectors)} vectors for {len(layout.chunks)} chunks")
    return Dataset(
        config=DatasetConfig(rows=len(vectors), dim=DIM, probe_buckets=cfg.probe_buckets),
        tenant_of_row=layout.tenant_of_row,
        vectors=vectors,
        tenant_size=layout.tenant_size,
        probe_tenants=layout.probe_tenants,
        queries=[(b, t, qvecs[i]) for i, (b, t, _) in enumerate(layout.queries)],
    )

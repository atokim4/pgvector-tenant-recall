"""Recall of the real langchain-postgres PGVectorStore (v2) for small tenants.

Runs the library's own `asimilarity_search_by_vector(..., filter={"user_id": ...})` on a table built from the
synthetic dataset and compares it with exact search in numpy. Unlike the replays in ptr/frameworks.py, every
query goes through langchain's own SQL, SET LOCAL handling and result mapping.

It creates (and drops first) its own database, `lcverify` by default, on the bench server.

The iterative_scan / max_scan_tuples rows need a langchain-postgres whose HNSWQueryOptions accepts those
fields; released versions up to 0.0.19 only have ef_search, and the script skips those rows there.

    pip install -e '.[langchain]'
    python scripts/check_langchain_v2.py --out results/langchain_v2.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import psycopg
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex, HNSWQueryOptions

from ptr.dataset import DatasetConfig, exact_topk, generate

TABLE = "lc_docs"
K = 10
MAX_BUCKET = 1_000  # 10k-row tenants add minutes and nothing new here


class VectorsOnly(Embeddings):
    """The check passes vectors in directly, so the store never needs to embed text."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


def doc_id(row_id: int) -> str:
    # row ids follow ptr.dataset.exact_topk (index + 1)
    return f"00000000-0000-0000-0000-{row_id:012d}"


def option_sets() -> list[tuple[str, HNSWQueryOptions | None, bool]]:
    """(label, query options, create a B-tree on the filter key first)."""
    sets: list[tuple[str, HNSWQueryOptions | None, bool]] = [("default (ef_search 40)", None, False)]
    if "iterative_scan" in {f.name for f in fields(HNSWQueryOptions)}:
        sets += [
            ("iterative_scan=strict_order", HNSWQueryOptions(iterative_scan="strict_order"), False),
            (
                "strict_order + max_scan_tuples=100000",
                HNSWQueryOptions(iterative_scan="strict_order", max_scan_tuples=100_000),
                False,
            ),
        ]
    else:
        print("this langchain-postgres has no HNSWQueryOptions.iterative_scan; skipping those rows")
    sets.append(("B-tree on (langchain_metadata->>'user_id')", None, True))
    return sets


async def run(server_dsn: str, db: str, out: Path) -> None:
    ds = generate(DatasetConfig())
    with psycopg.connect(server_dsn, autocommit=True) as c:
        c.execute(f'DROP DATABASE IF EXISTS "{db}"')
        c.execute(f'CREATE DATABASE "{db}"')
    dsn = psycopg.conninfo.make_conninfo(server_dsn, dbname=db)
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("CREATE EXTENSION vector")
        (pgvector,) = c.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    print(f"pgvector {pgvector}, {len(ds.tenant_of_row)} rows")

    engine = PGEngine.from_connection_string(sqlalchemy_url(dsn))
    try:
        await engine.ainit_vectorstore_table(TABLE, ds.config.dim)
        with psycopg.connect(dsn) as c, c.cursor() as cur:
            columns = "langchain_id, content, embedding, langchain_metadata"
            with cur.copy(f'COPY "{TABLE}" ({columns}) FROM STDIN') as copy:
                for i, (tenant, vec) in enumerate(zip(ds.tenant_of_row, ds.vectors, strict=True)):
                    embedding = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
                    meta = json.dumps({"user_id": f"u{int(tenant)}"})
                    copy.write_row((doc_id(i + 1), f"chunk {i + 1}", embedding, meta))
            c.commit()
            cur.execute(f'ANALYZE "{TABLE}"')
            c.commit()

        store = await PGVectorStore.create(engine, embedding_service=VectorsOnly(), table_name=TABLE)
        t0 = time.perf_counter()
        await store.aapply_vector_index(HNSWIndex())
        print(f"HNSW build {time.perf_counter() - t0:.0f} s")

        results = []
        for label, options, btree in option_sets():
            if btree:
                with psycopg.connect(dsn, autocommit=True) as c:
                    c.execute(f"CREATE INDEX lc_user ON \"{TABLE}\" ((langchain_metadata->>'user_id'))")
                    c.execute(f'ANALYZE "{TABLE}"')
            store = await PGVectorStore.create(
                engine, embedding_service=VectorsOnly(), table_name=TABLE, index_query_options=options
            )
            by_bucket: dict[int, list[tuple[float, bool, float, int]]] = {}
            for bucket, tenant, q in ds.queries:
                if bucket > MAX_BUCKET:
                    continue
                want = min(K, ds.tenant_size[tenant])
                truth = {row for row, _ in exact_topk(ds, tenant, q)[:want]}
                t = time.perf_counter()
                docs = await store.asimilarity_search_by_vector(
                    [float(x) for x in q], k=K, filter={"user_id": f"u{tenant}"}
                )
                ms = (time.perf_counter() - t) * 1000
                got = {int(d.id.rsplit("-", 1)[1]) for d in docs}
                leaks = sum(d.metadata.get("user_id") != f"u{tenant}" for d in docs)
                by_bucket.setdefault(bucket, []).append((len(got & truth) / want, len(docs) < want, ms, leaks))
            for bucket, rows in sorted(by_bucket.items()):
                row = {
                    "options": label,
                    "tenant_rows": bucket,
                    "queries": len(rows),
                    "recall": round(float(np.mean([r[0] for r in rows])), 3),
                    "short_share": round(float(np.mean([r[1] for r in rows])), 3),
                    "p50_ms": round(float(np.median([r[2] for r in rows])), 1),
                    "leaks": sum(r[3] for r in rows),
                }
                print(
                    f"{label:45s} tenant={bucket:5d} recall@{K}={row['recall']:.2f} "
                    f"short={row['short_share']:.0%} p50={row['p50_ms']} ms leaks={row['leaks']}"
                )
                results.append(row)
        out.write_text(json.dumps({"pgvector": pgvector, "k": K, "results": results}, indent=1) + "\n")
    finally:
        await engine.close()


def sqlalchemy_url(dsn: str) -> str:
    p = psycopg.conninfo.conninfo_to_dict(dsn)
    user = p.get("user", "postgres")
    query = "&".join(f"{k}={p[k]}" for k in ("host", "port") if k in p)
    return f"postgresql+psycopg://{user}@/{p['dbname']}?{query}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dsn",
        default=os.environ.get("PTR_DSN", "host=/tmp/ptr-pg port=5544 user=postgres dbname=postgres"),
        help="any database on the bench server (used to create the scratch database)",
    )
    ap.add_argument("--db", default="lcverify", help="scratch database to create")
    ap.add_argument("--out", type=Path, default=Path("results/langchain_v2.json"))
    args = ap.parse_args()
    asyncio.run(run(args.dsn, args.db, args.out))


if __name__ == "__main__":
    main()

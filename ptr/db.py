"""Schema, loading and index builds."""

from __future__ import annotations

import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Json, Jsonb

from .dataset import Dataset

PARTITIONS = 16
# every row sits in one collection, as in a single-collection langchain store
COLLECTION = "6f1d3c1e-2a8f-4f0e-9b7a-1c5e2d9a4b10"
HNSW_OPTS = "m = 16, ef_construction = 64"  # pgvector defaults, spelled out


def connect(dsn: str) -> psycopg.Connection:
    # prepare_threshold=None: psycopg never prepares on its own, so the plan cache
    # mode is chosen explicitly per run (see strategies.PLAN_CACHE_MODES)
    conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=None)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def server_versions(conn: psycopg.Connection) -> dict[str, str]:
    pg = conn.execute("SHOW server_version").fetchone()[0]
    vec = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
    return {"postgres": pg, "pgvector": vec}


def _copy_rows(conn: psycopg.Connection, table: str, ds: Dataset) -> None:
    with (
        conn.cursor() as cur,
        cur.copy(
            f"COPY {table} (id, tenant_id, metadata, meta_json, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy,
    ):
        copy.set_types(["int8", "int4", "jsonb", "json", "vector"])
        for i in range(len(ds.tenant_of_row)):
            tenant = int(ds.tenant_of_row[i])
            # real chunk metadata carries more than the owner, so whole documents are
            # unique per row (that matters for @> estimates on the full value)
            meta = {"user_id": f"user-{tenant}", "chunk": i + 1}
            copy.write_row((i + 1, tenant, Jsonb(meta), Json(meta), ds.vectors[i]))


def _timed(conn: psycopg.Connection, sql: str) -> float:
    start = time.perf_counter()
    conn.execute(sql)
    return round(time.perf_counter() - start, 1)


def load_table(conn: psycopg.Connection, ds: Dataset) -> dict[str, float]:
    dim = ds.config.dim
    conn.execute("DROP TABLE IF EXISTS items")
    conn.execute(
        "CREATE TABLE items (id bigint PRIMARY KEY, tenant_id int NOT NULL, "
        f"metadata jsonb NOT NULL, meta_json json NOT NULL, embedding vector({dim}) NOT NULL, "
        f"collection_id uuid NOT NULL DEFAULT '{COLLECTION}')"
    )
    # the one-row collection table that langchain-postgres joins against
    conn.execute("DROP TABLE IF EXISTS collections")
    conn.execute("CREATE TABLE collections (uuid uuid PRIMARY KEY, name text NOT NULL)")
    conn.execute("INSERT INTO collections VALUES (%s, 'docs')", (COLLECTION,))
    start = time.perf_counter()
    _copy_rows(conn, "items", ds)
    load = round(time.perf_counter() - start, 1)
    conn.execute("VACUUM ANALYZE items")
    conn.execute("ANALYZE collections")
    # both filter columns must select exactly the tenant's rows
    tenant = ds.probe_tenants[min(ds.probe_tenants)][0]
    by_col, by_json = conn.execute(
        "SELECT count(*) FILTER (WHERE tenant_id = %s), count(*) FILTER (WHERE metadata->>'user_id' = %s) FROM items",
        (tenant, f"user-{tenant}"),
    ).fetchone()
    if not by_col == by_json == ds.tenant_size[tenant]:
        raise RuntimeError(f"load check failed: {by_col=} {by_json=} expected {ds.tenant_size[tenant]}")
    return {"load_s": load}


def build_hnsw(conn: psycopg.Connection) -> dict[str, float]:
    """(Re)build the HNSW index and drop the B-trees, back to the plain layout.

    pgvector draws HNSW levels at random, so each build is a slightly different
    graph; the bench repeats the whole run over several builds.
    """
    for index in ("items_hnsw", "items_tenant", "items_user"):
        conn.execute(f"DROP INDEX IF EXISTS {index}")
    secs = _timed(conn, f"CREATE INDEX items_hnsw ON items USING hnsw (embedding vector_cosine_ops) WITH ({HNSW_OPTS})")
    conn.execute("ANALYZE items")
    return {"hnsw_build_s": secs}


def add_btree(conn: psycopg.Connection) -> dict[str, float]:
    conn.execute("DROP INDEX IF EXISTS items_tenant")
    conn.execute("DROP INDEX IF EXISTS items_user")
    secs = _timed(conn, "CREATE INDEX items_tenant ON items (tenant_id)")
    # an expression index also makes ANALYZE collect stats on metadata->>'user_id'
    expr = _timed(conn, "CREATE INDEX items_user ON items ((metadata->>'user_id'))")
    conn.execute("ANALYZE items")
    return {"btree_build_s": secs, "expr_btree_build_s": expr}


def build_partitioned(conn: psycopg.Connection, ds: Dataset) -> dict[str, float]:
    dim = ds.config.dim
    conn.execute("DROP TABLE IF EXISTS items_part")
    conn.execute(
        "CREATE TABLE items_part (id bigint NOT NULL, tenant_id int NOT NULL, "
        f"metadata jsonb NOT NULL, embedding vector({dim}) NOT NULL) PARTITION BY HASH (tenant_id)"
    )
    for p in range(PARTITIONS):
        conn.execute(
            f"CREATE TABLE items_part_{p} PARTITION OF items_part FOR VALUES WITH (MODULUS {PARTITIONS}, REMAINDER {p})"
        )
    conn.execute("INSERT INTO items_part SELECT id, tenant_id, metadata, embedding FROM items")
    conn.execute("VACUUM ANALYZE items_part")
    # an index on the parent cascades to every partition
    hnsw = _timed(
        conn,
        f"CREATE INDEX items_part_hnsw ON items_part USING hnsw (embedding vector_cosine_ops) WITH ({HNSW_OPTS})",
    )
    return {"part_hnsw_build_s": hnsw}


def create_indexes(conn: psycopg.Connection, statements: tuple[str, ...]) -> dict[str, float]:
    """Create a strategy's own indexes (CREATE INDEX <name> ...) and ANALYZE; returns name -> seconds."""
    created = {}
    for ddl in statements:
        name = ddl.split()[2]
        conn.execute(f"DROP INDEX IF EXISTS {name}")
        created[name] = _timed(conn, ddl)
    if created:
        conn.execute("ANALYZE items")  # expression indexes bring their own statistics
    return created


def drop_indexes(conn: psycopg.Connection, names: dict[str, float]) -> None:
    for name in names:
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    if names:
        conn.execute("ANALYZE items")


def relation_sizes(conn: psycopg.Connection) -> dict[str, str]:
    rows = conn.execute(
        "SELECT c.relname, pg_size_pretty(pg_total_relation_size(c.oid)) FROM pg_class c "
        "WHERE c.relname IN ('items', 'items_hnsw', 'items_tenant', 'items_user') ORDER BY 1"
    ).fetchall()
    return dict(rows)


def unfiltered_ids(conn: psycopg.Connection, q: np.ndarray, k: int, ef_search: int) -> list[int]:
    """Plain HNSW top-k with no WHERE clause."""
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
        rows = conn.execute("SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s", (q, k)).fetchall()
    return [int(r[0]) for r in rows]


def seqscan_topk(conn: psycopg.Connection, tenant: int, q: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Exact top-k from Postgres itself, with every index switched off."""
    with conn.transaction():
        conn.execute("SET LOCAL enable_indexscan = off")
        conn.execute("SET LOCAL enable_bitmapscan = off")
        rows = conn.execute(
            "SELECT id, embedding <=> %s AS d FROM items WHERE tenant_id = %s ORDER BY d LIMIT %s",
            (q, tenant, k),
        ).fetchall()
    return [(int(i), float(d)) for i, d in rows]

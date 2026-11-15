"""Run the benchmark: python -m ptr.bench --dsn ... [--rows 200000]."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psycopg
from psycopg.types.numeric import Int4

from . import db
from .dataset import Dataset, DatasetConfig, exact_topk, generate
from .frameworks import FRAMEWORK_STRATEGIES
from .metrics import QueryScore, percentile, score_query, summarize
from .rls import RLS_STRATEGIES
from .strategies import (
    ALL_PLAN_CACHE_MODES,
    AUTO,
    GENERIC,
    HNSW_BTREE,
    LAYOUT_ORDER,
    PARTITIONED,
    PLAN_CACHE_MODES,
    STRATEGIES,
    Strategy,
)

ALL_STRATEGIES = STRATEGIES + FRAMEWORK_STRATEGIES + RLS_STRATEGIES

QUERY_FIELDS = [
    "build",
    "strategy",
    "plan_cache",
    "bucket",
    "tenant",
    "query",
    "expected",
    "returned",
    "hits",
    "recall",
    "wrong_tenant",
    "ms",
    "access",
    "plan",
]
SUMMARY_FIELDS = [
    "build",
    "strategy",
    "plan_cache",
    "label",
    "bucket",
    "queries",
    "mean_recall",
    "min_recall",
    "short_share",
    "zero_share",
    "mean_returned",
    "expected",
    "p50_ms",
    "p95_ms",
    "hnsw_share",
    "hnsw_recall",
    "wrong_tenant",
    "plans",
]


AGG_FIELDS = [
    "strategy",
    "plan_cache",
    "label",
    "bucket",
    "builds",
    "mean_recall",
    "recall_low",
    "recall_high",
    "zero_share",
    "short_share",
    "hnsw_share",
    "p50_ms",
    "p95_ms",
]


def aggregate(rows: list[dict]) -> list[dict]:
    """Collapse per-build summaries: mean over builds, plus the lowest and highest build."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["strategy"], r["plan_cache"], r["bucket"]), []).append(r)
    out = []
    for (name, mode, bucket), rs in groups.items():
        recalls = [float(r["mean_recall"]) for r in rs]

        def avg(field: str, rs: list[dict] = rs) -> float:
            return round(sum(float(r[field]) for r in rs) / len(rs), 4)

        out.append(
            {
                "strategy": name,
                "plan_cache": mode,
                "label": rs[0]["label"],
                "bucket": bucket,
                "builds": len(rs),
                "mean_recall": round(sum(recalls) / len(recalls), 4),
                "recall_low": min(recalls),
                "recall_high": max(recalls),
                "zero_share": avg("zero_share"),
                "short_share": avg("short_share"),
                "hnsw_share": avg("hnsw_share"),
                "p50_ms": avg("p50_ms"),
                "p95_ms": avg("p95_ms"),
            }
        )
    return out


def compact_plan(root: dict) -> tuple[str, str]:
    """Compact plan (node types with index names, outermost first) and how rows were found.

    access is "hnsw" when an HNSW index produced the ordering, "btree" when a
    tenant-key index (B-tree or GIN) fed an exact sort, "seq" for a sequential
    scan plus exact sort, "function" for a SQL function whose inner plan is unknown.
    """
    parts: list[str] = []

    def walk(node: dict) -> None:
        name = node["Node Type"]
        if "Index Name" in node:
            name += f"({node['Index Name']})"
        parts.append(name)
        for child in node.get("Plans", []):
            walk(child)

    walk(root)
    # partition scans repeat the same node per partition; keep the first
    seen: list[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    plan = " > ".join(seen)
    if "hnsw" in plan or "embedding_idx" in plan:
        access = "hnsw"
    elif "Index" in plan:
        access = "btree"
    elif "Function Scan" in plan:
        access = "function"
    else:
        access = "seq"
    return plan, access


def plan_summary(conn: psycopg.Connection | psycopg.ClientCursor, sql: str, params: dict | None) -> tuple[str, str]:
    statement = sql if params is None else "EXPLAIN (FORMAT JSON) " + sql
    (doc,) = conn.execute(statement, params).fetchone()
    return compact_plan(doc[0]["Plan"])


class NestedPlans:
    """Plans of the statements inside a SQL function, read through auto_explain.

    EXPLAIN of `SELECT ... FROM match_documents(...)` only shows a Function Scan,
    and plpgsql switches its cached plan on its own after five calls, so the real
    plan has to be caught as it runs. Logging a plan (without ANALYZE) is cheap
    next to the query, but it is included in the timings of these strategies.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self.messages: list[str] = []
        conn.execute("LOAD 'auto_explain'")
        for key, value in {
            "auto_explain.log_min_duration": "0",
            "auto_explain.log_nested_statements": "on",
            "auto_explain.log_format": "json",
            "auto_explain.log_level": "notice",
            "client_min_messages": "notice",
        }.items():
            conn.execute("SELECT set_config(%s, %s, false)", (key, value))
        conn.add_notice_handler(lambda d: self.messages.append(d.message_primary or ""))

    def take(self) -> tuple[str, str]:
        """The plan of the innermost statement logged since the last call."""
        inner = None
        for msg in self.messages:
            _, _, body = msg.partition("plan:")
            try:
                doc = json.loads(body)
            except json.JSONDecodeError:
                continue
            if "match_documents" not in doc.get("Query Text", ""):
                inner = doc["Plan"]
        self.messages.clear()
        return compact_plan(inner) if inner else ("unknown", "function")


def params_for(tenant: int, q, k: int) -> dict:
    # the tenant id is sent as int4 so EXPLAIN (GENERIC_PLAN) sees the same types;
    # the other keys are the filter values in the shapes the framework replays send
    user = f"user-{tenant}"
    return {
        "tenant": Int4(tenant),
        "user": user,
        "q": q,
        "k": k,
        "key": "user_id",
        "coll": db.COLLECTION,
        "jpvars": json.dumps({"value": user}),
        "userjson": json.dumps(user),
        "filter": json.dumps({"user_id": user}),
        "jsonpath": f'$."user_id" == "{user}"',
        "threshold": 1.0,
    }


def unfiltered_baseline(conn: psycopg.Connection, ds: Dataset, k: int) -> dict:
    """Recall of HNSW with no filter at all, against exact search over the whole table."""
    out = {}
    vecs = ds.vectors.astype(np.float64)
    exact = []
    for _, _, q in ds.queries:
        d = 1.0 - vecs @ q.astype(np.float64)
        top = np.argsort(d, kind="stable")[:k]
        exact.append([(int(i) + 1, float(d[i])) for i in top])
    for ef in (40, 200):
        recalls = []
        for (_, _, q), truth in zip(ds.queries, exact, strict=True):
            got = db.unfiltered_ids(conn, q, k, ef)
            dist = {i: float(1.0 - vecs[i - 1] @ q.astype(np.float64)) for i in got}
            recalls.append(score_query(got, truth, k, dist).recall)
        out[f"ef_search_{ef}"] = round(float(np.mean(recalls)), 4)
    return out


def run_strategy(
    dsn: str,
    strat: Strategy,
    plan_cache: str,
    ds: Dataset,
    truth: list[list[tuple[int, float]]],
    k: int,
) -> tuple[list[dict], list[dict]]:
    conn = db.connect(dsn)
    settings = dict(strat.settings)
    if plan_cache == GENERIC:
        settings["plan_cache_mode"] = "force_generic_plan"
    for key, value in settings.items():
        conn.execute("SELECT set_config(%s, %s, false)", (key, value))
    if strat.role:
        conn.execute(f"SET ROLE {strat.role}")
    sql = strat.sql()
    # cur runs the searches; in auto mode it is a client-side-binding cursor that
    # EXECUTEs one named prepared statement, so the server's plan cache decides
    cur: psycopg.Connection | psycopg.ClientCursor = conn
    if plan_cache == AUTO:
        conn.execute(f"PREPARE ptr_search AS {strat.dollar_sql()}")
        sql = "EXECUTE ptr_search(" + ", ".join(f"%({n})s" for n in strat.param_names()) + ")"
        cur = psycopg.ClientCursor(conn)

    def bind_tenant(tenant: int) -> None:
        if strat.tenant_setting:
            conn.execute("SELECT set_config(%s, %s, false)", (strat.tenant_setting, str(tenant)))

    def generic_explain() -> tuple[str, str]:
        return plan_summary(conn, "EXPLAIN (GENERIC_PLAN, FORMAT JSON) " + strat.dollar_sql(), None)

    # under force_generic_plan every tenant runs the same plan, so one EXPLAIN covers
    # them all, unless the plan can depend on a per-tenant setting (RLS)
    generic_plan = None
    if plan_cache == GENERIC and not strat.tenant_setting:
        generic_plan = generic_explain()
    nested = NestedPlans(conn) if strat.function_call else None
    per_query: list[dict] = []
    scores: dict[int, list[QueryScore]] = {}
    times: dict[int, list[float]] = {}
    plans: dict[int, list[tuple[str, str]]] = {}
    # warm-up: one untimed query per bucket (in auto mode these are the custom plans
    # the server compares a generic plan against)
    for bucket in sorted(ds.probe_tenants):
        _, tenant, q = next(x for x in ds.queries if x[0] == bucket)
        bind_tenant(tenant)
        cur.execute(sql, params_for(tenant, q, k)).fetchall()
    if nested:
        nested.take()
    for idx, (bucket, tenant, q) in enumerate(ds.queries):
        params = params_for(tenant, q, k)
        bind_tenant(tenant)
        # custom plans depend on the tenant id, so every query gets its own EXPLAIN
        if not nested:
            if generic_plan:
                plan, access = generic_plan
            elif plan_cache == GENERIC:
                plan, access = generic_explain()
            else:
                plan, access = plan_summary(cur, sql, params)
        start = time.perf_counter()
        rows = cur.execute(sql, params).fetchall()
        ms = (time.perf_counter() - start) * 1000
        if nested:
            plan, access = nested.take()
        plans.setdefault(bucket, []).append((plan, access))
        t = truth[idx]
        dist = dict(t)
        s = score_query([r[0] for r in rows], t, k, dist, tenant_ids=set(dist))
        scores.setdefault(bucket, []).append(s)
        times.setdefault(bucket, []).append(ms)
        per_query.append(
            {
                "strategy": strat.name,
                "plan_cache": plan_cache,
                "bucket": bucket,
                "tenant": tenant,
                "query": idx,
                "expected": s.expected,
                "returned": s.returned,
                "hits": s.hits,
                "recall": round(s.recall, 4),
                "wrong_tenant": s.wrong_tenant,
                "ms": round(ms, 2),
                "access": access,
                "plan": plan,
            }
        )
    conn.close()
    summary = []
    for bucket in sorted(scores):
        b = summarize(scores[bucket])
        via_hnsw = [sc for sc, (_, a) in zip(scores[bucket], plans[bucket], strict=True) if a == "hnsw"]
        summary.append(
            {
                "strategy": strat.name,
                "plan_cache": plan_cache,
                "label": strat.label,
                "bucket": bucket,
                "queries": b.queries,
                "mean_recall": round(b.mean_recall, 4),
                "min_recall": round(b.min_recall, 4),
                "short_share": round(b.short_share, 4),
                "zero_share": round(b.zero_share, 4),
                "mean_returned": round(b.mean_returned, 2),
                "expected": round(b.expected, 2),
                "p50_ms": round(percentile(times[bucket], 50), 2),
                "p95_ms": round(percentile(times[bucket], 95), 2),
                "hnsw_share": round(sum(a == "hnsw" for _, a in plans[bucket]) / b.queries, 4),
                "hnsw_recall": round(summarize(via_hnsw).mean_recall, 4) if via_hnsw else "",
                "wrong_tenant": b.wrong_tenant,
                "plans": " | ".join(f"{n}x {p}" for (p, _), n in Counter(plans[bucket]).most_common()),
            }
        )
    return per_query, summary


def check_truth(conn: psycopg.Connection, ds: Dataset, truth: list, k: int, sample: int = 25) -> dict:
    """Compare numpy ground truth with a Postgres sequential scan on a sample of queries."""
    step = max(1, len(ds.queries) // sample)
    checked = mismatched = 0
    max_gap = 0.0
    for idx in range(0, len(ds.queries), step):
        _, tenant, q = ds.queries[idx]
        pg = db.seqscan_topk(conn, tenant, q, k)
        mine = truth[idx][: len(pg)]
        checked += 1
        max_gap = max([max_gap] + [abs(a[1] - b[1]) for a, b in zip(pg, mine, strict=True)])
        s = score_query([i for i, _ in pg], truth[idx], k, dict(truth[idx]))
        if s.recall < 1.0:
            mismatched += 1
    return {
        "checked": checked,
        "pg_vs_numpy_recall_below_1": mismatched,
        "max_distance_gap": max_gap,
    }


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dsn",
        default=os.environ.get("PTR_DSN", "host=/tmp/ptr-pg port=5544 user=postgres dbname=postgres"),
    )
    ap.add_argument(
        "--dataset",
        choices=("synthetic", "real"),
        default="synthetic",
        help="synthetic latent-space vectors, or Simple English Wikipedia + all-MiniLM-L6-v2 (see ptr/realdata.py)",
    )
    ap.add_argument(
        "--cache", type=Path, default=Path("/tmp/ptr-real"), help="parquet + embedding cache for --dataset real"
    )
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--tenant-offset", type=float, default=DatasetConfig.tenant_offset)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--builds", type=int, default=3, help="independent HNSW index builds to average over")
    ap.add_argument("--only", nargs="*", help="strategy names to run (default: all)")
    ap.add_argument(
        "--plan-cache",
        nargs="+",
        choices=ALL_PLAN_CACHE_MODES,
        default=list(PLAN_CACHE_MODES),
        help="auto runs only for strategies that list it (the RLS ones); see ptr/strategies.py",
    )
    args = ap.parse_args(argv)

    if args.builds < 1 or args.k < 1:
        ap.error("--builds and -k must be at least 1")
    t0 = time.perf_counter()
    if args.dataset == "real":
        from .realdata import RealConfig, build, probe_buckets_for

        real_cfg = RealConfig(rows=args.rows, probe_buckets=probe_buckets_for(args.rows))
        ds = build(real_cfg, args.cache)
        cfg = ds.config
    else:
        cfg = DatasetConfig(rows=args.rows, dim=args.dim, tenant_offset=args.tenant_offset)
        ds = generate(cfg)
    truth = [exact_topk(ds, tenant, q) for _, tenant, q in ds.queries]
    print(
        f"dataset: {cfg.rows} rows, {len(ds.tenant_size)} tenants, {len(ds.queries)} queries",
        f"({time.perf_counter() - t0:.1f}s)",
    )

    unknown = set(args.only or []) - {s.name for s in ALL_STRATEGIES}
    if unknown:
        ap.error(f"unknown strategies: {', '.join(sorted(unknown))}")
    strategies = [s for s in ALL_STRATEGIES if not args.only or s.name in args.only]
    conn = db.connect(args.dsn)
    meta: dict = {
        "versions": db.server_versions(conn),
        "python": platform.python_version(),
        "dataset": {"source": args.dataset, **(asdict(real_cfg) if args.dataset == "real" else asdict(cfg))},
        "k": args.k,
        "hnsw_options": db.HNSW_OPTS,
    }
    meta["load"] = db.load_table(conn, ds)
    meta["truth_check"] = check_truth(conn, ds, truth, args.k)
    print("truth check:", meta["truth_check"], flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    all_q: list[dict] = []
    all_s: list[dict] = []
    meta["builds"] = []
    meta["unfiltered_recall"] = []
    for build in range(1, args.builds + 1):
        timings = db.build_hnsw(conn)
        meta["unfiltered_recall"].append(unfiltered_baseline(conn, ds, args.k))
        print(f"build {build}: unfiltered HNSW recall@k {meta['unfiltered_recall'][-1]}", flush=True)
        for layout in LAYOUT_ORDER:
            todo = [s for s in strategies if s.layout == layout]
            if not todo:
                continue
            if layout == HNSW_BTREE:
                timings.update(db.add_btree(conn))
            if layout == PARTITIONED:
                timings.update(db.build_partitioned(conn, ds))
            for s in todo:
                for ddl in s.setup:
                    conn.execute(ddl.format(dim=cfg.dim))
                created = db.create_indexes(conn, s.indexes)
                timings.update({f"{name}_s": secs for name, secs in created.items()})
                for mode in [m for m in args.plan_cache if m in s.modes]:
                    t = time.perf_counter()
                    q_rows, s_rows = run_strategy(args.dsn, s, mode, ds, truth, args.k)
                    for row in q_rows + s_rows:
                        row["build"] = build
                    all_q += q_rows
                    all_s += s_rows
                    line = "  ".join(f"{r['bucket']}:{r['mean_recall']:.2f}" for r in s_rows)
                    print(f"{s.name:28s} {mode:8s} {line}  ({time.perf_counter() - t:.0f}s)", flush=True)
                db.drop_indexes(conn, created)
        meta["builds"].append(timings)
    meta["sizes"] = db.relation_sizes(conn)
    conn.close()

    write_csv(args.out / "queries.csv", QUERY_FIELDS, all_q)
    write_csv(args.out / "summary.csv", SUMMARY_FIELDS, all_s)
    write_csv(args.out / "summary_by_strategy.csv", AGG_FIELDS, aggregate(all_s))
    (args.out / "run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"wrote {args.out}/summary.csv, queries.csv, run.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

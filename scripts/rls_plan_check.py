"""Which plan does an RLS-filtered vector search get, and what does the planner know?

Run after `python -m ptr.bench` has loaded `items` (any dataset). For each policy
spelling it EXPLAINs the search (no WHERE clause, the policy adds the tenant filter)
as the RLS role, for the smallest and the biggest probe tenant, three ways:

* custom   - plain statement, literal LIMIT (what an unprepared query gets)
* generic  - EXPLAIN (GENERIC_PLAN) with LIMIT $2, the shape a prepared statement has
* generic, literal LIMIT - the same with LIMIT 10 written into the SQL

and prints the access path and the planner's row estimate for the scan on items.
Everything runs inside one transaction that is rolled back (the B-tree is dropped
in it to show the unindexed case), so the loaded table is left as it was.

    python scripts/rls_plan_check.py [--dsn ...]
"""

from __future__ import annotations

import argparse
import os

import psycopg

from ptr.rls import DIRECT, INITPLAN, POLICY, ROLE, SETTING, policy_ddl

SEARCH = "SELECT id FROM items ORDER BY embedding <=> {q} LIMIT {k}"


def scan_node(plan: dict) -> dict:
    """The node that reads items (seq, index or HNSW scan)."""
    if plan.get("Relation Name") == "items":
        return plan
    for child in plan.get("Plans", []):
        found = scan_node(child)
        if found:
            return found
    return {}


def explain(conn: psycopg.Connection, sql: str, generic: bool) -> str:
    """Access path and row estimate of the scan on items."""
    options = "GENERIC_PLAN, FORMAT JSON" if generic else "FORMAT JSON"
    (doc,) = conn.execute(f"EXPLAIN ({options}) {sql}").fetchone()
    node = scan_node(doc[0]["Plan"])
    return f"{node.get('Node Type', '?')}({node.get('Index Name', '-')}) est {node.get('Plan Rows', '?')} rows"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dsn",
        default=os.environ.get("PTR_DSN", "host=/tmp/ptr-pg port=5544 user=postgres dbname=postgres"),
    )
    args = ap.parse_args()
    # prepare_threshold=None: psycopg must not prepare the EXPLAINs themselves
    with psycopg.connect(args.dsn, prepare_threshold=None) as conn:
        by_size = "SELECT tenant_id, count(*) n FROM items GROUP BY tenant_id ORDER BY n, tenant_id"
        counts = conn.execute(by_size).fetchall()
        sizes = dict(counts)
        small, big = counts[0][0], counts[-1][0]
        dim = conn.execute("SELECT vector_dims(embedding) FROM items LIMIT 1").fetchone()[0]
        q = f"'[{','.join(['0.1'] * dim)}]'::vector"
        for layout, drop in (("B-tree on tenant_id", False), ("no B-tree", True)):
            if drop:
                conn.execute("RESET ROLE")
                conn.execute("DROP INDEX IF EXISTS items_tenant")
            for name, using in (("current_setting()", DIRECT), ("(SELECT current_setting())", INITPLAN)):
                conn.execute("RESET ROLE")
                for ddl in policy_ddl(using):
                    conn.execute(ddl)
                conn.execute(f"SET ROLE {ROLE}")
                print(f"\n== policy {name}, {layout}")
                for tenant in (small, big):
                    conn.execute("SELECT set_config(%s, %s, true)", (SETTING, str(tenant)))
                    shapes = {
                        "custom": (SEARCH.format(q=q, k=10), False),
                        "generic": (SEARCH.format(q="$1", k="$2"), True),
                        "generic, LIMIT 10": (SEARCH.format(q="$1", k=10), True),
                    }
                    for label, (sql, generic) in shapes.items():
                        out = explain(conn, sql, generic)
                        print(f"  tenant {tenant:>5} ({sizes[tenant]:>6} rows)  {label:18s} {out}")
        conn.rollback()
    print(f"\n(rolled back: policy {POLICY} and the B-tree are as the benchmark left them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

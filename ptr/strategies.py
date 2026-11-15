"""The query shapes and index setups under test.

Every strategy filters by tenant and asks for the k nearest rows. They differ
in the physical layout (which indexes exist), the session settings, and the
SQL shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# layouts, built in this order so each one only adds to the previous
HNSW = "hnsw"  # items + HNSW on embedding
HNSW_BTREE = "hnsw+btree"  # same table, plus a B-tree on tenant_id
PARTITIONED = "partitioned"  # items_part, hash-partitioned by tenant_id, HNSW per partition
FRAMEWORK = "framework"  # items + HNSW, plus whatever indexes the framework itself creates

LAYOUT_TABLE = {HNSW: "items", FRAMEWORK: "items", HNSW_BTREE: "items", PARTITIONED: "items_part"}
LAYOUT_ORDER = (HNSW, FRAMEWORK, HNSW_BTREE, PARTITIONED)

# plan cache modes: "custom" plans every query with its real tenant id; "generic"
# is what a driver that prepares statements ends up with (psycopg 3 after five
# runs, JDBC after five, asyncpg at once): one plan for every tenant
CUSTOM = "custom"
GENERIC = "generic"
PLAN_CACHE_MODES = (CUSTOM, GENERIC)
# "auto" leaves plan_cache_mode alone and runs a named prepared statement, so the
# server makes the custom-or-generic choice itself (after five custom plans), as it
# does for every driver that prepares; opt-in per strategy
AUTO = "auto"
ALL_PLAN_CACHE_MODES = (*PLAN_CACHE_MODES, AUTO)

JSONB_FILTER = "metadata->>'user_id' = %(user)s"  # the filter shape of JSONB-metadata stores

# SQL shapes
POST_FILTER = "post_filter"  # WHERE tenant_id = $1 ORDER BY embedding <=> $2 LIMIT k
RELAXED_CTE = "relaxed_cte"  # the same inside a MATERIALIZED CTE, re-sorted outside
EXACT = "exact"  # ORDER BY (embedding <=> $2) + 0 keeps the HNSW index out of the plan


@dataclass(frozen=True)
class Strategy:
    name: str
    label: str
    layout: str
    shape: str = POST_FILTER
    settings: dict[str, str] = field(default_factory=dict)
    filter: str = "tenant_id = %(tenant)s"
    # framework replays: the store's own statement, verbatim apart from table and column names
    query: str | None = None
    # indexes the framework creates (dropped again after the strategy), and other DDL
    # such as a SQL function; "{dim}" is replaced with the vector dimension
    indexes: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()
    # the plan cache modes this code path can really end up in
    modes: tuple[str, ...] = PLAN_CACHE_MODES
    # the statement calls a SQL function; its inner plan is read through auto_explain
    function_call: bool = False
    # run as this role (no BYPASSRLS), with the tenant id put in this setting before every query
    role: str | None = None
    tenant_setting: str | None = None

    @property
    def table(self) -> str:
        return LAYOUT_TABLE[self.layout]

    def sql(self) -> str:
        if self.query is not None:
            return self.query
        t = self.table
        if self.shape == POST_FILTER:
            return f"SELECT id, tenant_id FROM {t} WHERE {self.filter} ORDER BY embedding <=> %(q)s LIMIT %(k)s"
        if self.shape == RELAXED_CTE:
            return (
                "WITH r AS MATERIALIZED ("
                f"SELECT id, tenant_id, embedding <=> %(q)s AS d FROM {t} "
                f"WHERE {self.filter} ORDER BY d LIMIT %(k)s"
                ") SELECT id, tenant_id FROM r ORDER BY d"
            )
        if self.shape == EXACT:
            return f"SELECT id, tenant_id FROM {t} WHERE {self.filter} ORDER BY (embedding <=> %(q)s) + 0 LIMIT %(k)s"
        raise ValueError(f"unknown shape {self.shape}")

    def param_names(self) -> list[str]:
        """Placeholder names in order of first use: the $n order of dollar_sql()."""
        return list(dict.fromkeys(re.findall(r"%\((\w+)\)s", self.sql())))

    def dollar_sql(self) -> str:
        """The same statement with $n placeholders, for EXPLAIN (GENERIC_PLAN)."""
        numbers: dict[str, str] = {}

        def number(m: re.Match[str]) -> str:
            return numbers.setdefault(m.group(1), f"${len(numbers) + 1}")

        return re.sub(r"%\((\w+)\)s", number, self.sql())


STRATEGIES: list[Strategy] = [
    Strategy("hnsw_default", "HNSW, defaults (ef_search 40)", HNSW),
    Strategy("hnsw_ef200", "HNSW, ef_search 200", HNSW, settings={"hnsw.ef_search": "200"}),
    Strategy(
        "hnsw_ef1000",
        "HNSW, ef_search 1000 (max)",
        HNSW,
        settings={"hnsw.ef_search": "1000"},
    ),
    Strategy(
        "iter_strict",
        "iterative_scan strict_order",
        HNSW,
        settings={"hnsw.iterative_scan": "strict_order"},
    ),
    Strategy(
        "iter_relaxed",
        "iterative_scan relaxed_order",
        HNSW,
        RELAXED_CTE,
        settings={"hnsw.iterative_scan": "relaxed_order"},
    ),
    Strategy(
        "iter_relaxed_100k",
        "relaxed_order, max_scan_tuples 100k",
        HNSW,
        RELAXED_CTE,
        settings={
            "hnsw.iterative_scan": "relaxed_order",
            "hnsw.max_scan_tuples": "100000",
            "hnsw.scan_mem_multiplier": "4",
        },
    ),
    Strategy("jsonb_default", "JSONB filter, HNSW defaults", HNSW, filter=JSONB_FILTER),
    Strategy(
        "jsonb_relaxed",
        "JSONB filter, relaxed_order",
        HNSW,
        RELAXED_CTE,
        settings={"hnsw.iterative_scan": "relaxed_order"},
        filter=JSONB_FILTER,
    ),
    Strategy("btree_planner", "+ B-tree on tenant_id, planner picks", HNSW_BTREE),
    Strategy(
        "btree_planner_relaxed",
        "+ B-tree, planner picks, relaxed_order",
        HNSW_BTREE,
        RELAXED_CTE,
        settings={"hnsw.iterative_scan": "relaxed_order"},
    ),
    Strategy("btree_exact", "+ B-tree, exact (ORDER BY dist + 0)", HNSW_BTREE, EXACT),
    Strategy("jsonb_expr_index", "JSONB filter + expression B-tree", HNSW_BTREE, filter=JSONB_FILTER),
    Strategy("part16_default", "16 hash partitions, HNSW each", PARTITIONED),
    Strategy(
        "part16_relaxed",
        "16 hash partitions, relaxed_order",
        PARTITIONED,
        RELAXED_CTE,
        settings={"hnsw.iterative_scan": "relaxed_order"},
    ),
]

BY_NAME = {s.name: s for s in STRATEGIES}

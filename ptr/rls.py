"""Tenant isolation through Row Level Security instead of a WHERE clause.

The app runs as a role without BYPASSRLS, puts the tenant in a setting before each
query (as Supabase does with the JWT, or an app with `set_config('app.tenant_id', ...)`)
and sends the search with no tenant filter at all; the policy adds it.

Two policy spellings are measured:

* `current_setting(...)` called straight in the policy;
* the same wrapped in a scalar subquery, `(SELECT current_setting(...))`. Supabase's RLS
  performance guide recommends this wrap for `auth.uid()`, so the function runs once
  per statement as an InitPlan instead of once per row.
"""

from __future__ import annotations

from .strategies import AUTO, HNSW, HNSW_BTREE, PLAN_CACHE_MODES, Strategy

ROLE = "ptr_app"
SETTING = "app.tenant_id"
POLICY = "ptr_tenant"

DIRECT = f"tenant_id = current_setting('{SETTING}')::int"
INITPLAN = f"tenant_id = (SELECT current_setting('{SETTING}')::int)"

_SEARCH = "SELECT id, tenant_id FROM items ORDER BY embedding <=> %(q)s LIMIT %(k)s"
_SEARCH_RELAXED = (
    "WITH r AS MATERIALIZED ("
    "SELECT id, tenant_id, embedding <=> %(q)s AS d FROM items ORDER BY d LIMIT %(k)s"
    ") SELECT id, tenant_id FROM r ORDER BY d"
)


def policy_ddl(using: str) -> tuple[str, ...]:
    """Role, grant and one SELECT policy on items; re-runnable, replaces the previous policy."""
    return (
        f"DO $$BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{ROLE}') "
        f"THEN CREATE ROLE {ROLE} NOLOGIN; END IF; END$$",
        f"GRANT SELECT ON items TO {ROLE}",
        "ALTER TABLE items ENABLE ROW LEVEL SECURITY",
        f"DROP POLICY IF EXISTS {POLICY} ON items",
        f"CREATE POLICY {POLICY} ON items FOR SELECT TO {ROLE} USING ({using})",
    )


MODES = (*PLAN_CACHE_MODES, AUTO)


def _strategies(layout: str, suffix: str, where: str) -> list[Strategy]:
    common = {"role": ROLE, "tenant_setting": SETTING, "modes": MODES}
    return [
        Strategy(
            f"rls_direct{suffix}",
            f"RLS current_setting(), {where}",
            layout,
            query=_SEARCH,
            setup=policy_ddl(DIRECT),
            **common,
        ),
        Strategy(
            f"rls_initplan{suffix}",
            f"RLS (SELECT current_setting()), {where}",
            layout,
            query=_SEARCH,
            setup=policy_ddl(INITPLAN),
            **common,
        ),
        Strategy(
            f"rls_initplan_relaxed{suffix}",
            f"RLS (SELECT ...), {where}, relaxed_order",
            layout,
            query=_SEARCH_RELAXED,
            settings={"hnsw.iterative_scan": "relaxed_order"},
            setup=policy_ddl(INITPLAN),
            **common,
        ),
    ]


# HNSW only: nobody indexed the policy column; HNSW_BTREE: a B-tree on tenant_id,
# as the Supabase RLS guide advises for policy columns
RLS_STRATEGIES: list[Strategy] = _strategies(HNSW, "_noindex", "no B-tree") + _strategies(
    HNSW_BTREE, "", "B-tree on tenant_id"
)

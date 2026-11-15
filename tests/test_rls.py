from ptr.rls import DIRECT, INITPLAN, POLICY, RLS_STRATEGIES, ROLE, SETTING, policy_ddl
from ptr.strategies import AUTO, HNSW, HNSW_BTREE, Strategy


def test_rls_searches_carry_no_tenant_filter_of_their_own():
    for s in RLS_STRATEGIES:
        sql = s.sql()
        assert "WHERE" not in sql and "tenant_id =" not in sql, s.name
        assert s.role == ROLE and s.tenant_setting == SETTING
        assert AUTO in s.modes


def test_every_policy_spelling_runs_with_and_without_the_btree():
    by_layout = {layout: {s.setup[-1] for s in RLS_STRATEGIES if s.layout == layout} for layout in (HNSW, HNSW_BTREE)}
    assert by_layout[HNSW] == by_layout[HNSW_BTREE]
    assert any(DIRECT in ddl for ddl in by_layout[HNSW]) and any(INITPLAN in ddl for ddl in by_layout[HNSW])


def test_policy_ddl_replaces_the_previous_policy_before_creating_one():
    ddl = policy_ddl(DIRECT)
    drop = ddl.index(f"DROP POLICY IF EXISTS {POLICY} ON items")
    create = next(i for i, d in enumerate(ddl) if d.startswith("CREATE POLICY"))
    assert drop < create == len(ddl) - 1
    assert f"TO {ROLE} USING ({DIRECT})" in ddl[create]
    assert all("{" not in d for d in ddl)  # setup DDL goes through str.format(dim=...)


def test_param_names_follow_dollar_sql_numbering():
    s = Strategy("t", "t", HNSW, filter="tenant_id = %(tenant)s AND metadata->>'u' = %(user)s")
    assert s.param_names() == ["tenant", "user", "q", "k"]
    assert s.dollar_sql().index("$1") < s.dollar_sql().index("$2") < s.dollar_sql().index("$3")
    rls = RLS_STRATEGIES[0]
    assert rls.param_names() == ["q", "k"] and "$3" not in rls.dollar_sql()

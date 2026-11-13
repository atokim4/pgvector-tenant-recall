from ptr.strategies import EXACT, JSONB_FILTER, RELAXED_CTE, STRATEGIES, Strategy


def test_every_strategy_filters_by_tenant_and_limits():
    for s in STRATEGIES:
        sql = s.sql()
        assert f"WHERE {s.filter}" in sql
        assert s.filter in ("tenant_id = %(tenant)s", JSONB_FILTER)
        assert "LIMIT %(k)s" in sql
        assert "%(" not in s.dollar_sql()


def test_relaxed_order_is_resorted_outside_the_cte():
    for s in STRATEGIES:
        if s.settings.get("hnsw.iterative_scan") == "relaxed_order":
            assert s.shape == RELAXED_CTE, s.name
            assert s.sql().endswith("ORDER BY d")


def test_exact_shape_hides_the_order_by_from_hnsw():
    exact = [s for s in STRATEGIES if s.shape == EXACT]
    assert exact and all("<=> %(q)s) + 0" in s.sql() for s in exact)


def test_names_are_unique():
    names = [s.name for s in STRATEGIES]
    assert len(names) == len(set(names))


def test_dollar_sql_numbers_placeholders_in_order_and_reuses_them():
    s = Strategy("t", "t", "hnsw", filter="tenant_id = %(tenant)s AND tenant_id = %(tenant)s")
    assert s.dollar_sql() == (
        "SELECT id, tenant_id FROM items WHERE tenant_id = $1 AND tenant_id = $1 ORDER BY embedding <=> $2 LIMIT $3"
    )


def test_every_framework_replay_cites_its_source():
    from ptr.frameworks import FRAMEWORK_STRATEGIES, SOURCES

    variants = {"mem0_exprindex_boundkey", "supabase_match_documents_gin"}
    assert set(SOURCES) == {s.name for s in FRAMEWORK_STRATEGIES} - variants


def test_framework_indexes_are_named_and_dropped_by_name():
    from ptr.frameworks import FRAMEWORK_STRATEGIES

    for s in FRAMEWORK_STRATEGIES:
        for ddl in s.indexes:
            assert ddl.startswith("CREATE INDEX fw_"), ddl


def test_framework_queries_only_use_known_parameters():
    import re

    from ptr.bench import params_for
    from ptr.frameworks import FRAMEWORK_STRATEGIES

    known = set(params_for(1, None, 10))
    for s in FRAMEWORK_STRATEGIES:
        assert set(re.findall(r"%\((\w+)\)s", s.sql())) <= known, s.name

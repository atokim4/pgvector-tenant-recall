import numpy as np
import pytest

from ptr.realdata import RealConfig, paragraphs, plan_layout


def fake_articles(n: int = 3_000, seed: int = 1) -> list[tuple[str, list[str]]]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        size = int(rng.choice([1, 1, 2, 3, 5, 10, 10, 25, 60]))
        out.append((f"Article {i}", [f"article {i} paragraph {j}" for j in range(size)]))
    return out


CFG = RealConfig(rows=8_000, background_tenants=40, queries_per_bucket=4, probe_buckets={1: 3, 10: 3, 100: 2})


def test_paragraphs_drops_short_lines_and_truncates():
    text = "short\n\n" + "a" * 100 + "\n  " + "b" * 90 + "  \n"
    assert paragraphs(text, 80, 50) == ["a" * 50, "b" * 50]


def test_probe_tenants_have_exact_sizes_and_rows_add_up():
    layout = plan_layout(fake_articles(), CFG)
    assert len(layout.chunks) == CFG.rows == len(layout.tenant_of_row)
    assert sum(layout.tenant_size.values()) == CFG.rows
    for bucket, ids in layout.probe_tenants.items():
        assert len(ids) == CFG.probe_buckets[bucket]
        assert all(layout.tenant_size[t] == bucket for t in ids)


def test_small_probe_tenants_own_one_whole_article():
    layout = plan_layout(fake_articles(), CFG)
    for bucket in (1, 10):
        for t in layout.probe_tenants[bucket]:
            rows = [c for c, o in zip(layout.chunks, layout.tenant_of_row, strict=True) if o == t]
            assert len({r.split(" paragraph")[0] for r in rows}) == 1


def test_queries_name_one_of_the_tenants_own_articles():
    layout = plan_layout(fake_articles(), CFG)
    assert len(layout.queries) == 3 * CFG.queries_per_bucket
    for _, tenant, text in layout.queries:
        title = text.removeprefix("What is ").removesuffix("?")
        own = {
            c.split(" paragraph")[0] for c, o in zip(layout.chunks, layout.tenant_of_row, strict=True) if o == tenant
        }
        assert title.lower() in own


def test_layout_is_deterministic():
    a, b = plan_layout(fake_articles(), CFG), plan_layout(fake_articles(), CFG)
    assert a.chunks == b.chunks and a.queries == b.queries


def test_too_few_rows_is_an_error():
    with pytest.raises(ValueError):
        plan_layout(fake_articles(), RealConfig(rows=300, background_tenants=200, probe_buckets={100: 2}))

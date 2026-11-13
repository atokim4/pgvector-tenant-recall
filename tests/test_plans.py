import json

from ptr.bench import NestedPlans, compact_plan


def node(kind, children=(), index=None):
    n = {"Node Type": kind, "Plans": list(children)}
    if index:
        n["Index Name"] = index
    return n


def test_hnsw_ordering_is_recognised():
    plan, access = compact_plan(node("Limit", [node("Index Scan", index="items_hnsw")]))
    assert plan == "Limit > Index Scan(items_hnsw)"
    assert access == "hnsw"


def test_partition_scans_collapse_and_count_as_hnsw():
    parts = [node("Index Scan", index=f"items_part_{i}_embedding_idx") for i in range(3)]
    plan, access = compact_plan(node("Limit", [node("Merge Append", parts)]))
    assert plan.count("Index Scan") == 3  # different index names stay visible
    assert access == "hnsw"


def test_tenant_index_and_seq_scan():
    gin = node("Limit", [node("Sort", [node("Bitmap Heap Scan", [node("Bitmap Index Scan", index="fw_meta_gin")])])])
    assert compact_plan(gin)[1] == "btree"
    assert compact_plan(node("Limit", [node("Sort", [node("Seq Scan")])]))[1] == "seq"
    assert compact_plan(node("Function Scan"))[1] == "function"


class FakeConn:
    def __init__(self):
        self.handler = None

    def execute(self, *args):
        return None

    def add_notice_handler(self, fn):
        self.handler = fn


class Notice:
    def __init__(self, text):
        self.message_primary = text


def test_nested_plans_take_the_inner_statement():
    conn = FakeConn()
    nested = NestedPlans(conn)
    inner = {"Query Text": "SELECT id FROM items WHERE metadata @> filter", "Plan": node("Limit", [node("Seq Scan")])}
    outer = {"Query Text": "SELECT id FROM match_documents($1)", "Plan": node("Function Scan")}
    conn.handler(Notice("duration: 0.1 ms  plan:\n" + json.dumps(inner)))
    conn.handler(Notice("duration: 0.2 ms  plan:\n" + json.dumps(outer)))
    assert nested.take() == ("Limit > Seq Scan", "seq")
    assert nested.take() == ("unknown", "function")  # cleared after each take

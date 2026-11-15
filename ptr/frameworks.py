"""Filtered searches replayed the way each framework's pgvector store sends them.

Each query below was copied from the store's current source (version and commit
in `SOURCES`) and only renamed onto the bench table:

* `items.metadata` (jsonb) stands for the store's JSONB metadata column
  (langchain `cmetadata`, mem0 `payload`, haystack `meta`, vecs `metadata`).
* `items.meta_json` (json) stands for plain-`json` metadata columns
  (langchain-postgres v2 `langchain_metadata`, llama-index `metadata_` by default).
* `items.collection_id` + `collections` stand for langchain's collection join.

The tenant is `metadata.user_id`, the way these stores are used for per-user
memory or per-customer RAG. `modes` says which plan cache modes the code path can
really reach with its driver (see `Source.prepared`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .strategies import CUSTOM, FRAMEWORK, GENERIC, Strategy


@dataclass(frozen=True)
class Source:
    framework: str
    version: str  # latest release at the time of reading
    commit: str  # main-branch commit that was read
    where: str  # file and lines
    filter_sql: str  # the WHERE clause the store emits for {"user_id": ...}
    ann_index: str  # what vector index the store creates
    prepared: str  # driver, and whether one plan ends up shared by every tenant


SOURCES: dict[str, Source] = {
    "langchain_pgvector": Source(
        "langchain-postgres PGVector",
        "0.0.19",
        "9afb650a",
        "langchain_postgres/vectorstores.py L1132-1135, L1394-1431",
        "collection_id = :uuid AND jsonb_path_match(cmetadata, '$.user_id == $value', :vars)",
        "none (GIN jsonb_path_ops on cmetadata only)",
        "psycopg 3 via SQLAlchemy, default prepare_threshold: stable SQL text, so prepared after 5 runs",
    ),
    "langchain_pgvectorstore": Source(
        "langchain-postgres PGVectorStore (v2)",
        "0.0.19",
        "9afb650a",
        "langchain_postgres/v2/async_vectorstore.py L691-703, L1276-1312",
        "langchain_metadata->>'user_id' = :user_id_<random> (json column)",
        "opt-in apply_vector_index(HNSWIndex()), m=16 ef_construction=64",
        "bind name gets a random suffix per call, so a prepared plan is never reused: custom plans",
    ),
    "llamaindex": Source(
        "llama-index PGVectorStore",
        "0.9.0",
        "baeabbfe",
        "llama_index/vector_stores/postgres/base.py L690-752, L794-866",
        "metadata_->>'user_id' = 'u1' (value inlined, json column by default)",
        "opt-in hnsw_kwargs; also SET LOCAL enable_bitmapscan = off",
        "psycopg2 / asyncpg, value inlined as a literal: custom plans",
    ),
    "llamaindex_indexed_key": Source(
        "llama-index PGVectorStore + indexed_metadata_keys={('user_id','text')}",
        "0.9.0",
        "baeabbfe",
        "base.py L104-163 (btree on CAST(metadata_->>key AS VARCHAR))",
        "metadata_->>'user_id' = 'u1'",
        "opt-in hnsw_kwargs + the opt-in B-tree on the key",
        "literal: custom plans",
    ),
    "haystack": Source(
        "haystack PgvectorDocumentStore (search_strategy='hnsw')",
        "8.2.1",
        "8515da62",
        "integrations/pgvector/.../document_store.py L1590-1615, filters.py L160-195",
        "meta->>'user_id' = %s",
        "default exact_nearest_neighbor (no ANN index); HNSW opt-in",
        "psycopg 3, vector and k inlined as literals: SQL text unique per query, custom plans",
    ),
    "mem0": Source(
        "mem0 PGVector",
        "mem0ai 2.2.1",
        "f1791daa",
        "mem0/vector_stores/pgvector.py L37, L113, L291-298, L356-363",
        "payload->>%s = %s (the key is a bind parameter too)",
        "HNSW vector_cosine_ops by default (hnsw=True)",
        "psycopg 3 ConnectionPool, default prepare_threshold: prepared after 5 runs",
    ),
    "mem0_pr7607": Source(
        "mem0 with the change from closed PR #7607",
        "mem0ai 2.2.1",
        "PR #7607 (closed unmerged 2026-10-11)",
        "B-tree on (payload->>'user_id'), ef_search = max(top_k, 64), filter key inlined",
        "payload->>'user_id' = %s",
        "HNSW + expression B-tree",
        "psycopg 3, prepared after 5 runs",
    ),
    "vecs": Source(
        "Supabase vecs",
        "0.4.5",
        "240d8709",
        "src/vecs/collection.py L547-569, L887-936",
        "metadata -> 'user_id' = CAST('\"u1\"' AS JSONB) (string $eq)",
        "explicit create_index(): HNSW m=16 ef_construction=64",
        "SQLAlchemy + psycopg2: custom plans",
    ),
    "supabase_match_documents": Source(
        "Supabase docs match_documents() (LangChain guide)",
        "docs",
        "supabase/supabase 64fad1a7",
        "apps/docs/content/guides/ai/langchain.mdx L37-68",
        "metadata @> filter, inside a plpgsql function",
        "none on the page",
        "plpgsql caches the statement: after 5 calls it switches to a generic plan if that looks cheaper",
    ),
    "langchainjs_n8n": Source(
        "@langchain/community PGVectorStore (used by the n8n PGVector node)",
        "1.1.27 (n8n 2.47.7 pin)",
        "n8n f9dcb57a",
        "libs/langchain-community/src/vectorstores/pgvector.ts L811, L953-957",
        "metadata->>'user_id' = $n",
        "only via an explicit createHnswIndex(); n8n never calls it",
        "node-postgres without a statement name: never prepared, custom plans",
    ),
    "spring_ai": Source(
        "Spring AI PgVectorStore",
        "2.0.2",
        "703119a1",
        "PgVectorStore.java L490-496, L780; PgVectorFilterExpressionConverter.java L106, L117-120",
        'embedding <=> ? < 1 AND metadata::jsonb @@ \'$."user_id" == "u1"\'::jsonpath (value inlined)',
        "HNSW by default (no WITH options) when initializeSchema(true)",
        "PgJDBC, filter value inlined into the SQL text: custom plans",
    ),
    "semantic_kernel_py": Source(
        "Semantic Kernel Python Postgres connector",
        "semantic-kernel 1.47.0",
        "3bbde09a",
        "python/semantic_kernel/connectors/postgres.py L768, L858-881, L957",
        "(tenant_id = %s) on a typed column",
        "HNSW by default (no WITH options)",
        "psycopg 3 AsyncConnectionPool, default prepare_threshold: prepared after 5 runs",
    ),
}

GIN_PATH_OPS = "CREATE INDEX fw_meta_gin ON items USING gin (metadata jsonb_path_ops)"

MATCH_DOCUMENTS = """
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector({dim}),
  match_count int DEFAULT null,
  filter jsonb DEFAULT '{{}}'
) RETURNS TABLE (id bigint, tenant_id int, similarity float)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT id, tenant_id, 1 - (items.embedding <=> query_embedding) AS similarity
  FROM items
  WHERE metadata @> filter
  ORDER BY items.embedding <=> query_embedding
  LIMIT match_count;
END;
$$
"""

FRAMEWORK_STRATEGIES: list[Strategy] = [
    Strategy(
        "langchain_pgvector",
        "langchain-postgres PGVector",
        FRAMEWORK,
        query=(
            "SELECT e.id, e.tenant_id, e.embedding <=> %(q)s AS distance "
            "FROM items e JOIN collections c ON e.collection_id = c.uuid "
            "WHERE e.collection_id = CAST(%(coll)s AS uuid) "
            "AND jsonb_path_match(e.metadata, CAST('$.user_id == $value' AS JSONPATH), CAST(%(jpvars)s AS JSONB)) "
            "ORDER BY distance ASC LIMIT %(k)s"
        ),
        indexes=(GIN_PATH_OPS,),
    ),
    Strategy(
        "langchain_pgvectorstore",
        "langchain-postgres PGVectorStore (v2)",
        FRAMEWORK,
        query=(
            "SELECT id, tenant_id, embedding <=> %(q)s AS distance FROM items "
            "WHERE meta_json->>'user_id' = %(user)s ORDER BY embedding <=> %(q)s LIMIT %(k)s"
        ),
        settings={"hnsw.ef_search": "40"},
        modes=(CUSTOM,),
    ),
    Strategy(
        "llamaindex",
        "llama-index PGVectorStore",
        FRAMEWORK,
        query=(
            "SELECT id, tenant_id, embedding <=> %(q)s AS distance FROM items "
            "WHERE meta_json->>'user_id' = %(user)s ORDER BY distance asc LIMIT %(k)s"
        ),
        settings={"hnsw.ef_search": "40", "enable_bitmapscan": "off"},
        modes=(CUSTOM,),
    ),
    Strategy(
        "llamaindex_indexed_key",
        "llama-index + indexed_metadata_keys",
        FRAMEWORK,
        query=(
            "SELECT id, tenant_id, embedding <=> %(q)s AS distance FROM items "
            "WHERE meta_json->>'user_id' = %(user)s ORDER BY distance asc LIMIT %(k)s"
        ),
        settings={"hnsw.ef_search": "40", "enable_bitmapscan": "off"},
        indexes=("CREATE INDEX fw_li_user ON items ((CAST(meta_json->>'user_id' AS VARCHAR)))",),
        modes=(CUSTOM,),
    ),
    Strategy(
        "haystack",
        "haystack PgvectorDocumentStore, hnsw",
        FRAMEWORK,
        query=(
            "SELECT id, tenant_id, 1 - (embedding <=> %(q)s) AS score FROM items "
            "WHERE metadata->>'user_id' = %(user)s ORDER BY embedding <=> %(q)s ASC LIMIT %(k)s"
        ),
        modes=(CUSTOM,),
    ),
    Strategy(
        "mem0",
        "mem0 PGVector",
        FRAMEWORK,
        query=(
            "SELECT id, tenant_id, embedding <=> %(q)s::vector AS distance FROM items "
            "WHERE metadata->>%(key)s = %(user)s ORDER BY distance LIMIT %(k)s"
        ),
    ),
    Strategy(
        "mem0_exprindex_boundkey",
        "mem0 + B-tree on payload->>'user_id', key still bound",
        FRAMEWORK,
        query=(
            "SELECT id, tenant_id, embedding <=> %(q)s::vector AS distance FROM items "
            "WHERE metadata->>%(key)s = %(user)s ORDER BY distance LIMIT %(k)s"
        ),
        indexes=("CREATE INDEX fw_mem0_user ON items ((metadata->>'user_id'))",),
    ),
    Strategy(
        "mem0_pr7607",
        "mem0 + PR #7607 (B-tree, literal key, ef_search 64)",
        FRAMEWORK,
        query=(
            "SELECT id, tenant_id, embedding <=> %(q)s::vector AS distance FROM items "
            "WHERE metadata->>'user_id' = %(user)s ORDER BY distance LIMIT %(k)s"
        ),
        settings={"hnsw.ef_search": "64"},
        indexes=("CREATE INDEX fw_mem0_user ON items ((metadata->>'user_id'))",),
    ),
    Strategy(
        "vecs",
        "Supabase vecs",
        FRAMEWORK,
        query=(
            "SELECT id, tenant_id FROM items WHERE metadata -> 'user_id' = CAST(%(userjson)s AS JSONB) "
            "ORDER BY embedding <=> %(q)s LIMIT %(k)s"
        ),
        settings={"hnsw.ef_search": "40", "ivfflat.probes": "10"},
        modes=(CUSTOM,),
    ),
    Strategy(
        "supabase_match_documents",
        "Supabase match_documents() (metadata @> filter)",
        FRAMEWORK,
        query="SELECT id, tenant_id FROM match_documents(%(q)s, %(k)s, CAST(%(filter)s AS jsonb))",
        setup=(MATCH_DOCUMENTS,),
        function_call=True,
    ),
    Strategy(
        "supabase_match_documents_gin",
        "match_documents() + GIN jsonb_path_ops",
        FRAMEWORK,
        query="SELECT id, tenant_id FROM match_documents(%(q)s, %(k)s, CAST(%(filter)s AS jsonb))",
        setup=(MATCH_DOCUMENTS,),
        function_call=True,
        indexes=(GIN_PATH_OPS,),
    ),
    Strategy(
        "langchainjs_n8n",
        "langchainjs PGVectorStore / n8n node, HNSW added",
        FRAMEWORK,
        query=(
            'SELECT id, tenant_id, embedding <=> %(q)s AS "_distance" FROM items '
            "WHERE metadata->>'user_id' = %(user)s ORDER BY \"_distance\" ASC LIMIT %(k)s"
        ),
        modes=(CUSTOM,),
    ),
    Strategy(
        "spring_ai",
        "Spring AI PgVectorStore",
        FRAMEWORK,
        query=(
            "SELECT id, tenant_id, embedding <=> %(q)s AS distance FROM items "
            "WHERE embedding <=> %(q)s < %(threshold)s AND meta_json::jsonb @@ CAST(%(jsonpath)s AS jsonpath) "
            "ORDER BY distance LIMIT %(k)s"
        ),
        modes=(CUSTOM,),
    ),
    Strategy(
        "semantic_kernel_py",
        "Semantic Kernel (Python), typed tenant column",
        FRAMEWORK,
        query=(
            "SELECT id, tenant_id, embedding <=> %(q)s as dist FROM items "
            "WHERE (tenant_id = %(tenant)s) ORDER BY dist LIMIT %(k)s"
        ),
    ),
]

# The plan cache mode each replay settles into after a few calls with its real
# driver: prepared drivers end on the generic plan; for match_documents() the
# "custom" run is plpgsql's own auto mode, which is what Supabase users get.
STEADY_MODE: dict[str, str] = {
    s.name: (
        GENERIC
        if GENERIC in s.modes and s.name not in ("supabase_match_documents", "supabase_match_documents_gin")
        else CUSTOM
    )
    for s in FRAMEWORK_STRATEGIES
}

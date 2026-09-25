# Scaling notes

The current setup (embedded Qdrant, SQLite checkpointer) is intentionally the
simplest thing that works correctly — it's meant for a single demo instance,
not concurrent production traffic. This document is a short, honest list of
what would need to change, and why it isn't done here.

## 1. Vector store: Qdrant embedded → Qdrant Cloud / self-hosted cluster

**Now:** `QdrantClient(path="./qdrant_storage")` — a file-backed local instance.
Zero setup, but it's a single-process file lock, so it won't work if the app
is scaled to multiple workers or containers.

**Later:** point the same client at a managed or self-hosted Qdrant instance.
This is a config change, not a rewrite, since `core/tools.py` only talks to
the `QdrantClient` object:

```python
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
```

## 2. Checkpointer: SQLite → PostgreSQL

**Now:** `SqliteSaver` writing to a local `customer_support.db` file. Fine for
one process; will hit "database is locked" errors under concurrent writers.

**Later:** swap in `PostgresSaver` from `langgraph-checkpoint-postgres`,
backed by a managed Postgres instance with connection pooling:

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

pool = ConnectionPool(conninfo=os.getenv("DATABASE_URL"), max_size=20)
checkpointer = PostgresSaver(pool)
checkpointer.setup()
```

## 3. Things that aren't code changes, just operational concerns

- **Cost control:** repeated identical questions (e.g. "what's the torque spec
  for M8 lugs") are a good candidate for a semantic cache (Redis + cosine
  similarity threshold) to avoid re-paying for the same LLM call.
- **Rate limiting:** per-user request limits at the API gateway layer, to
  avoid one user driving up the OpenRouter bill.
- **Old sessions:** checkpointer threads older than N days could be archived
  or dropped on a schedule instead of growing the DB indefinitely.

None of this is implemented — it's a plan, not a claim.

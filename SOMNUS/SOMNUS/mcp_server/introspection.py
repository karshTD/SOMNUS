"""Introspection: why does the agent believe what it believes?

Uses crdb_internal for stats (pg_total_relation_size is not implemented in
CockroachDB and errors out) and AS OF SYSTEM TIME for belief time travel.
"""

from __future__ import annotations

import json
from typing import Any

from memory.store import MemoryStore


def schema_info(store: MemoryStore) -> dict[str, Any]:
    if not hasattr(store, "connection"):
        return {"backend": "in-memory", "tables": ["episodes", "schemas", "provenance"]}
    query = """
        SELECT table_name, column_name, data_type
          FROM information_schema.columns
         WHERE table_name IN ('episodes','schemas','provenance','prediction_errors','contexts')
         ORDER BY table_name, ordinal_position
    """
    with store.connection() as conn, conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(query)
        tables: dict[str, list[dict[str, str]]] = {}
        for table, column, dtype in cur.fetchall():
            tables.setdefault(table, []).append({"name": column, "type": dtype})
    return {"backend": "cockroachdb", "tables": tables}


def table_stats(store: MemoryStore) -> dict[str, Any]:
    if hasattr(store, "table_stats"):
        return store.table_stats()  # type: ignore[attr-defined]
    return {
        "live_episodes": store.episode_count(),
        "schemas": store.schema_count(),
        "backend": "in-memory",
    }


def explain_belief(store: MemoryStore, embedding: list[float], limit: int = 3) -> dict[str, Any]:
    """Schema -> provenance -> episodes. 'Why do you believe this?' is a JOIN."""
    matches = store.recall_schemas(embedding, limit=limit)
    out = []
    for match in matches:
        episode_ids = store.provenance_for(match.schema.id)
        out.append(
            {
                "schema": match.schema.to_dict(),
                "similarity": round(match.similarity, 4),
                "supporting_episodes": len(episode_ids),
                "episode_sample": episode_ids[:5],
            }
        )
    return {"matches": out}


def beliefs_as_of(store: MemoryStore, timestamp: str = "-1h") -> dict[str, Any]:
    """MVCC time travel. Bounded by the table GC window (default ~25h)."""
    if not hasattr(store, "beliefs_as_of"):
        return {"error": "time travel requires the CockroachDB backend"}
    try:
        return {"as_of": timestamp, "beliefs": store.beliefs_as_of(timestamp)}  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        return {
            "error": str(exc),
            "hint": "AS OF SYSTEM TIME cannot look back further than gc.ttlseconds "
            "(default ~25h). Raise it on the schemas table for longer history.",
        }


def agent_state(state: dict[str, Any] | None) -> str:
    return json.dumps(state or {"status": "unknown"}, indent=2, default=str)

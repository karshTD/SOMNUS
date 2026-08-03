"""Database and agent state introspection tools for MCP."""

from __future__ import annotations

import json
from typing import Any

from memory.cortex import Cortex


def get_schema_info(cortex: Cortex) -> dict[str, Any]:
    """Return CockroachDB schema metadata for semantic_memory."""
    query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'semantic_memory'
        ORDER BY ordinal_position
    """
    with cortex.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            columns = [
                {"name": row[0], "type": row[1], "nullable": row[2]}
                for row in cur.fetchall()
            ]
    return {"table": "semantic_memory", "columns": columns}


def get_table_stats(cortex: Cortex) -> dict[str, Any]:
    """Return row counts and storage hints for meta-plasticity decisions."""
    with cortex.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM semantic_memory")
            row_count = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT pg_total_relation_size('semantic_memory') AS bytes
                """
            )
            size_bytes = int(cur.fetchone()[0] or 0)

    return {
        "semantic_memory_rows": row_count,
        "semantic_memory_bytes": size_bytes,
        "density_high": row_count >= 10_000,
    }


def get_agent_state(agent_state: dict[str, Any] | None) -> str:
    """Serialize current agent wake state for LLM consumption."""
    return json.dumps(agent_state or {"status": "unknown"}, indent=2)

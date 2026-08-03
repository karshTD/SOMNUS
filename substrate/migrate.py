"""Database schema initialization for CockroachDB + pgvector."""

from __future__ import annotations

import os
import sys

import psycopg2

from infra.aws_client import EMBEDDING_DIMENSION

MIGRATION_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semantic_memory (
    id SERIAL PRIMARY KEY,
    rule_text TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIMENSION}) NOT NULL,
    source_episodes TEXT[] DEFAULT '{{}}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS semantic_memory_embedding_idx
    ON semantic_memory USING hnsw (embedding vector_cosine_ops);
"""


def migrate(db_url: str | None = None) -> None:
    """Connect to CockroachDB and apply schema migrations."""
    url = db_url or os.getenv("COCKROACH_DB_URL")
    if not url:
        raise ValueError("COCKROACH_DB_URL is required for migration")

    conn = psycopg2.connect(url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(MIGRATION_SQL)
        print("Migration complete: semantic_memory table ready.")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate(sys.argv[1] if len(sys.argv) > 1 else None)

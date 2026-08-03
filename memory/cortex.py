"""Slow semantic memory interface backed by CockroachDB + pgvector."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from infra.aws_client import EMBEDDING_DIMENSION


@dataclass
class SemanticRule:
    id: int
    rule_text: str
    source_episodes: list[str]
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rule_text": self.rule_text,
            "source_episodes": self.source_episodes,
            "similarity": round(self.similarity, 4),
        }


class Cortex:
    """Connection pool and pgvector cosine recall for generalized rules."""

    def __init__(self, db_url: str | None = None, min_conn: int = 1, max_conn: int = 5) -> None:
        self.db_url = db_url or os.getenv("COCKROACH_DB_URL", "")
        self._pool: pool.SimpleConnectionPool | None = None
        if self.db_url:
            self._pool = pool.SimpleConnectionPool(min_conn, max_conn, self.db_url)

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        if not self._pool:
            raise RuntimeError("COCKROACH_DB_URL is not configured")
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def recall_similar(self, vector: list[float], limit: int = 5) -> list[SemanticRule]:
        """Fetch generalized rules by cosine distance (<=>)."""
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(f"Expected vector dimension {EMBEDDING_DIMENSION}")

        query = """
            SELECT id, rule_text, source_episodes,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM semantic_memory
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        vector_literal = "[" + ",".join(str(v) for v in vector) + "]"

        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (vector_literal, vector_literal, limit))
                rows = cur.fetchall()

        return [
            SemanticRule(
                id=row["id"],
                rule_text=row["rule_text"],
                source_episodes=row.get("source_episodes") or [],
                similarity=float(row["similarity"]),
            )
            for row in rows
        ]

    def insert_rule(
        self,
        rule_text: str,
        embedding: list[float],
        source_episodes: list[str] | None = None,
    ) -> int:
        """Insert a consolidated semantic rule."""
        vector_literal = "[" + ",".join(str(v) for v in embedding) + "]"
        query = """
            INSERT INTO semantic_memory (rule_text, embedding, source_episodes)
            VALUES (%s, %s::vector, %s)
            RETURNING id
        """
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (rule_text, vector_literal, source_episodes or []))
                row = cur.fetchone()
                return int(row[0])

    def row_count(self) -> int:
        """Return total rows in semantic_memory (for meta-plasticity checks)."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM semantic_memory")
                return int(cur.fetchone()[0])

    def health_check(self) -> bool:
        """Verify database connectivity."""
        if not self._pool:
            return False
        try:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._pool:
            self._pool.closeall()
            self._pool = None

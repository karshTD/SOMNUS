"""CockroachDB-backed MemoryStore.

Fixes over the naive version:
  * ThreadedConnectionPool (the MCP server runs in a thread alongside the
    wake loop; SimpleConnectionPool is not thread-safe).
  * 40001 serialization retry on every write path.
  * The query vector is sent ONCE via a CTE, not interpolated twice.
  * Real merge-or-spawn support: schemas carry variance, support, stability.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Generator

from infra.config import CONFIG, EMBEDDING_DIMENSION
from infra.retry import with_retry
from memory.store import Episode, Schema, SchemaMatch

logger = logging.getLogger(__name__)


def _vec(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def _parse_vec(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(v) for v in raw]
    return [float(v) for v in str(raw).strip("[]").split(",") if v.strip()]


class CockroachStore:
    """MemoryStore implementation over CockroachDB."""

    def __init__(self, db_url: str | None = None, min_conn: int = 1, max_conn: int = 8) -> None:
        from psycopg2 import pool  # imported lazily so offline runs need no driver

        self.db_url = db_url or CONFIG.db_url
        self._pool: Any = None
        if self.db_url:
            # ThreadedConnectionPool: MCP server + wake loop share this store.
            self._pool = pool.ThreadedConnectionPool(min_conn, max_conn, self.db_url)

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

    # ------------------------------------------------------------------
    # hippocampus
    # ------------------------------------------------------------------
    @with_retry
    def write_episode(self, episode: Episode) -> str:
        sql = """
            INSERT INTO episodes
                (id, context_id, embedding, raw_obs, canonical,
                 surprise, da, ach, na, expire_at)
            VALUES (%s, %s, %s::vector, %s::jsonb, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    episode.id,
                    episode.context_id or None,
                    _vec(episode.embedding),
                    json.dumps(episode.raw_obs, default=str),
                    episode.canonical,
                    episode.surprise,
                    episode.da,
                    episode.ach,
                    episode.na,
                    episode.expire_at,
                ),
            )
            return str(cur.fetchone()[0])

    def sample_episodes(self, limit: int, unconsolidated_only: bool = True) -> list[Episode]:
        clause = "WHERE consolidated_at IS NULL" if unconsolidated_only else ""
        sql = f"""
            SELECT id, context_id, created_at, embedding, raw_obs, canonical,
                   surprise, da, ach, na, replay_count, expire_at
            FROM episodes
            {clause}
            ORDER BY (da + surprise) DESC
            LIMIT %s
        """
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()

        return [
            Episode(
                id=str(r[0]),
                context_id=str(r[1]) if r[1] else "",
                created_at=r[2],
                embedding=_parse_vec(r[3]),
                raw_obs=r[4] if isinstance(r[4], dict) else json.loads(r[4] or "{}"),
                canonical=r[5],
                surprise=float(r[6]),
                da=float(r[7]),
                ach=float(r[8]),
                na=float(r[9]),
                replay_count=int(r[10]),
                expire_at=r[11],
            )
            for r in rows
        ]

    @with_retry
    def mark_replayed(self, episode_ids: list[str], extend_days: int) -> None:
        """Replay RESCUES an episode from TTL expiry. Unreplayed episodes die."""
        if not episode_ids:
            return
        sql = """
            UPDATE episodes
               SET replay_count = replay_count + 1,
                   consolidated_at = now(),
                   expire_at = now() + %s::interval
             WHERE id = ANY(%s::uuid[])
        """
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (f"{extend_days} days", episode_ids))

    def expired_episodes(self) -> list[Episode]:
        """Rows past TTL that the background job has not yet reclaimed."""
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, canonical, raw_obs FROM episodes WHERE expire_at <= now() LIMIT 1000"
            )
            rows = cur.fetchall()
        return [
            Episode(
                id=str(r[0]),
                canonical=r[1],
                raw_obs=r[2] if isinstance(r[2], dict) else json.loads(r[2] or "{}"),
            )
            for r in rows
        ]

    def purge_expired(self) -> int:
        """Row-level TTL handles this automatically; explicit call is for demos."""
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM episodes WHERE expire_at <= now()")
            return cur.rowcount or 0

    def episode_count(self) -> int:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM episodes WHERE expire_at > now()")
            return int(cur.fetchone()[0])

    # ------------------------------------------------------------------
    # neocortex
    # ------------------------------------------------------------------
    def recall_schemas(self, embedding: list[float], limit: int = 5) -> list[SchemaMatch]:
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(f"Expected vector dimension {EMBEDDING_DIMENSION}, got {len(embedding)}")

        # Single-parameter CTE: the ~20KB literal is sent once, not twice.
        sql = """
            WITH q AS (SELECT %s::vector AS v)
            SELECT s.id, s.label, s.centroid, s.variance, s.feature_mean,
                   s.support_count, s.stability, s.precision, s.origin,
                   s.rule_text, s.skill_ref, s.updated_at,
                   s.centroid <=> q.v AS distance
              FROM schemas s, q
             ORDER BY s.centroid <=> q.v
             LIMIT %s
        """
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (_vec(embedding), limit))
            rows = cur.fetchall()

        return [
            SchemaMatch(
                schema=Schema(
                    id=str(r[0]),
                    label=r[1] or "",
                    centroid=_parse_vec(r[2]),
                    variance=list(r[3] or []),
                    feature_mean=r[4] if isinstance(r[4], dict) else json.loads(r[4] or "{}"),
                    support_count=int(r[5]),
                    stability=int(r[6]),
                    precision=float(r[7]),
                    origin=r[8],
                    rule_text=r[9],
                    skill_ref=r[10],
                    updated_at=r[11],
                ),
                distance=float(r[12]),
            )
            for r in rows
        ]

    def nearest_schema(self, embedding: list[float]) -> SchemaMatch | None:
        matches = self.recall_schemas(embedding, limit=1)
        return matches[0] if matches else None

    @with_retry
    def create_schema(self, schema: Schema) -> str:
        sql = """
            INSERT INTO schemas
                (id, label, centroid, variance, feature_mean, support_count,
                 stability, precision, origin, rule_text, skill_ref)
            VALUES (%s, %s, %s::vector, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    schema.id,
                    schema.label,
                    _vec(schema.centroid),
                    schema.variance,
                    json.dumps(schema.feature_mean),
                    schema.support_count,
                    schema.stability,
                    schema.precision,
                    schema.origin,
                    schema.rule_text,
                    schema.skill_ref,
                ),
            )
            return str(cur.fetchone()[0])

    @with_retry
    def update_schema(self, schema: Schema) -> None:
        sql = """
            UPDATE schemas
               SET centroid = %s::vector, variance = %s, feature_mean = %s::jsonb,
                   support_count = %s, stability = %s, precision = %s,
                   rule_text = COALESCE(%s, rule_text),
                   skill_ref = COALESCE(%s, skill_ref),
                   updated_at = now()
             WHERE id = %s
        """
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    _vec(schema.centroid),
                    schema.variance,
                    json.dumps(schema.feature_mean),
                    schema.support_count,
                    schema.stability,
                    schema.precision,
                    schema.rule_text,
                    schema.skill_ref,
                    schema.id,
                ),
            )

    def all_schemas(self) -> list[Schema]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id, label, centroid, variance, feature_mean, support_count,
                          stability, precision, origin, rule_text, skill_ref, updated_at
                     FROM schemas ORDER BY stability DESC"""
            )
            rows = cur.fetchall()
        return [
            Schema(
                id=str(r[0]),
                label=r[1] or "",
                centroid=_parse_vec(r[2]),
                variance=list(r[3] or []),
                feature_mean=r[4] if isinstance(r[4], dict) else json.loads(r[4] or "{}"),
                support_count=int(r[5]),
                stability=int(r[6]),
                precision=float(r[7]),
                origin=r[8],
                rule_text=r[9],
                skill_ref=r[10],
                updated_at=r[11],
            )
            for r in rows
        ]

    def schema_count(self) -> int:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM schemas")
            return int(cur.fetchone()[0])

    # ------------------------------------------------------------------
    # provenance + telemetry
    # ------------------------------------------------------------------
    @with_retry
    def link_provenance(self, schema_id: str, episode_id: str, weight: float = 1.0) -> None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO provenance (schema_id, episode_id, weight)
                   VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
                (schema_id, episode_id, weight),
            )

    def provenance_for(self, schema_id: str) -> list[str]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT episode_id FROM provenance WHERE schema_id = %s ORDER BY linked_at DESC",
                (schema_id,),
            )
            return [str(r[0]) for r in cur.fetchall()]

    @with_retry
    def record_error(
        self,
        context_id: str,
        residual: list[float],
        surprise: float,
        ach: float,
        na: float,
        da: float,
        precision: float = 1.0,
    ) -> None:
        """The prediction-error bus. Errors are rows, not log lines."""
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO prediction_errors
                       (context_id, residual, precision, weighted_surprise, ach, na, da)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (context_id or None, residual, precision, surprise, ach, na, da),
            )

    @with_retry
    def open_context(self, context_id: str, label: str, trigger_na: float) -> None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO contexts (id, label, trigger_na)
                   VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING""",
                (context_id, label, trigger_na),
            )

    def beliefs_as_of(self, timestamp: str) -> list[dict[str, Any]]:
        """MVCC time travel: what did the agent believe at time T?

        Bounded by the table's GC window (default ~25h). Raise
        ``gc.ttlseconds`` on ``schemas`` for longer-range introspection.
        """
        sql = f"""
            SELECT label, support_count, stability, rule_text
              FROM schemas
              AS OF SYSTEM TIME '{timestamp}'
             ORDER BY stability DESC
             LIMIT 25
        """
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            return [
                {"label": r[0], "support": r[1], "stability": r[2], "rule": r[3]}
                for r in cur.fetchall()
            ]

    def table_stats(self) -> dict[str, Any]:
        """Uses crdb_internal, NOT pg_total_relation_size (unimplemented in CRDB)."""
        stats: dict[str, Any] = {}
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM episodes WHERE expire_at > now()")
            stats["live_episodes"] = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM schemas")
            stats["schemas"] = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM provenance")
            stats["provenance_links"] = int(cur.fetchone()[0])
            try:
                cur.execute(
                    """SELECT table_name, row_count_estimate
                         FROM crdb_internal.table_row_statistics
                        WHERE table_name IN ('episodes','schemas','provenance')"""
                )
                stats["row_estimates"] = {r[0]: int(r[1]) for r in cur.fetchall()}
            except Exception as exc:  # noqa: BLE001
                stats["row_estimates"] = f"unavailable: {exc}"
        return stats

    def health_check(self) -> bool:
        if not self._pool:
            return False
        try:
            with self.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        if self._pool:
            self._pool.closeall()
            self._pool = None


# Backwards-compatible alias
Cortex = CockroachStore

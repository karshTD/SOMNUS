"""CockroachDB schema. Real CRDB syntax -- not Postgres+pgvector.

Three corrections over the naive version:

  * NO ``CREATE EXTENSION vector``. CockroachDB has a *native* VECTOR type;
    there is no extension to install and the statement errors out.
  * NO ``USING hnsw (...)``. CockroachDB deliberately rejected HNSW (it builds
    its graph in memory and resists sharding) in favour of C-SPANN/SPFresh.
    The correct form is an inline ``VECTOR INDEX`` or ``CREATE VECTOR INDEX``.
  * Row-level TTL on ``episodes``. Forgetting is a property of the substrate,
    not application code. Episodes expire unless replay rescues them.

Statements are applied one at a time so an unsupported optional feature
(e.g. vector indexing on an older cluster) degrades to a warning rather than
aborting the whole migration -- exact ``<=>`` search still works without an index.
"""

from __future__ import annotations

import logging
import os
import sys

from infra.config import CONFIG, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

REQUIRED: list[tuple[str, str]] = [
    (
        "contexts",
        """
        CREATE TABLE IF NOT EXISTS contexts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            label STRING,
            opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            closed_at TIMESTAMPTZ,
            trigger_na FLOAT
        )
        """,
    ),
    (
        "episodes",
        f"""
        CREATE TABLE IF NOT EXISTS episodes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            context_id UUID,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            embedding VECTOR({EMBEDDING_DIMENSION}) NOT NULL,
            raw_obs JSONB NOT NULL,
            canonical STRING NOT NULL,
            surprise FLOAT NOT NULL DEFAULT 0,
            da FLOAT NOT NULL DEFAULT 0,
            ach FLOAT NOT NULL DEFAULT 0,
            na FLOAT NOT NULL DEFAULT 0,
            replay_count INT NOT NULL DEFAULT 0,
            consolidated_at TIMESTAMPTZ,
            expire_at TIMESTAMPTZ NOT NULL,
            INDEX episodes_priority_idx (consolidated_at, surprise DESC)
        ) WITH (
            ttl_expiration_expression = 'expire_at',
            ttl_job_cron = '*/15 * * * *'
        )
        """,
    ),
    (
        "schemas",
        f"""
        CREATE TABLE IF NOT EXISTS schemas (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            label STRING,
            centroid VECTOR({EMBEDDING_DIMENSION}) NOT NULL,
            variance FLOAT[] NOT NULL DEFAULT ARRAY[],
            feature_mean JSONB NOT NULL DEFAULT '{{}}',
            support_count INT NOT NULL DEFAULT 0,
            stability INT NOT NULL DEFAULT 0,
            precision FLOAT NOT NULL DEFAULT 1.0,
            origin STRING NOT NULL DEFAULT 'nrem',
            rule_text STRING,
            skill_ref STRING,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
    ),
    (
        "provenance",
        """
        CREATE TABLE IF NOT EXISTS provenance (
            schema_id UUID NOT NULL REFERENCES schemas(id) ON DELETE CASCADE,
            episode_id UUID NOT NULL,
            weight FLOAT NOT NULL DEFAULT 1.0,
            linked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (schema_id, episode_id)
        )
        """,
    ),
    (
        "prediction_errors",
        """
        CREATE TABLE IF NOT EXISTS prediction_errors (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            context_id UUID,
            level INT NOT NULL DEFAULT 1,
            residual FLOAT[] NOT NULL,
            precision FLOAT NOT NULL DEFAULT 1.0,
            weighted_surprise FLOAT NOT NULL,
            ach FLOAT NOT NULL DEFAULT 0,
            na FLOAT NOT NULL DEFAULT 0,
            da FLOAT NOT NULL DEFAULT 0,
            INDEX prediction_errors_time_idx (created_at DESC)
        )
        """,
    ),
    (
        "plasticity_params",
        """
        CREATE TABLE IF NOT EXISTS plasticity_params (
            key STRING PRIMARY KEY,
            value FLOAT NOT NULL,
            updated_by STRING,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
    ),
]

# Optional: vector indexes. Exact <=> search works without them; they only
# matter once the store grows past a few thousand rows.
OPTIONAL: list[tuple[str, str]] = [
    ("episodes vector index", "CREATE VECTOR INDEX IF NOT EXISTS episodes_embedding_idx ON episodes (embedding)"),
    ("schemas vector index", "CREATE VECTOR INDEX IF NOT EXISTS schemas_centroid_idx ON schemas (centroid)"),
]

SEED_PARAMS = """
UPSERT INTO plasticity_params (key, value, updated_by) VALUES
    ('alpha_base', {alpha}, 'migration'),
    ('na_threshold', {na}, 'migration'),
    ('novelty_threshold', {nov}, 'migration'),
    ('harden_at', {harden}, 'migration')
"""


def migrate(db_url: str | None = None) -> dict[str, list[str]]:
    """Apply the schema. Returns {'applied': [...], 'skipped': [...]}."""
    import psycopg2

    url = db_url or CONFIG.db_url or os.getenv("COCKROACH_DB_URL", "")
    if not url:
        raise ValueError("COCKROACH_DB_URL is required for migration")

    applied: list[str] = []
    skipped: list[str] = []

    conn = psycopg2.connect(url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for name, ddl in REQUIRED:
                cur.execute(ddl)
                applied.append(name)
                print(f"  [ok] {name}")

            for name, ddl in OPTIONAL:
                try:
                    cur.execute(ddl)
                    applied.append(name)
                    print(f"  [ok] {name}")
                except Exception as exc:  # noqa: BLE001
                    skipped.append(f"{name}: {exc}")
                    print(f"  [skip] {name} -- {exc}")
                    print("         Exact <=> search still works; index is a scale optimisation.")

            cur.execute(
                SEED_PARAMS.format(
                    alpha=CONFIG.plasticity.alpha_base,
                    na=CONFIG.neuromod.na_threshold,
                    nov=CONFIG.sleep.novelty_threshold,
                    harden=CONFIG.plasticity.harden_at,
                )
            )
            applied.append("plasticity seed")
    finally:
        conn.close()

    print(f"\nMigration complete: {len(applied)} applied, {len(skipped)} skipped.")
    return {"applied": applied, "skipped": skipped}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate(sys.argv[1] if len(sys.argv) > 1 else None)

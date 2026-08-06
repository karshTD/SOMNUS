"""In-process MemoryStore. Used by the forgetting benchmark and unit tests.

Implements the identical semantics as the CockroachDB store, including TTL
expiry, so benchmark results transfer directly to the live system.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from infra.embeddings import cosine_distance
from memory.store import Episode, Schema, SchemaMatch


class InMemoryStore:
    def __init__(self) -> None:
        self._episodes: dict[str, Episode] = {}
        self._schemas: dict[str, Schema] = {}
        self._provenance: list[tuple[str, str, float]] = []
        self._consolidated: set[str] = set()

    # --- hippocampus -----------------------------------------------------
    def write_episode(self, episode: Episode) -> str:
        self._episodes[episode.id] = episode
        return episode.id

    def sample_episodes(self, limit: int, unconsolidated_only: bool = True) -> list[Episode]:
        now = datetime.now(timezone.utc)
        pool = [
            e
            for e in self._episodes.values()
            if e.expire_at > now and (not unconsolidated_only or e.id not in self._consolidated)
        ]
        pool.sort(key=lambda e: (e.da + e.surprise), reverse=True)
        return pool[:limit]

    def mark_replayed(self, episode_ids: list[str], extend_days: int) -> None:
        for eid in episode_ids:
            ep = self._episodes.get(eid)
            if ep is None:
                continue
            ep.replay_count += 1
            ep.expire_at = datetime.now(timezone.utc) + timedelta(days=extend_days)
            self._consolidated.add(eid)

    def expired_episodes(self) -> list[Episode]:
        now = datetime.now(timezone.utc)
        return [e for e in self._episodes.values() if e.expire_at <= now]

    def purge_expired(self) -> int:
        expired = self.expired_episodes()
        for ep in expired:
            self._episodes.pop(ep.id, None)
        return len(expired)

    def episode_count(self) -> int:
        now = datetime.now(timezone.utc)
        return sum(1 for e in self._episodes.values() if e.expire_at > now)

    # --- neocortex -------------------------------------------------------
    def recall_schemas(self, embedding: list[float], limit: int = 5) -> list[SchemaMatch]:
        matches = [
            SchemaMatch(schema=s, distance=cosine_distance(embedding, s.centroid))
            for s in self._schemas.values()
        ]
        matches.sort(key=lambda m: m.distance)
        return matches[:limit]

    def nearest_schema(self, embedding: list[float]) -> SchemaMatch | None:
        matches = self.recall_schemas(embedding, limit=1)
        return matches[0] if matches else None

    def create_schema(self, schema: Schema) -> str:
        self._schemas[schema.id] = schema
        return schema.id

    def update_schema(self, schema: Schema) -> None:
        self._schemas[schema.id] = schema

    def all_schemas(self) -> list[Schema]:
        return list(self._schemas.values())

    def schema_count(self) -> int:
        return len(self._schemas)

    # --- provenance ------------------------------------------------------
    def link_provenance(self, schema_id: str, episode_id: str, weight: float = 1.0) -> None:
        self._provenance.append((schema_id, episode_id, weight))

    def provenance_for(self, schema_id: str) -> list[str]:
        return [eid for sid, eid, _ in self._provenance if sid == schema_id]

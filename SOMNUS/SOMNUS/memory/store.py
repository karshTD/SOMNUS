"""Memory store protocol and record types.

Two implementations satisfy this protocol:
  * memory.cortex.CockroachStore  -- production, CockroachDB
  * memory.inmemory.InMemoryStore -- benchmark and unit tests, zero deps

Keeping the interface explicit is what lets the forgetting benchmark run
offline with no credentials while the exact same consolidation code path
runs against CockroachDB in the demo.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Episode:
    """A hippocampal trace: one-shot, high resolution, disposable by default."""

    id: str = field(default_factory=new_id)
    context_id: str = ""
    created_at: datetime = field(default_factory=_now)
    embedding: list[float] = field(default_factory=list)
    raw_obs: dict[str, Any] = field(default_factory=dict)
    canonical: str = ""
    surprise: float = 0.0
    da: float = 0.0
    ach: float = 0.0
    na: float = 0.0
    replay_count: int = 0
    expire_at: datetime = field(default_factory=lambda: _now() + timedelta(hours=24))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "context_id": self.context_id,
            "created_at": self.created_at.isoformat(),
            "canonical": self.canonical,
            "raw_obs": self.raw_obs,
            "surprise": round(self.surprise, 4),
            "da": round(self.da, 4),
            "ach": round(self.ach, 4),
            "na": round(self.na, 4),
            "replay_count": self.replay_count,
            "expire_at": self.expire_at.isoformat(),
        }


@dataclass
class Schema:
    """A cortical prototype: statistical, slowly updated, metaplastically hardened."""

    id: str = field(default_factory=new_id)
    label: str = ""
    centroid: list[float] = field(default_factory=list)
    variance: list[float] = field(default_factory=list)
    feature_mean: dict[str, float] = field(default_factory=dict)
    support_count: int = 0
    stability: int = 0
    precision: float = 1.0
    origin: str = "nrem"
    rule_text: str | None = None
    skill_ref: str | None = None
    updated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "support_count": self.support_count,
            "stability": self.stability,
            "precision": round(self.precision, 4),
            "origin": self.origin,
            "rule_text": self.rule_text,
            "skill_ref": self.skill_ref,
            "feature_mean": {k: round(v, 3) for k, v in self.feature_mean.items()},
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class SchemaMatch:
    schema: Schema
    distance: float

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


@runtime_checkable
class MemoryStore(Protocol):
    """The substrate contract shared by CockroachDB and in-memory backends."""

    # --- hippocampus -----------------------------------------------------
    def write_episode(self, episode: Episode) -> str: ...
    def sample_episodes(self, limit: int, unconsolidated_only: bool = True) -> list[Episode]: ...
    def mark_replayed(self, episode_ids: list[str], extend_days: int) -> None: ...
    def expired_episodes(self) -> list[Episode]: ...
    def purge_expired(self) -> int: ...
    def episode_count(self) -> int: ...

    # --- neocortex -------------------------------------------------------
    def nearest_schema(self, embedding: list[float]) -> SchemaMatch | None: ...
    def recall_schemas(self, embedding: list[float], limit: int = 5) -> list[SchemaMatch]: ...
    def create_schema(self, schema: Schema) -> str: ...
    def update_schema(self, schema: Schema) -> None: ...
    def all_schemas(self) -> list[Schema]: ...
    def schema_count(self) -> int: ...

    # --- provenance ------------------------------------------------------
    def link_provenance(self, schema_id: str, episode_id: str, weight: float = 1.0) -> None: ...
    def provenance_for(self, schema_id: str) -> list[str]: ...

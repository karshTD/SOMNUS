"""Hippocampus: fast, one-shot, disposable episodic writes.

Corrected from the original design, which put episodes in S3. S3 cannot do ANN
search, so pattern completion -- retrieving a full episode from a partial cue,
the defining function of the hippocampus -- was impossible; it was a write-only
bucket. Episodes now live in CockroachDB with a vector index and row-level TTL.

S3 keeps a real job: cold archive for episodes that expire without being
consolidated, so nothing is irrecoverably lost even though the fast store
forgets by default.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from infra.config import CONFIG
from memory.separation import PatternSeparator
from memory.store import Episode, MemoryStore

logger = logging.getLogger(__name__)


class Hippocampus:
    def __init__(
        self,
        store: MemoryStore,
        aws: Any | None = None,
        separator: PatternSeparator | None = None,
        ttl_hours: int | None = None,
    ) -> None:
        self.store = store
        self.aws = aws
        self.separator = separator
        self.ttl_hours = ttl_hours or CONFIG.episode_ttl_hours

    def encode(
        self,
        observation: dict[str, Any],
        embedding: list[float],
        canonical: str,
        context_id: str,
        surprise: float,
        da: float,
        ach: float,
        na: float,
    ) -> str:
        """One-shot write. TTL is DA-scaled: salient episodes live longer."""
        vector = self.separator.separate(embedding) if self.separator else embedding
        hours = self.ttl_hours * (1.0 + min(da, 1.0))
        episode = Episode(
            context_id=context_id,
            embedding=vector,
            raw_obs=observation,
            canonical=canonical,
            surprise=surprise,
            da=da,
            ach=ach,
            na=na,
            expire_at=datetime.now(timezone.utc) + timedelta(hours=hours),
        )
        return self.store.write_episode(episode)

    def archive_expiring(self, prefix: str = "archive/") -> int:
        """Cold-store episodes about to expire. Optional; requires S3."""
        if self.aws is None or not getattr(self.aws, "s3_bucket", ""):
            return 0
        expiring = self.store.expired_episodes()
        if not expiring:
            return 0
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        for episode in expiring:
            try:
                self.aws.write_json(f"{prefix}{day}/{episode.id}.json", episode.to_dict())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Archive failed for %s: %s", episode.id, exc)
        return len(expiring)

"""Fast episodic memory writer (S3 Hippocampus)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from infra.aws_client import AWSClient


class Hippocampus:
    """Serialize anomaly episodes to JSON and push to S3 immediately."""

    def __init__(self, aws_client: AWSClient | None = None) -> None:
        self.aws = aws_client or AWSClient()

    def write_episode(self, event_data: dict[str, Any]) -> str:
        """Write a raw episodic memory object to the hippocampus bucket."""
        episode_id = event_data.get("episode_id") or str(uuid.uuid4())
        key = f"episodes/{datetime.now(timezone.utc).strftime('%Y%m%d')}/{episode_id}.json"
        payload = {
            "episode_id": episode_id,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            **event_data,
        }
        self.aws.write_json(key, payload)
        return key

    def list_recent_keys(self, limit: int = 20) -> list[str]:
        """Return the most recent episode keys."""
        keys = self.aws.list_episodes()
        return sorted(keys, reverse=True)[:limit]

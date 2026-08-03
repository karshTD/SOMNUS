"""REM sleep: consolidate S3 episodes into CockroachDB semantic rules."""

from __future__ import annotations

import json
import logging
from typing import Any

from infra.aws_client import AWSClient
from memory.cortex import Cortex

logger = logging.getLogger(__name__)

CONSOLIDATION_PROMPT = """Extract a generalized system rule from these anomaly episodes.
Return ONE concise operational rule (1-3 sentences) that would help prevent or remediate
similar incidents in the future.

Episodes:
{episodes}
"""


def consolidate_episodes(
    aws: AWSClient | None = None,
    cortex: Cortex | None = None,
) -> dict[str, Any]:
    """
    Fetch raw JSON from S3, summarize via Bedrock, embed, insert into cortex, delete S3 objects.
    """
    aws = aws or AWSClient()
    cortex = cortex or Cortex()

    keys = aws.list_episodes()
    if not keys:
        return {"consolidated": 0, "message": "No episodes to consolidate"}

    episodes: list[dict[str, Any]] = []
    for key in keys:
        try:
            episodes.append(aws.read_json(key))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", key, exc)

    if not episodes:
        return {"consolidated": 0, "message": "No readable episodes"}

    prompt = CONSOLIDATION_PROMPT.format(episodes=json.dumps(episodes, indent=2))
    rule_text = aws.reason(prompt)
    embedding = aws.embed_text(rule_text)
    source_ids = [ep.get("episode_id", "unknown") for ep in episodes]

    rule_id = cortex.insert_rule(rule_text, embedding, source_episodes=source_ids)

    deleted = 0
    for key in keys:
        try:
            aws.delete_object(key)
            deleted += 1
        except Exception as exc:
            logger.warning("Failed to delete %s: %s", key, exc)

    return {
        "consolidated": 1,
        "rule_id": rule_id,
        "rule_text": rule_text,
        "episodes_processed": len(episodes),
        "s3_objects_deleted": deleted,
    }


def handler(event: dict[str, Any] | None = None, context: Any = None) -> dict[str, Any]:
    """AWS Lambda entry point for the sleep cycle."""
    try:
        result = consolidate_episodes()
        return {"statusCode": 200, "body": json.dumps(result)}
    except Exception as exc:
        logger.exception("Sleep cycle failed")
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}

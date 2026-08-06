"""AWS Lambda entry point for the sleep cycle.

Thin wrapper. All logic lives in ``sleep_cycle.consolidation`` so the same code
path runs in Lambda, in-process, and in the offline benchmark.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

RULE_PROMPT = """These are anomaly episodes a monitoring agent consolidated into one schema.
State ONE concise operational rule (max 2 sentences) that would help remediate
similar incidents. Reply with the rule only.

Episodes:
{episodes}
"""


def _bedrock_rule_writer(aws: Any) -> Any:
    def write(episodes: list[Any]) -> str:
        compact = [
            {
                "state": e.canonical,
                "cpu": round(float(e.raw_obs.get("cpu_percent", 0)), 1),
                "rps": round(float(e.raw_obs.get("rps", 0)), 1),
                "net": round(float(e.raw_obs.get("network_mbps", 0)), 1),
            }
            for e in episodes
        ]
        return aws.reason(RULE_PROMPT.format(episodes=json.dumps(compact)), max_tokens=200).strip()

    return write


def run_sleep_cycle(store: Any = None, aws: Any = None) -> dict[str, Any]:
    from infra.aws_client import AWSClient
    from infra.config import CONFIG
    from memory.cortex import CockroachStore
    from memory.hippocampus import Hippocampus
    from sleep_cycle.consolidation import consolidate
    from sleep_cycle.skills import emit_skill

    store = store or CockroachStore()
    aws = aws or AWSClient()

    archived = 0
    try:
        archived = Hippocampus(store=store, aws=aws).archive_expiring()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Archive step skipped: %s", exc)

    report = consolidate(
        store,
        rule_writer=_bedrock_rule_writer(aws) if CONFIG.sleep.generate_rules else None,
        skill_emitter=emit_skill,
    )
    payload = report.to_dict()
    payload["archived_to_s3"] = archived
    return payload


def handler(event: dict[str, Any] | None = None, context: Any = None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    try:
        return {"statusCode": 200, "body": json.dumps(run_sleep_cycle())}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sleep cycle failed")
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}


# Backwards-compatible alias
consolidate_episodes = run_sleep_cycle

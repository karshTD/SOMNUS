"""Executes simulated remediation actions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from core.simulator import Simulator

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    action: str
    detail: str
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "detail": self.detail,
            "success": self.success,
        }


REMEDIATION_SCRIPTS: dict[str, Callable[[dict[str, Any]], str]] = {
    "block_ip": lambda ctx: f"Blocking IP {ctx.get('source_ip', 'unknown')}",
    "scale_service": lambda ctx: f"Scaling service {ctx.get('service', 'api')} +2 replicas",
    "rate_limit": lambda ctx: f"Applying rate limit {ctx.get('limit_rps', 100)} RPS",
    "flush_cache": lambda ctx: f"Flushing cache namespace {ctx.get('namespace', 'default')}",
}


def execute_decision(
    decision: dict[str, Any],
    simulator: Simulator | None = None,
) -> ActionResult:
    """Run a remediation action and reset simulator anomaly flags."""
    action_name = decision.get("action", "log_only")
    context = decision.get("context", {})

    if action_name in REMEDIATION_SCRIPTS:
        detail = REMEDIATION_SCRIPTS[action_name](context)
    else:
        detail = f"No-op remediation for decision: {decision}"

    logger.info("REMEDIATION: %s — %s", action_name, detail)

    if simulator is not None:
        simulator.reset_anomaly()

    return ActionResult(action=action_name, detail=detail, success=True)

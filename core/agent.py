"""Active Inference wake loop for the SOMNUS agent."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core.actions import execute_decision
from core.neuromod import DEFAULT_SURPRISE_THRESHOLD, compute_prediction_error
from core.simulator import Simulator
from infra.aws_client import AWSClient
from infra.ccloud import CCloudManager
from memory.cortex import Cortex
from memory.hippocampus import Hippocampus

logger = logging.getLogger(__name__)


@dataclass
class WakeState:
    cycle: int = 0
    last_error: float = 0.0
    last_surprised: bool = False
    recent_episodes: list[str] = field(default_factory=list)
    last_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "last_error": round(self.last_error, 4),
            "last_surprised": self.last_surprised,
            "recent_episodes": self.recent_episodes[-10:],
            "last_decision": self.last_decision,
        }


class SomnusAgent:
    """Generate predictions, detect surprise, recall cortex rules, act, and store episodes."""

    def __init__(
        self,
        poll_interval: float = 2.0,
        surprise_threshold: float = DEFAULT_SURPRISE_THRESHOLD,
        on_state_change: Callable[[WakeState], None] | None = None,
    ) -> None:
        self.poll_interval = poll_interval
        self.surprise_threshold = surprise_threshold
        self.on_state_change = on_state_change

        self.simulator = Simulator()
        self.aws = AWSClient()
        self.hippocampus = Hippocampus(self.aws)
        self.cortex = Cortex()
        self.ccloud = CCloudManager()
        self.state = WakeState()
        self._running = False

    def _predict_telemetry(self) -> dict[str, float]:
        """Generate expected next telemetry from baseline (prior beliefs)."""
        return {
            "cpu_percent": self.simulator.base_cpu,
            "rps": self.simulator.base_rps,
            "network_mbps": self.simulator.base_network,
        }

    def _decide_action(
        self,
        actual: dict[str, Any],
        rules: list[Any],
        error_score: float,
    ) -> dict[str, Any]:
        """Choose remediation based on cortex recall and telemetry."""
        if rules:
            top = rules[0]
            rule_lower = top.rule_text.lower()
            if "ip" in rule_lower or "block" in rule_lower:
                action = "block_ip"
            elif "scale" in rule_lower or "replica" in rule_lower:
                action = "scale_service"
            elif "rate" in rule_lower or "limit" in rule_lower:
                action = "rate_limit"
            else:
                action = "flush_cache"
            return {
                "action": action,
                "context": {
                    "service": "api-gateway",
                    "source_ip": actual.get("source_ip", "10.0.0.99"),
                    "rule_id": top.id,
                    "similarity": top.similarity,
                },
                "reason": top.rule_text,
            }

        if actual.get("cpu_percent", 0) > 80:
            return {
                "action": "scale_service",
                "context": {"service": "api-gateway"},
                "reason": f"High CPU surprise (error={error_score:.2f})",
            }
        return {
            "action": "rate_limit",
            "context": {"limit_rps": 200},
            "reason": f"Traffic spike surprise (error={error_score:.2f})",
        }

    def wake_step(self) -> WakeState:
        """Execute one active inference cycle."""
        self.state.cycle += 1
        predicted = self._predict_telemetry()
        actual = self.simulator.emit()

        result = compute_prediction_error(
            predicted,
            actual,
            surprise_threshold=self.surprise_threshold,
        )
        self.state.last_error = result.error_score
        self.state.last_surprised = result.is_surprised

        if result.is_surprised:
            logger.warning(
                "Surprise detected (error=%.3f): %s",
                result.error_score,
                json.dumps(result.per_metric_errors),
            )

            rules: list[Any] = []
            try:
                event_text = json.dumps({"predicted": predicted, "actual": actual})
                query_vector = self.aws.embed_text(event_text)
                rules = self.cortex.recall_similar(query_vector, limit=3)
            except Exception as exc:
                logger.warning("Cortex recall skipped: %s", exc)

            decision = self._decide_action(actual, rules, result.error_score)
            self.state.last_decision = decision

            try:
                episode_key = self.hippocampus.write_episode(
                    {
                        "predicted": predicted,
                        "actual": actual,
                        "prediction_error": result.to_dict(),
                        "decision": decision,
                    }
                )
                self.state.recent_episodes.append(episode_key)
            except Exception as exc:
                logger.warning("Hippocampus write skipped: %s", exc)

            execute_decision(decision, simulator=self.simulator)

            try:
                density = self.cortex.row_count()
                self.ccloud.check_and_scale(density)
            except Exception as exc:
                logger.debug("Meta-plasticity check skipped: %s", exc)

        if self.on_state_change:
            self.on_state_change(self.state)

        return self.state

    def wake_loop(self) -> None:
        """Continuous wake loop until stopped."""
        self._running = True
        logger.info("SOMNUS wake loop started (interval=%ss)", self.poll_interval)
        while self._running:
            try:
                self.wake_step()
            except Exception:
                logger.exception("Wake step failed")
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False
        self.cortex.close()

    def get_state(self) -> dict[str, Any]:
        return self.state.to_dict()

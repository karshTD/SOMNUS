"""The wake loop.

Perception -> prediction error -> neuromodulation -> encoding -> learning ->
action. Every step writes to the substrate, so the agent's cognitive state is
queryable SQL rather than process memory.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core.actions import execute_decision
from core.control import drain_commands
from core.neuromod import NeuromodulatorySystem
from core.predictor import ContextualPredictor
from core.simulator import Simulator
from infra.config import CONFIG
from infra.embeddings import ObservationEncoder
from memory.hippocampus import Hippocampus
from memory.separation import PatternSeparator
from memory.store import MemoryStore
from sleep_cycle.consolidation import violation_penalty

logger = logging.getLogger(__name__)


@dataclass
class WakeState:
    cycle: int = 0
    surprise: float = 0.0
    ach: float = 0.0
    na: float = 0.0
    da: float = 0.0
    boundary: bool = False
    context: dict[str, Any] = field(default_factory=dict)
    last_boundary: dict[str, Any] | None = None
    last_decision: dict[str, Any] | None = None
    predicted: dict[str, float] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)
    episodes_written: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "surprise": round(self.surprise, 4),
            "ach": round(self.ach, 4),
            "na": round(self.na, 4),
            "da": round(self.da, 4),
            "boundary": self.boundary,
            "context": self.context,
            "last_boundary": self.last_boundary,
            "last_decision": self.last_decision,
            "predicted": {k: round(v, 2) for k, v in self.predicted.items()},
            "actual": self.actual,
            "episodes_written": self.episodes_written,
            "history": self.history[-200:],
        }


class SomnusAgent:
    def __init__(
        self,
        store: MemoryStore,
        simulator: Simulator | None = None,
        aws: Any | None = None,
        poll_interval: float | None = None,
        on_state_change: Callable[[WakeState], None] | None = None,
    ) -> None:
        self.store = store
        self.simulator = simulator or Simulator()
        self.poll_interval = poll_interval or CONFIG.poll_interval
        self.on_state_change = on_state_change

        self.encoder = ObservationEncoder()
        self.neuromod = NeuromodulatorySystem()
        self.predictor = ContextualPredictor(store=store, enable_recall=True)
        self.hippocampus = Hippocampus(
            store=store,
            aws=aws,
            separator=PatternSeparator(self.encoder.dimension),
        )
        self.state = WakeState()
        self._running = False
        self._rng = __import__("random").Random()

    # ------------------------------------------------------------------
    def _handle_commands(self) -> None:
        for cmd in drain_commands():
            name, args = cmd.get("command"), cmd.get("args", {})
            try:
                if name == "inject_anomaly":
                    self.simulator.trigger_anomaly(int(args.get("ticks", 6)))
                    logger.warning("COMMAND: anomaly injected for %s ticks", args.get("ticks", 6))
                elif name == "set_regime":
                    self.simulator.set_regime(str(args["regime"]))
                    logger.warning("COMMAND: regime -> %s", args["regime"])
                elif name == "drift":
                    self.simulator.set_drift(args["start"], args["end"], float(args["t"]))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Command %s failed: %s", name, exc)

    def _decide(self, observation: dict[str, Any], surprise: float) -> dict[str, Any] | None:
        if surprise < 0.35:
            return None
        try:
            _, embedding = self.encoder.encode(observation)
            matches = self.store.recall_schemas(embedding, limit=3)
        except Exception:  # noqa: BLE001
            matches = []

        if matches and matches[0].schema.rule_text:
            rule = matches[0].schema.rule_text.lower()
            action = (
                "block_ip" if "block" in rule or " ip" in rule
                else "scale_service" if "scale" in rule or "replica" in rule
                else "rate_limit" if "rate" in rule or "limit" in rule
                else "flush_cache"
            )
            return {
                "action": action,
                "context": {"service": "api-gateway", "schema_id": matches[0].schema.id},
                "reason": matches[0].schema.rule_text,
                "similarity": round(matches[0].similarity, 4),
            }

        if float(observation.get("cpu_percent", 0)) > 80:
            return {"action": "scale_service", "context": {"service": "api-gateway"}, "reason": "high CPU"}
        return {"action": "rate_limit", "context": {"limit_rps": 200}, "reason": "traffic spike"}

    def wake_step(self) -> WakeState:
        self._handle_commands()
        self.state.cycle += 1

        predicted = self.predictor.predict()
        observation = self.simulator.emit()
        observation.pop("_truth", None)  # ground truth is benchmark-only

        canonical, embedding = self.encoder.encode(observation)
        nm = self.neuromod.step(predicted, observation, embedding=embedding)

        # --- the error bus: errors are rows, not log lines
        try:
            residual = [
                float(observation.get(m, 0.0)) - float(predicted.get(m, 0.0))
                for m in ("cpu_percent", "rps", "network_mbps")
            ]
            if hasattr(self.store, "record_error"):
                self.store.record_error(
                    self.predictor.context_id, residual, nm.surprise, nm.ach, nm.na, nm.da
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error-bus write skipped: %s", exc)

        # --- noradrenaline: context boundary
        if nm.boundary:
            try:
                violation_penalty(self.store, embedding)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Violation penalty skipped: %s", exc)
            info = self.predictor.on_boundary(embedding, observation)
            self.neuromod.reset_context()
            self.state.last_boundary = info
            if hasattr(self.store, "open_context"):
                try:
                    self.store.open_context(info["context_id"], info["label"], nm.na)
                except Exception:  # noqa: BLE001
                    pass
            logger.warning("CONTEXT BOUNDARY (na=%.2f): %s", nm.na, info["outcome"])

        # --- dopamine/surprise-gated encoding
        if self._rng.random() < self.neuromod.encode_probability(nm):
            try:
                self.hippocampus.encode(
                    observation, embedding, canonical, self.predictor.context_id,
                    nm.surprise, nm.da, nm.ach, nm.na,
                )
                self.state.episodes_written += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hippocampal write failed: %s", exc)

        # --- acetylcholine-modulated learning
        self.predictor.update(observation, self.neuromod.learning_rate(nm, CONFIG.plasticity.alpha_base))

        decision = self._decide(observation, nm.surprise)
        if decision:
            execute_decision(decision, simulator=self.simulator)
            self.state.last_decision = decision

        self.state.surprise, self.state.ach = nm.surprise, nm.ach
        self.state.na, self.state.da = nm.na, nm.da
        self.state.boundary = nm.boundary
        self.state.predicted, self.state.actual = predicted, observation
        self.state.context = self.predictor.snapshot()
        self.state.history.append(
            {
                "cycle": self.state.cycle,
                "surprise": round(nm.surprise, 4),
                "ach": round(nm.ach, 4),
                "na": round(nm.na, 4),
                "boundary": nm.boundary,
                "cpu": observation.get("cpu_percent"),
                "regime": observation.get("regime"),
            }
        )

        if self.on_state_change:
            self.on_state_change(self.state)
        return self.state

    def wake_loop(self) -> None:
        self._running = True
        logger.info("SOMNUS wake loop started (interval=%ss)", self.poll_interval)
        while self._running:
            start = time.monotonic()
            try:
                self.wake_step()
            except Exception:
                logger.exception("Wake step failed")
            # drift correction: hold the cadence regardless of step duration
            time.sleep(max(0.0, self.poll_interval - (time.monotonic() - start)))

    def stop(self) -> None:
        self._running = False
        if hasattr(self.store, "close"):
            self.store.close()

    def get_state(self) -> dict[str, Any]:
        return self.state.to_dict()

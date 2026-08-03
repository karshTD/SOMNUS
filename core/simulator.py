"""Generates fake telemetry data for the active inference wake loop."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TelemetrySnapshot:
    timestamp: float
    cpu_percent: float
    rps: float
    network_mbps: float
    anomaly: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 2),
            "rps": round(self.rps, 2),
            "network_mbps": round(self.network_mbps, 2),
            "anomaly": self.anomaly,
        }


@dataclass
class Simulator:
    """Emits timestamped JSON telemetry with optional anomaly injection."""

    base_cpu: float = 35.0
    base_rps: float = 120.0
    base_network: float = 50.0
    _anomaly_active: bool = field(default=False, init=False)

    def emit(self) -> dict[str, Any]:
        """Return a telemetry snapshot as JSON-serializable dict."""
        cpu = self.base_cpu + random.uniform(-5, 5)
        rps = self.base_rps + random.uniform(-10, 10)
        network = self.base_network + random.uniform(-8, 8)

        if self._anomaly_active:
            cpu = random.uniform(85, 99)
            rps = random.uniform(800, 1200)
            network = random.uniform(400, 600)

        snapshot = TelemetrySnapshot(
            timestamp=time.time(),
            cpu_percent=cpu,
            rps=rps,
            network_mbps=network,
            anomaly=self._anomaly_active,
        )
        return snapshot.to_dict()

    def trigger_anomaly(self) -> None:
        """Force the next telemetry reads to reflect an anomaly."""
        self._anomaly_active = True

    def reset_anomaly(self) -> None:
        """Clear injected anomaly state."""
        self._anomaly_active = False

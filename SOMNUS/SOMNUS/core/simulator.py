"""Environment: telemetry regimes with injectable distribution shift.

Regimes are named so the benchmark can move the world between them and then
move it back -- which is the whole point of a forgetting benchmark.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Regime:
    name: str
    cpu: float
    rps: float
    network: float
    jitter: float = 0.35

    def sample(self, rng: random.Random) -> dict[str, float]:
        def n(mu: float) -> float:
            return max(0.0, mu * (1.0 + rng.gauss(0.0, self.jitter)))

        return {"cpu_percent": n(self.cpu), "rps": n(self.rps), "network_mbps": n(self.network)}


REGIMES: dict[str, Regime] = {
    # Task A -- ordinary daytime serving load.
    "steady": Regime("steady", cpu=35.0, rps=120.0, network=50.0),
    # Task B -- post-incident degraded/rebalancing regime.
    "surge": Regime("surge", cpu=78.0, rps=640.0, network=310.0),
    # A third world, for the generalisation check.
    "batch": Regime("batch", cpu=58.0, rps=45.0, network=180.0),
}


@dataclass
class Simulator:
    """Emits telemetry from the active regime; supports shift and anomaly."""

    regime_name: str = "steady"
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _anomaly_ticks: int = field(default=0, init=False, repr=False)
    _blend: "Regime | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # Compatibility shims for older callers -----------------------------
    @property
    def base_cpu(self) -> float:
        return self.regime.cpu

    @property
    def base_rps(self) -> float:
        return self.regime.rps

    @property
    def base_network(self) -> float:
        return self.regime.network

    @property
    def regime(self) -> Regime:
        if self._blend is not None:
            return self._blend
        return REGIMES[self.regime_name]

    def set_regime(self, name: str) -> None:
        if name not in REGIMES:
            raise ValueError(f"Unknown regime {name!r}. Options: {sorted(REGIMES)}")
        self.regime_name = name
        self._blend = None

    def set_drift(self, start: str, end: str, t: float) -> None:
        """Interpolate between two regimes. t in [0, 1].

        Gradual drift is the realistic failure mode -- systems degrade, they
        rarely teleport. It is also the condition under which schema drift
        occurs, because each successive observation stays inside the novelty
        threshold of the last, chain-linking the old prototype to the new
        regime. This is precisely what interleaved replay exists to prevent.
        """
        a, b = REGIMES[start], REGIMES[end]
        t = max(0.0, min(1.0, t))
        self._blend = Regime(
            name=f"{start}->{end}@{t:.2f}",
            cpu=a.cpu + t * (b.cpu - a.cpu),
            rps=a.rps + t * (b.rps - a.rps),
            network=a.network + t * (b.network - a.network),
        )
        self.regime_name = self._blend.name

    def trigger_anomaly(self, ticks: int = 6) -> None:
        """Transient spike, distinct from a sustained regime shift."""
        self._anomaly_ticks = ticks

    def reset_anomaly(self) -> None:
        self._anomaly_ticks = 0

    def emit(self) -> dict[str, Any]:
        if self._anomaly_ticks > 0:
            self._anomaly_ticks -= 1
            values = {
                "cpu_percent": self._rng.uniform(88, 99),
                "rps": self._rng.uniform(900, 1300),
                "network_mbps": self._rng.uniform(450, 620),
            }
            anomaly = True
        else:
            values = self.regime.sample(self._rng)
            anomaly = False

        truth = (
            {"cpu_percent": 93.5, "rps": 1100.0, "network_mbps": 535.0}
            if anomaly
            else {"cpu_percent": self.regime.cpu, "rps": self.regime.rps, "network_mbps": self.regime.network}
        )
        return {
            "timestamp": time.time(),
            "regime": self.regime_name,
            # Ground-truth regime means. Used ONLY by the benchmark to separate
            # model error from irreducible environment noise. The agent never
            # reads this.
            "_truth": truth,
            "cpu_percent": round(values["cpu_percent"], 2),
            "rps": round(values["rps"], 2),
            "network_mbps": round(values["network_mbps"], 2),
            "anomaly": anomaly,
        }

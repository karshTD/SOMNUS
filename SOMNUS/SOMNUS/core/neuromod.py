"""Neuromodulatory nuclei.

Three small, fast functions computing global scalar state from the error
stream. They are NOT hand-tuned knobs -- each is derived from the agent's own
prediction errors, which is what makes the system self-regulating.

  ACh -- expected uncertainty.   "How noisy is this world normally?"
                                 (Yu & Dayan 2005)
  NA  -- unexpected uncertainty. "Is this surprise beyond my noise model?"
                                 Sustained NA => context boundary. This is a
                                 Bayesian changepoint detector in neuroscience
                                 costume, which is exactly correct.
  DA  -- value / novelty.        Gates episode WRITE probability and replay
                                 priority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from infra.config import CONFIG, NeuromodConfig

METRIC_WEIGHTS = {"cpu_percent": 0.35, "rps": 0.35, "network_mbps": 0.30}


@dataclass
class NeuromodState:
    surprise: float = 0.0
    ach: float = 0.0
    na: float = 0.0
    da: float = 0.0
    boundary: bool = False
    per_metric: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surprise": round(self.surprise, 4),
            "ach": round(self.ach, 4),
            "na": round(self.na, 4),
            "da": round(self.da, 4),
            "boundary": self.boundary,
            "per_metric": {k: round(v, 4) for k, v in self.per_metric.items()},
        }


def normalized_delta(predicted: float, actual: float) -> float:
    if predicted == 0 and actual == 0:
        return 0.0
    denom = max(abs(predicted), abs(actual), 1.0)
    return min(abs(actual - predicted) / denom, 1.0)


def weighted_surprise(predicted: dict[str, float], actual: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Precision-weighted prediction error S(t) in [0, 1]."""
    per_metric: dict[str, float] = {}
    total = weight_total = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        delta = normalized_delta(float(predicted.get(metric, 0.0)), float(actual.get(metric, 0.0)))
        per_metric[metric] = delta
        total += delta * weight
        weight_total += weight
    return (total / weight_total if weight_total else 0.0), per_metric


class NeuromodulatorySystem:
    """Stateful nuclei. One instance per agent."""

    def __init__(self, config: NeuromodConfig | None = None) -> None:
        self.cfg = config or CONFIG.neuromod
        self._ach = 0.0
        self._ach_var = 0.0
        self._initialised = False
        self._cusum = 0.0
        self._observations = 0
        self._observations = 0
        self._seen_regions: list[list[float]] = []

    @property
    def ach(self) -> float:
        return max(self._ach, self.cfg.ach_floor)

    @property
    def ach_std(self) -> float:
        return max(self._ach_var**0.5, self.cfg.ach_floor)

    def reset_context(self) -> None:
        """A new context has its own noise model. Do not carry ACh across."""
        self._ach = 0.0
        self._ach_var = 0.0
        self._initialised = False
        self._cusum = 0.0
        self._observations = 0

    def _novelty(self, embedding: list[float] | None) -> float:
        """Distance to the nearest previously-seen embedding, in [0, 1]."""
        if not embedding:
            return 0.0
        from infra.embeddings import cosine_distance

        if not self._seen_regions:
            self._seen_regions.append(list(embedding))
            return 1.0
        nearest = min(cosine_distance(embedding, r) for r in self._seen_regions)
        if nearest > 0.3 and len(self._seen_regions) < 256:
            self._seen_regions.append(list(embedding))
        return min(nearest, 1.0)

    def step(
        self,
        predicted: dict[str, float],
        actual: dict[str, float],
        embedding: list[float] | None = None,
        reward: float = 0.0,
    ) -> NeuromodState:
        surprise, per_metric = weighted_surprise(predicted, actual)

        # --- ACh: expected uncertainty. Track BOTH the mean and the spread of
        # surprise inside this context -- the spread is the noise model.
        alpha = 2.0 / (self.cfg.ach_halflife + 1.0)
        if not self._initialised:
            self._ach = surprise
            self._ach_var = 0.0
            self._initialised = True
            na = 0.0
        else:
            deviation = surprise - self._ach
            self._ach = (1 - alpha) * self._ach + alpha * surprise
            self._ach_var = (1 - alpha) * self._ach_var + alpha * deviation**2
            # --- NA: unexpected uncertainty as a z-score, not a ratio.
            #
            # A ratio (S / ACh) collapses in a noisy world: raise the jitter and
            # the mean surprise rises with it, so a genuine regime change never
            # clears the threshold. The z-score asks the right question --
            # "is this surprise large RELATIVE TO HOW MUCH SURPRISE NORMALLY
            # VARIES here?" -- which is what a changepoint detector needs.
            na = deviation / self.ach_std

        # CUSUM changepoint detection.
        #
        # A single-tick threshold cannot work here and we measured why: in
        # steady state NA spikes to 3.8, while a genuine regime shift only
        # reaches 3.2 on its first tick. The distributions overlap, so ANY
        # instantaneous rule trades false positives against missed shifts.
        #
        # What separates them is PERSISTENCE: noise spikes are isolated, a real
        # shift keeps NA elevated for many consecutive ticks. CUSUM accumulates
        # evidence above a slack allowance and decays otherwise, which is
        # exactly the classical sequential changepoint test.
        # Warm-up. Before enough observations exist, ach_std is the floor and
        # NA is meaninglessly large -- a changepoint detector cannot fire before
        # it has a noise model to compare against.
        self._observations += 1
        if self._observations < self.cfg.warmup_ticks:
            self._cusum = 0.0
            na = 0.0

        self._cusum = max(0.0, self._cusum + (na - self.cfg.na_slack))
        boundary = self._cusum > self.cfg.na_threshold
        if boundary:
            self._cusum = 0.0

        # --- DA: value + novelty bonus
        da = reward + self.cfg.da_novelty_beta * self._novelty(embedding)

        return NeuromodState(
            surprise=surprise,
            ach=self.ach,
            na=na,
            da=da,
            boundary=boundary,
            per_metric=per_metric,
        )

    def encode_probability(self, state: NeuromodState) -> float:
        """P(write to hippocampus). High surprise or high DA => encode."""
        p = self.cfg.encode_floor + 0.6 * state.surprise + 0.4 * min(state.da, 1.0)
        return max(0.0, min(1.0, p))

    def learning_rate(self, state: NeuromodState, base: float) -> float:
        """ACh raises the rate: when the world is genuinely noisy, trust input."""
        gain = 1.0 + CONFIG.plasticity.ach_gain * min(state.ach, 1.0)
        return max(CONFIG.plasticity.alpha_min, min(1.0, base * gain))


# --- backwards-compatible functional API -------------------------------
def compute_prediction_error(
    predicted: dict[str, float],
    actual: dict[str, float],
    surprise_threshold: float = 0.35,
) -> NeuromodState:
    surprise, per_metric = weighted_surprise(predicted, actual)
    state = NeuromodState(surprise=surprise, per_metric=per_metric)
    state.boundary = surprise > surprise_threshold
    return state

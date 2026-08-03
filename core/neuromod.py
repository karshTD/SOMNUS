"""Prediction error calculator (dopamine / noradrenaline analog)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_SURPRISE_THRESHOLD = 0.35

METRIC_WEIGHTS = {
    "cpu_percent": 0.35,
    "rps": 0.35,
    "network_mbps": 0.30,
}


@dataclass
class PredictionResult:
    error_score: float
    is_surprised: bool
    per_metric_errors: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_score": round(self.error_score, 4),
            "is_surprised": self.is_surprised,
            "per_metric_errors": {
                k: round(v, 4) for k, v in self.per_metric_errors.items()
            },
        }


def _normalized_delta(predicted: float, actual: float) -> float:
    """Return a 0-1 error for a single metric."""
    if predicted == 0 and actual == 0:
        return 0.0
    denom = max(abs(predicted), abs(actual), 1.0)
    return min(abs(actual - predicted) / denom, 1.0)


def compute_prediction_error(
    predicted: dict[str, float],
    actual: dict[str, float],
    surprise_threshold: float = DEFAULT_SURPRISE_THRESHOLD,
) -> PredictionResult:
    """Compare predicted vs actual telemetry; return error score in [0, 1]."""
    per_metric: dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for metric, weight in METRIC_WEIGHTS.items():
        pred_val = float(predicted.get(metric, 0.0))
        act_val = float(actual.get(metric, 0.0))
        delta = _normalized_delta(pred_val, act_val)
        per_metric[metric] = delta
        weighted_sum += delta * weight
        weight_total += weight

    error_score = weighted_sum / weight_total if weight_total else 0.0
    if actual.get("anomaly"):
        error_score = min(1.0, error_score + 0.25)

    return PredictionResult(
        error_score=error_score,
        is_surprised=error_score > surprise_threshold,
        per_metric_errors=per_metric,
    )

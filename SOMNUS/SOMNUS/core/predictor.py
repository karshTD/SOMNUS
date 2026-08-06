"""The learning component. THIS is what the old build was missing.

The previous predictor returned a constant (``simulator.base_cpu``), so no
part of the system ever got better at prediction and every "continual
learning" claim was unsupported by a single line of code.

Here the predictor is genuinely stateful:

  * It holds one running estimate per CONTEXT, not one global estimate.
  * Its learning rate is modulated by ACh.
  * On a context boundary it queries the cortical store for a matching schema
    and, on a hit, RESTORES that schema's statistics as the prior for the new
    context. This is what prevents catastrophic forgetting: returning to an old
    regime recalls the old model instead of relearning it from scratch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from infra.config import CONFIG
from memory.store import MemoryStore, Schema, new_id

logger = logging.getLogger(__name__)

TRACKED = ("cpu_percent", "rps", "network_mbps")


@dataclass
class ContextModel:
    """Running per-metric estimate for one context."""

    context_id: str
    label: str = ""
    mean: dict[str, float] = field(default_factory=dict)
    n: int = 0
    restored_from: str | None = None

    def predict(self) -> dict[str, float]:
        return dict(self.mean)

    def update(self, observation: dict[str, Any], lr: float) -> None:
        for metric in TRACKED:
            value = float(observation.get(metric, 0.0))
            if metric not in self.mean:
                self.mean[metric] = value
            else:
                self.mean[metric] += lr * (value - self.mean[metric])
        self.n += 1


class ContextualPredictor:
    """Multi-context predictor with schema-backed prior restoration."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        prior: dict[str, float] | None = None,
        enable_recall: bool = True,
        recall_threshold: float = 0.35,
    ) -> None:
        self.store = store
        self.enable_recall = enable_recall
        self.recall_threshold = recall_threshold
        self.contexts: dict[str, ContextModel] = {}
        self.current = ContextModel(context_id=new_id(), label="ctx-0", mean=dict(prior or {}))
        self.contexts[self.current.context_id] = self.current
        self.boundary_count = 0
        self.recall_hits = 0
        self.recall_misses = 0

    @property
    def context_id(self) -> str:
        return self.current.context_id

    def predict(self) -> dict[str, float]:
        return self.current.predict()

    def update(self, observation: dict[str, Any], lr: float) -> None:
        self.current.update(observation, lr)

    def on_boundary(self, embedding: list[float] | None, observation: dict[str, Any]) -> dict[str, Any]:
        """Declare a new context. Try to restore a prior from cortical memory.

        Returns a small dict describing what happened, for the dashboard/demo.
        """
        self.boundary_count += 1
        restored: Schema | None = None
        distance = None

        if self.enable_recall and self.store is not None and embedding:
            try:
                match = self.store.nearest_schema(embedding)
                if match and match.distance <= self.recall_threshold and match.schema.feature_mean:
                    restored = match.schema
                    distance = match.distance
            except Exception as exc:  # noqa: BLE001
                logger.warning("Schema recall failed at boundary: %s", exc)

        model = ContextModel(
            context_id=new_id(),
            label=f"ctx-{self.boundary_count}",
        )

        if restored is not None:
            # RECALL: adopt the remembered regime as the prior. No relearning.
            model.mean = dict(restored.feature_mean)
            model.restored_from = restored.id
            model.n = restored.support_count
            self.recall_hits += 1
            outcome = "restored"
        else:
            # NOVEL: seed from the observation that triggered the boundary.
            model.mean = {m: float(observation.get(m, 0.0)) for m in TRACKED}
            self.recall_misses += 1
            outcome = "novel"

        self.contexts[model.context_id] = model
        self.current = model
        return {
            "outcome": outcome,
            "context_id": model.context_id,
            "label": model.label,
            "restored_from": model.restored_from,
            "distance": round(distance, 4) if distance is not None else None,
            "schema_label": restored.label if restored else None,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "context_id": self.current.context_id,
            "label": self.current.label,
            "mean": {k: round(v, 2) for k, v in self.current.mean.items()},
            "observations": self.current.n,
            "restored_from": self.current.restored_from,
            "boundaries": self.boundary_count,
            "recall_hits": self.recall_hits,
            "recall_misses": self.recall_misses,
        }


class NaiveePredictor:  # noqa: N801 - kept for the ablation registry
    pass


class NaivePredictor:
    """Control arm: one global estimate, fixed rate, no contexts, no recall.

    This is what a straightforward 'fine-tune on recent data' agent does, and
    it is the curve SOMNUS has to beat in the benchmark.
    """

    def __init__(self, prior: dict[str, float] | None = None, lr: float | None = None) -> None:
        self.mean: dict[str, float] = dict(prior or {})
        self.lr = lr if lr is not None else CONFIG.plasticity.alpha_base
        self.n = 0

    @property
    def context_id(self) -> str:
        return "global"

    def predict(self) -> dict[str, float]:
        return dict(self.mean)

    def update(self, observation: dict[str, Any], lr: float | None = None) -> None:
        rate = self.lr if lr is None else lr
        for metric in TRACKED:
            value = float(observation.get(metric, 0.0))
            self.mean[metric] = value if metric not in self.mean else self.mean[metric] + rate * (
                value - self.mean[metric]
            )
        self.n += 1

    def on_boundary(self, embedding: list[float] | None, observation: dict[str, Any]) -> dict[str, Any]:
        return {"outcome": "ignored", "context_id": "global"}

    def snapshot(self) -> dict[str, Any]:
        return {"context_id": "global", "mean": {k: round(v, 2) for k, v in self.mean.items()}, "observations": self.n}

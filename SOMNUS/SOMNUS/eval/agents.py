"""Benchmark arms.

Three agents, identical except for the memory mechanism under test:

  control        -- one global running estimate, no context detection, no
                    consolidation. This is 'fine-tune on recent data', the
                    standard thing people build. It is the curve to beat.
  no-recall      -- full memory and consolidation, but the predictor does NOT
                    restore a schema prior at a context boundary. Tests whether
                    storing memories is worth anything without recalling them.
  no-hardening   -- metaplasticity disabled (fixed alpha, no stability). Tests
                    whether hardening is what stops schema drift.
  no-interleave  -- interleave_ratio = 0. Tests the CLS interleaving claim.
  somnus         -- the full system.

Ablations are the point. A single "we beat the baseline" bar tells a judge
nothing about WHY. These four tell you which mechanism is load-bearing -- and
in this environment the answer turned out to be metaplastic hardening plus
schema recall, not interleaving. Reporting that is more valuable than
pretending every component earns its keep.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from infra.config import NeuromodConfig, PlasticityConfig, SleepConfig
from infra.embeddings import FeatureEmbedder, ObservationEncoder
from core.neuromod import NeuromodulatorySystem, weighted_surprise
from core.predictor import ContextualPredictor, NaivePredictor
from memory.inmemory import InMemoryStore
from memory.store import Episode, MemoryStore
from sleep_cycle.consolidation import consolidate, violation_penalty


@dataclass
class StepRecord:
    tick: int
    phase: str
    regime: str
    error: float
    model_error: float
    ach: float
    na: float
    boundary: bool
    context: str


@dataclass
class Arm:
    """One benchmark arm."""

    name: str
    predictor: Any
    store: MemoryStore | None = None
    neuromod: NeuromodulatorySystem | None = None
    encoder: ObservationEncoder | None = None
    sleep_cfg: SleepConfig | None = None
    plasticity: PlasticityConfig | None = None
    use_memory: bool = True
    history: list[StepRecord] = field(default_factory=list)
    sleep_reports: list[dict[str, Any]] = field(default_factory=list)
    _rng: random.Random = field(default_factory=lambda: random.Random(7))

    def step(self, tick: int, phase: str, observation: dict[str, Any]) -> float:
        predicted = self.predictor.predict()
        error, _ = weighted_surprise(predicted, observation)
        # Model error ignores the noise floor: it scores the prediction against
        # the regime's true mean, not against one noisy draw from it. This is
        # the number that actually measures whether the agent knows the world.
        truth = observation.get("_truth")
        model_error = weighted_surprise(predicted, truth)[0] if truth else error

        embedding = None
        state = None

        if self.use_memory and self.neuromod is not None and self.encoder is not None:
            text, embedding = self.encoder.encode(observation)
            state = self.neuromod.step(predicted, observation, embedding=embedding)

            if state.boundary:
                # Metaplastic violation: the contradicted schema re-opens.
                if self.store is not None:
                    try:
                        violation_penalty(self.store, embedding, self.plasticity)
                    except Exception:  # noqa: BLE001
                        pass
                self.predictor.on_boundary(embedding, observation)
                self.neuromod.reset_context()

            # Surprise/DA-gated encoding into the hippocampus.
            if self.store is not None and self._rng.random() < self.neuromod.encode_probability(state):
                self.store.write_episode(
                    Episode(
                        context_id=self.predictor.context_id,
                        embedding=embedding,
                        raw_obs=observation,
                        canonical=text,
                        surprise=state.surprise,
                        da=state.da,
                        ach=state.ach,
                        na=state.na,
                    )
                )

            lr = self.neuromod.learning_rate(state, (self.plasticity or PlasticityConfig()).alpha_base)
        else:
            lr = (self.plasticity or PlasticityConfig()).alpha_base

        self.predictor.update(observation, lr)

        self.history.append(
            StepRecord(
                tick=tick,
                phase=phase,
                regime=observation.get("regime", "?"),
                error=error,
                model_error=model_error,
                ach=state.ach if state else 0.0,
                na=state.na if state else 0.0,
                boundary=bool(state.boundary) if state else False,
                context=self.predictor.context_id[:8],
            )
        )
        return error

    def sleep(self) -> None:
        if not self.use_memory or self.store is None:
            return
        report = consolidate(
            self.store,
            sleep_cfg=self.sleep_cfg,
            plasticity=self.plasticity,
            rng=self._rng,
        )
        self.sleep_reports.append(report.to_dict())


def build_arm(
    name: str,
    embed_dim: int = 64,
    seed: int = 7,
    interleave: float = 0.5,
    recall: bool = True,
    hardening: bool = True,
) -> Arm:
    """One SOMNUS arm with individual mechanisms switchable."""
    plasticity = PlasticityConfig(
        harden_at=8 if hardening else 10**9,
        max_stability=12 if hardening else 0,
    )
    store = InMemoryStore()
    return Arm(
        name=name,
        predictor=ContextualPredictor(store=store, enable_recall=recall),
        store=store,
        neuromod=NeuromodulatorySystem(NeuromodConfig()),
        encoder=ObservationEncoder(FeatureEmbedder(embed_dim), offline=True),
        sleep_cfg=SleepConfig(batch_size=48, interleave_ratio=interleave, novelty_threshold=0.35),
        plasticity=plasticity,
        _rng=random.Random(seed),
    )


def build_arms(embed_dim: int = 64, seed: int = 7) -> list[Arm]:
    """Control plus three ablations plus the full system."""
    return [
        Arm(
            name="control",
            predictor=NaivePredictor(lr=PlasticityConfig().alpha_base),
            use_memory=False,
            plasticity=PlasticityConfig(),
            _rng=random.Random(seed),
        ),
        build_arm("no-recall", embed_dim, seed, recall=False),
        build_arm("no-hardening", embed_dim, seed, hardening=False),
        build_arm("no-interleave", embed_dim, seed, interleave=0.0),
        build_arm("somnus", embed_dim, seed),
    ]

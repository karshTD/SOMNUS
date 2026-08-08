"""NREM consolidation. The heart of the system.


"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from infra.config import CONFIG, PlasticityConfig, SleepConfig
from infra.embeddings import cosine_distance
from memory.store import Episode, MemoryStore, Schema

logger = logging.getLogger(__name__)

TRACKED = ("cpu_percent", "rps", "network_mbps")


@dataclass
class ConsolidationReport:
    replayed_new: int = 0
    replayed_interleaved: int = 0
    schemas_created: int = 0
    schemas_updated: int = 0
    episodes_rescued: int = 0
    episodes_expired: int = 0
    hardened: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replayed_new": self.replayed_new,
            "replayed_interleaved": self.replayed_interleaved,
            "schemas_created": self.schemas_created,
            "schemas_updated": self.schemas_updated,
            "episodes_rescued": self.episodes_rescued,
            "episodes_expired": self.episodes_expired,
            "hardened": self.hardened,
        }


def _generative_sample(schema: Schema, rng: random.Random) -> Episode:
    """Draw a pseudo-episode from a schema: centroid + learned variance.

    This is generative replay -- the schema regenerates its own training data
    so it can be rehearsed alongside new experience without storing the
    originals (which have long since expired).
    """
    centroid = np.asarray(schema.centroid, dtype=float)
    if schema.variance and len(schema.variance) == len(centroid):
        sigma = np.sqrt(np.maximum(np.asarray(schema.variance, dtype=float), 0.0))
    else:
        sigma = np.zeros_like(centroid)

    noise = np.array([rng.gauss(0.0, 1.0) for _ in range(len(centroid))])
    sampled = centroid + 0.25 * sigma * noise
    norm = np.linalg.norm(sampled)
    if norm > 0:
        sampled = sampled / norm

    raw = {m: schema.feature_mean.get(m, 0.0) for m in TRACKED}
    return Episode(
        id=f"replay::{schema.id}",
        embedding=sampled.tolist(),
        raw_obs=raw,
        canonical=f"[generative replay] {schema.label}",
        surprise=0.0,
        da=0.0,
    )


def effective_alpha(schema: Schema, plasticity: PlasticityConfig) -> float:
    """Metaplastic cascade: confirmed knowledge hardens against revision."""
    return max(plasticity.alpha_min, plasticity.alpha_base * (2.0**-schema.stability))


def _merge(schema: Schema, episode: Episode, alpha: float) -> None:
    """Move the prototype toward the instance, at the schema's own rate."""
    c = np.asarray(schema.centroid, dtype=float)
    e = np.asarray(episode.embedding, dtype=float)
    delta = e - c
    new_c = c + alpha * delta

    # Welford-ish running variance on the residual
    if schema.variance and len(schema.variance) == len(c):
        v = np.asarray(schema.variance, dtype=float)
    else:
        v = np.zeros_like(c)
    schema.variance = ((1 - alpha) * v + alpha * delta**2).tolist()

    norm = np.linalg.norm(new_c)
    schema.centroid = (new_c / norm).tolist() if norm > 0 else new_c.tolist()

    for metric in TRACKED:
        value = float(episode.raw_obs.get(metric, 0.0))
        current = schema.feature_mean.get(metric)
        schema.feature_mean[metric] = value if current is None else current + alpha * (value - current)

    schema.support_count += 1


def consolidate(
    store: MemoryStore,
    sleep_cfg: SleepConfig | None = None,
    plasticity: PlasticityConfig | None = None,
    rng: random.Random | None = None,
    rule_writer: Callable[[list[Episode]], str] | None = None,
    skill_emitter: Callable[[Schema], str] | None = None,
) -> ConsolidationReport:
    """Run one NREM pass. Store-agnostic: works on CockroachDB or in-memory."""
    sleep_cfg = sleep_cfg or CONFIG.sleep
    plasticity = plasticity or CONFIG.plasticity
    rng = rng or random.Random()
    report = ConsolidationReport()

    n_new = max(1, int(sleep_cfg.batch_size * (1 - sleep_cfg.interleave_ratio)))
    n_old = sleep_cfg.batch_size - n_new

    new_episodes = store.sample_episodes(n_new, unconsolidated_only=True)
    if not new_episodes:
        report.episodes_expired = store.purge_expired()
        return report

    # --- INTERLEAVING: rehearse existing knowledge alongside new experience.
    existing = store.all_schemas()
    interleaved: list[Episode] = []
    if existing and n_old > 0 and sleep_cfg.interleave_ratio > 0:
        for _ in range(n_old):
            interleaved.append(_generative_sample(rng.choice(existing), rng))

    report.replayed_new = len(new_episodes)
    report.replayed_interleaved = len(interleaved)

    batch = new_episodes + interleaved
    rng.shuffle(batch)

    rescued: list[str] = []
    touched: dict[str, Schema] = {s.id: s for s in existing}

    for episode in batch:
        is_replay = episode.id.startswith("replay::")

        match = store.nearest_schema(episode.embedding)
        if match is None or match.distance > sleep_cfg.novelty_threshold:
            schema = Schema(
                label=episode.canonical[:120] or "unlabelled",
                centroid=list(episode.embedding),
                variance=[0.0] * len(episode.embedding),
                feature_mean={m: float(episode.raw_obs.get(m, 0.0)) for m in TRACKED},
                support_count=1,
                stability=0,
                origin="nrem",
            )
            store.create_schema(schema)
            touched[schema.id] = schema
            report.schemas_created += 1
        else:
            schema = touched.get(match.schema.id, match.schema)
            alpha = effective_alpha(schema, plasticity)
            _merge(schema, episode, alpha)
            schema.stability = min(plasticity.max_stability, schema.stability + 1)
            store.update_schema(schema)
            touched[schema.id] = schema
            report.schemas_updated += 1

        if not is_replay:
            store.link_provenance(schema.id, episode.id)
            rescued.append(episode.id)

        report.details.append(
            {
                "episode": episode.id,
                "kind": "interleaved" if is_replay else "new",
                "schema": schema.id,
                "stability": schema.stability,
                "alpha": round(effective_alpha(schema, plasticity), 4),
            }
        )

    # --- TTL RESCUE. Replayed episodes survive; the rest simply expire.
    store.mark_replayed(rescued, sleep_cfg.rescue_days)
    report.episodes_rescued = len(rescued)

    # --- Optional: Bedrock rule text + skill compilation for hardened schemas.
    for schema in touched.values():
        if schema.stability >= plasticity.harden_at and schema.skill_ref is None:
            if skill_emitter is not None:
                try:
                    schema.skill_ref = skill_emitter(schema)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skill emission failed: %s", exc)
            if rule_writer is not None and not schema.rule_text:
                try:
                    sources = [e for e in new_episodes if not e.id.startswith("replay::")]
                    schema.rule_text = rule_writer(sources[:8])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Rule generation failed: %s", exc)
            store.update_schema(schema)
            report.hardened.append(schema.label or schema.id)

    report.episodes_expired = store.purge_expired()
    return report


def violation_penalty(store: MemoryStore, embedding: list[float], plasticity: PlasticityConfig | None = None) -> str | None:
    """An NA spike re-opens the violated schema to revision.

    Knowledge that has earned trust resists change; knowledge that is
    contradicted becomes plastic again. This is the other half of
    metaplasticity -- without it, hardening is just freezing.
    """
    plasticity = plasticity or CONFIG.plasticity
    match = store.nearest_schema(embedding)
    if match is None:
        return None
    schema = match.schema
    before = schema.stability
    schema.stability = max(0, schema.stability - plasticity.violation_penalty)
    if schema.stability != before:
        store.update_schema(schema)
        logger.info("Metaplastic violation: schema %s stability %d -> %d", schema.id, before, schema.stability)
    return schema.id

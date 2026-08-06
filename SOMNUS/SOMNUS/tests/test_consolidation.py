"""Consolidation: merge-or-spawn, interleaving, metaplasticity, TTL rescue."""

import random
from datetime import datetime, timedelta, timezone

import pytest

from infra.config import PlasticityConfig, SleepConfig
from memory.inmemory import InMemoryStore
from memory.store import Episode, Schema
from sleep_cycle.consolidation import consolidate, effective_alpha, violation_penalty


def _episode(vec, cpu=35.0, **kw) -> Episode:
    return Episode(
        embedding=list(vec),
        raw_obs={"cpu_percent": cpu, "rps": 120.0, "network_mbps": 50.0},
        canonical=kw.pop("canonical", "test state"),
        surprise=kw.pop("surprise", 0.5),
        da=kw.pop("da", 0.5),
        **kw,
    )


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def test_similar_episodes_merge_into_one_schema(store) -> None:
    rng = random.Random(0)
    for _ in range(40):
        store.write_episode(_episode([1.0 + rng.gauss(0, 0.01), 0.02, 0.0]))
    consolidate(store, rng=random.Random(1))
    assert store.schema_count() == 1, "near-identical episodes must not spawn many schemas"


def test_distinct_episodes_spawn_separate_schemas(store) -> None:
    for _ in range(10):
        store.write_episode(_episode([1.0, 0.0, 0.0]))
    for _ in range(10):
        store.write_episode(_episode([-1.0, 0.0, 0.0]))
    consolidate(store, rng=random.Random(1))
    assert store.schema_count() >= 2, "opposed episodes must not be merged"


def test_metaplasticity_hardens_confirmed_schemas() -> None:
    p = PlasticityConfig()
    fresh, mid, seasoned = Schema(stability=0), Schema(stability=4), Schema(stability=10)
    assert effective_alpha(fresh, p) == pytest.approx(p.alpha_base)
    assert effective_alpha(fresh, p) > effective_alpha(mid, p) > effective_alpha(seasoned, p)
    assert effective_alpha(fresh, p) > effective_alpha(seasoned, p) * 10
    assert effective_alpha(seasoned, p) >= p.alpha_min


def test_violation_reopens_a_hardened_schema(store) -> None:
    schema = Schema(centroid=[1.0, 0.0, 0.0], stability=10, feature_mean={"cpu_percent": 35})
    store.create_schema(schema)
    violation_penalty(store, [1.0, 0.0, 0.0], PlasticityConfig())
    assert store.all_schemas()[0].stability == 10 - PlasticityConfig().violation_penalty


def test_replay_rescues_episodes_from_ttl(store) -> None:
    ep = _episode([1.0, 0.0, 0.0])
    ep.expire_at = datetime.now(timezone.utc) + timedelta(seconds=1)
    store.write_episode(ep)
    consolidate(store, rng=random.Random(1))
    assert store._episodes[ep.id].expire_at > datetime.now(timezone.utc) + timedelta(days=1)


def test_unreplayed_episodes_expire(store) -> None:
    doomed = _episode([1.0, 0.0, 0.0])
    doomed.expire_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    store.write_episode(doomed)
    assert store.episode_count() == 0, "TTL is the forgetting mechanism; no delete code needed"


def test_interleaving_injects_generative_replay(store) -> None:
    store.create_schema(
        Schema(centroid=[0.0, 1.0, 0.0], variance=[0.01] * 3, feature_mean={"cpu_percent": 78})
    )
    for _ in range(8):
        store.write_episode(_episode([1.0, 0.0, 0.0]))
    report = consolidate(
        store, sleep_cfg=SleepConfig(batch_size=16, interleave_ratio=0.5), rng=random.Random(2)
    )
    assert report.replayed_interleaved > 0, "existing knowledge must be rehearsed alongside new"


def test_provenance_is_recorded(store) -> None:
    for _ in range(6):
        store.write_episode(_episode([1.0, 0.0, 0.0]))
    consolidate(store, rng=random.Random(1))
    schema = store.all_schemas()[0]
    assert len(store.provenance_for(schema.id)) > 0, "'why do you believe this' must be a JOIN"


def test_empty_store_is_a_noop(store) -> None:
    assert consolidate(store).replayed_new == 0

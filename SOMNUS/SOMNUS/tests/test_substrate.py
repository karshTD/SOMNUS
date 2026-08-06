"""Substrate contracts and the two bugs that made the old build unrunnable."""

import pytest

from infra.config import EMBEDDING_DIMENSION
from infra.embeddings import (
    CachingEmbedder,
    FeatureEmbedder,
    LocalEmbedder,
    ObservationEncoder,
    canonical_text,
    cosine_distance,
)
from memory.inmemory import InMemoryStore
from memory.separation import PatternSeparator
from memory.store import MemoryStore
from substrate.migrate import OPTIONAL, REQUIRED


def test_pattern_separation_preserves_identity() -> None:
    """Separation must be deterministic: the same cue must retrieve the same trace."""
    sep = PatternSeparator(128)
    a = [float(i % 7) for i in range(128)]
    assert sep.separate(a) == sep.separate(a)


def test_embedding_dimension_is_valid_for_titan_v2() -> None:
    """1536 is Titan v1. Titan v2 accepts only 1024/512/256 and raises otherwise."""
    assert EMBEDDING_DIMENSION in (256, 512, 1024)


def test_migration_uses_cockroachdb_not_pgvector_syntax() -> None:
    ddl = " ".join(sql for _, sql in REQUIRED + OPTIONAL).lower()
    assert "create extension" not in ddl, "CockroachDB has a native VECTOR type"
    assert "using hnsw" not in ddl, "CockroachDB uses C-SPANN, not HNSW"
    assert "vector index" in ddl
    assert "ttl_expiration_expression" in ddl, "forgetting must be a substrate property"


def test_canonicalisation_collapses_float_noise() -> None:
    a = canonical_text({"cpu_percent": 94.72, "rps": 1101.3, "network_mbps": 500.1})
    b = canonical_text({"cpu_percent": 94.79, "rps": 1098.7, "network_mbps": 504.9})
    assert a == b, "embedding raw JSON embeds noise and destroys cache hit rate"


def test_cache_avoids_repeat_embedding_calls() -> None:
    embedder = CachingEmbedder(LocalEmbedder(64))
    for _ in range(20):
        embedder.embed("system state: cpu light")
    assert embedder.misses == 1


def test_feature_embedder_preserves_geometry() -> None:
    """Similar states must be near; different regimes must be far."""
    f = FeatureEmbedder(64)
    quiet_a = f.embed_observation({"cpu_percent": 35, "rps": 120, "network_mbps": 50})
    quiet_b = f.embed_observation({"cpu_percent": 37, "rps": 125, "network_mbps": 52})
    loud = f.embed_observation({"cpu_percent": 90, "rps": 900, "network_mbps": 480})
    assert cosine_distance(quiet_a, quiet_b) < cosine_distance(quiet_a, loud)


def test_pattern_separation_decorrelates_overlapping_inputs() -> None:
    """Similar-but-distinct incidents must not collide in the episodic store."""
    import random

    rng = random.Random(4)
    sep = PatternSeparator(128)
    a = [rng.gauss(0, 1) for _ in range(128)]
    b = list(a)
    for i in range(64):  # 50% of the pattern replaced
        b[i] = rng.gauss(0, 1)
    before = cosine_distance(a, b)
    after = cosine_distance(sep.separate(a), sep.separate(b))
    assert after > before, f"separation should push overlapping patterns apart ({before=}, {after=})"


def test_encoder_returns_text_and_vector() -> None:
    encoder = ObservationEncoder(FeatureEmbedder(32), offline=True)
    text, vector = encoder.encode({"cpu_percent": 35, "rps": 120, "network_mbps": 50})
    assert text.startswith("system state:") and len(vector) == 32


def test_in_memory_store_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryStore(), MemoryStore)

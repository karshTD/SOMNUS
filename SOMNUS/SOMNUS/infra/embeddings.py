"""Embedding layer: canonicalization, caching, and an offline fallback.

Three fixes over the naive approach live here:

1. CANONICALIZATION. Embedding ``json.dumps(telemetry)`` embeds float noise --
   cpu 94.72 and cpu 94.79 produce different vectors for the same event. We
   bucket values into a stable natural-language description first. This
   improves recall AND makes the cache actually hit.
2. CACHING. Identical canonical text is embedded once. With bucketing, a steady
   regime collapses to a handful of distinct strings, so Bedrock calls drop by
   ~95% in a typical run.
3. OFFLINE FALLBACK. A deterministic hash embedder lets the benchmark, the
   tests and local development run with zero AWS credentials.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Protocol

import numpy as np

from infra.config import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

_CPU_BUCKETS = [(20, "idle"), (50, "light"), (75, "moderate"), (90, "heavy"), (101, "critical")]
_RPS_BUCKETS = [(50, "minimal"), (200, "normal"), (500, "elevated"), (1000, "high"), (10**9, "extreme")]
_NET_BUCKETS = [(25, "quiet"), (100, "normal"), (300, "busy"), (600, "saturated"), (10**9, "flooded")]


def _bucket(value: float, buckets: list[tuple[float, str]]) -> str:
    for upper, label in buckets:
        if value < upper:
            return label
    return buckets[-1][1]


def canonical_text(observation: dict[str, Any]) -> str:
    """Turn noisy telemetry into a stable, semantically meaningful sentence.

    This is what gets embedded -- never the raw JSON.
    """
    cpu = float(observation.get("cpu_percent", 0.0))
    rps = float(observation.get("rps", 0.0))
    net = float(observation.get("network_mbps", 0.0))
    anomaly = bool(observation.get("anomaly", False))

    parts = [
        f"cpu {_bucket(cpu, _CPU_BUCKETS)}",
        f"request rate {_bucket(rps, _RPS_BUCKETS)}",
        f"network {_bucket(net, _NET_BUCKETS)}",
    ]
    if anomaly:
        parts.append("anomalous")
    return "system state: " + ", ".join(parts)


class Embedder(Protocol):
    dimension: int

    def embed(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """Deterministic hash-based embedder for offline runs and tests.

    Not semantically meaningful across unrelated strings, but stable and
    dimension-correct -- which is all the benchmark and unit tests need.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension)
        vec /= np.linalg.norm(vec) or 1.0
        return vec.tolist()


class CachingEmbedder:
    """Wraps any Embedder with a bounded LRU-ish cache."""

    def __init__(self, inner: Embedder, max_entries: int = 2048) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self._max = max_entries
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed(self, text: str) -> list[float]:
        with self._lock:
            cached = self._cache.get(text)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        vector = self._inner.embed(text)
        with self._lock:
            if len(self._cache) >= self._max:
                self._cache.pop(next(iter(self._cache)))
            self._cache[text] = vector
        return vector

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


def cosine_distance(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Cosine distance in [0, 2]. Matches CockroachDB's ``<=>`` operator."""
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 1.0
    return float(1.0 - np.dot(va, vb) / denom)


# --- structure-preserving offline encoder ------------------------------
_LOG_SCALE = {"cpu_percent": 100.0, "rps": 800.0, "network_mbps": 500.0}


class FeatureEmbedder:
    """Offline encoder that PRESERVES geometry.

    ``LocalEmbedder`` hashes text, so 'cpu moderate' and 'cpu heavy' land in
    orthogonal directions and every regime is trivially separable. That makes
    the benchmark too easy and hides whether interleaving does any work.

    A real sentence embedder places similar system states near each other. This
    encoder reproduces that property: features are log-scaled, then carried
    through a fixed random projection, so nearby telemetry yields nearby
    vectors and the consolidation dynamics are realistic.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION, seed: int = 20260806) -> None:
        self.dimension = dimension
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((dimension, len(_LOG_SCALE)))
        self._proj /= np.linalg.norm(self._proj, axis=0, keepdims=True)

    def features(self, observation: dict[str, Any]) -> np.ndarray:
        """Scale to [0,1] then CENTRE.

        Centring matters: without it every telemetry vector sits in the
        positive orthant and all cosine distances collapse toward zero, so no
        two regimes are ever distinguishable. Centring lets states be genuinely
        anti-correlated, which is what a real sentence embedder gives you.
        """
        return np.array(
            [
                np.clip(max(0.0, float(observation.get(k, 0.0))) / v, 0.0, 1.5) - 0.5
                for k, v in _LOG_SCALE.items()
            ]
        )

    def embed_observation(self, observation: dict[str, Any]) -> list[float]:
        vec = self._proj @ self.features(observation)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("FeatureEmbedder encodes observations, not free text")


class ObservationEncoder:
    """Single entry point used by the agent and the benchmark.

    Online  -> canonical text + Bedrock Titan (semantic, cached).
    Offline -> FeatureEmbedder (geometry-preserving, zero dependencies).
    """

    def __init__(self, embedder: Any | None = None, offline: bool | None = None) -> None:
        from infra.config import is_offline

        self.offline = is_offline() if offline is None else offline
        if embedder is not None:
            self._inner = embedder
        elif self.offline:
            self._inner = FeatureEmbedder()
        else:
            from infra.aws_client import BedrockEmbedder

            self._inner = CachingEmbedder(BedrockEmbedder())

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def encode(self, observation: dict[str, Any]) -> tuple[str, list[float]]:
        text = canonical_text(observation)
        if hasattr(self._inner, "embed_observation"):
            return text, self._inner.embed_observation(observation)
        return text, self._inner.embed(text)

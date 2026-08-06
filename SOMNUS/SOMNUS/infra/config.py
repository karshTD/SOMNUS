"""Central configuration. Every tunable lives here or in the DB, never inline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _i(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _b(key: str, default: bool = False) -> bool:
    return os.getenv(key, "1" if default else "0").lower() in {"1", "true", "yes"}


# Titan Text Embeddings V2 accepts ONLY 1024 (default), 512, or 256.
# 1536 is the v1 size and raises ValidationException on v2.
EMBEDDING_DIMENSION = _i("SOMNUS_EMBED_DIM", 1024)
EMBEDDING_MODEL_ID = os.getenv("SOMNUS_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
REASONING_MODEL_ID = os.getenv("SOMNUS_REASON_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")

if EMBEDDING_DIMENSION not in (256, 512, 1024):
    raise ValueError(
        f"SOMNUS_EMBED_DIM={EMBEDDING_DIMENSION} is invalid for Titan v2. "
        "Accepted values: 1024, 512, 256."
    )

def is_offline() -> bool:
    """Resolved at call time, not import time, so tests and notebooks can
    toggle it without reimporting the package."""
    return _b("SOMNUS_OFFLINE", False) or not os.getenv("COCKROACH_DB_URL", "")


OFFLINE = is_offline()


@dataclass(frozen=True)
class NeuromodConfig:
    """Neuromodulatory nuclei parameters."""

    ach_halflife: int = _i("SOMNUS_ACH_HALFLIFE", 80)
    ach_floor: float = _f("SOMNUS_ACH_FLOOR", 0.02)
    # CUSUM parameters. na_slack is the per-tick allowance subtracted before
    # accumulating; na_threshold is the decision boundary on the accumulator.
    na_slack: float = _f("SOMNUS_NA_SLACK", 1.2)
    na_threshold: float = _f("SOMNUS_NA_THRESHOLD", 3.0)
    warmup_ticks: int = _i("SOMNUS_WARMUP_TICKS", 25)
    da_novelty_beta: float = _f("SOMNUS_DA_NOVELTY_BETA", 0.5)
    encode_floor: float = _f("SOMNUS_ENCODE_FLOOR", 0.05)


@dataclass(frozen=True)
class PlasticityConfig:
    """Learning-rate and metaplasticity parameters."""

    alpha_base: float = _f("SOMNUS_ALPHA_BASE", 0.06)
    alpha_min: float = _f("SOMNUS_ALPHA_MIN", 0.002)
    ach_gain: float = _f("SOMNUS_ACH_GAIN", 1.5)
    harden_at: int = _i("SOMNUS_HARDEN_AT", 8)
    violation_penalty: int = _i("SOMNUS_VIOLATION_PENALTY", 3)
    max_stability: int = _i("SOMNUS_MAX_STABILITY", 12)


@dataclass(frozen=True)
class SleepConfig:
    """Consolidation parameters."""

    batch_size: int = _i("SOMNUS_SLEEP_BATCH", 64)
    interleave_ratio: float = _f("SOMNUS_INTERLEAVE", 0.5)
    novelty_threshold: float = _f("SOMNUS_NOVELTY_THRESHOLD", 0.35)
    rescue_days: int = _i("SOMNUS_RESCUE_DAYS", 30)
    generate_rules: bool = _b("SOMNUS_GENERATE_RULES", True)


@dataclass(frozen=True)
class Config:
    db_url: str = field(default_factory=lambda: os.getenv("COCKROACH_DB_URL", ""))
    cluster_id: str = field(default_factory=lambda: os.getenv("COCKROACH_CLUSTER_ID", ""))
    region: str = field(default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_ARCHIVE_BUCKET", ""))
    episode_ttl_hours: int = _i("SOMNUS_EPISODE_TTL_HOURS", 24)
    poll_interval: float = _f("SOMNUS_POLL_INTERVAL", 2.0)

    neuromod: NeuromodConfig = field(default_factory=NeuromodConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)


CONFIG = Config()

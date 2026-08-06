"""Declarative -> procedural: hardened schemas compile into Agent Skills.

When a schema survives enough confirming replays that metaplasticity has
effectively frozen it, the agent writes it out as a reusable skill. This is
basal-ganglia-style habit formation: knowledge that has been confirmed often
enough stops being deliberated over and becomes a procedure.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from memory.store import Schema

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(os.getenv("SOMNUS_SKILLS_DIR", Path(__file__).parent.parent / "skills"))

TEMPLATE = """---
name: {slug}
description: >-
  Auto-compiled from a consolidated SOMNUS schema after {stability} confirming
  replays across {support} episodes. Generated, not hand-written.
origin: {origin}
schema_id: {schema_id}
---

# {label}

## Learned regime

{features}

## Rule

{rule}

## Provenance

This skill exists because the schema survived {stability} consolidation passes
without being contradicted. Metaplasticity had frozen its learning rate to
{alpha:.4f}, at which point it was compiled to a procedure.
"""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "schema"


def emit_skill(schema: Schema) -> str:
    """Write a skill file for a hardened schema. Returns the relative path."""
    from infra.config import CONFIG
    from sleep_cycle.consolidation import effective_alpha

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(schema.label or schema.id)
    path = SKILLS_DIR / f"{slug}.md"

    features = "\n".join(f"- **{k}**: {v:.1f}" for k, v in sorted(schema.feature_mean.items())) or "- (none)"
    path.write_text(
        TEMPLATE.format(
            slug=slug,
            label=schema.label or slug,
            stability=schema.stability,
            support=schema.support_count,
            origin=schema.origin,
            schema_id=schema.id,
            features=features,
            rule=schema.rule_text or "_No natural-language rule generated (Bedrock disabled)._",
            alpha=effective_alpha(schema, CONFIG.plasticity),
        ),
        encoding="utf-8",
    )
    logger.info("Compiled schema %s to skill %s", schema.id, path.name)
    return str(path.relative_to(SKILLS_DIR.parent))

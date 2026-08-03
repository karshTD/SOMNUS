"""Naive RAG baseline for evaluation against SOMNUS dual-memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from infra.aws_client import AWSClient
from memory.cortex import Cortex
from memory.hippocampus import Hippocampus


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[dict[str, Any]]
    method: str = "baseline_rag"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "method": self.method,
        }


class BaselineRAG:
    """
    Standard RAG: embed query -> vector search cortex -> prompt LLM.
    Does NOT consult hippocampus episodic memory (recent un-vectorized events).
    """

    def __init__(
        self,
        aws: AWSClient | None = None,
        cortex: Cortex | None = None,
    ) -> None:
        self.aws = aws or AWSClient()
        self.cortex = cortex or Cortex()

    def query(self, question: str, top_k: int = 3) -> RAGResponse:
        vector = self.aws.embed_text(question)
        rules = self.cortex.recall_similar(vector, limit=top_k)

        context_blocks = [r.rule_text for r in rules]
        prompt = f"""Answer the question using ONLY the context below.
If the context is insufficient, say you don't know.

Context:
{chr(10).join(f"- {c}" for c in context_blocks) or "(empty)"}

Question: {question}
"""
        answer = self.aws.reason(prompt)
        sources = [r.to_dict() for r in rules]
        return RAGResponse(query=question, answer=answer, sources=sources)


class SomnusQuery:
    """
    SOMNUS-aware query: combines cortex rules AND recent hippocampus episodes.
    Demonstrates superiority for recent/unconsolidated context.
    """

    def __init__(
        self,
        aws: AWSClient | None = None,
        cortex: Cortex | None = None,
        hippocampus: Hippocampus | None = None,
    ) -> None:
        self.aws = aws or AWSClient()
        self.cortex = cortex or Cortex()
        self.hippocampus = hippocampus or Hippocampus(self.aws)

    def query(self, question: str, top_k: int = 3) -> RAGResponse:
        vector = self.aws.embed_text(question)
        rules = self.cortex.recall_similar(vector, limit=top_k)

        recent_episodes: list[dict[str, Any]] = []
        for key in self.hippocampus.list_recent_keys(limit=5):
            try:
                recent_episodes.append(self.aws.read_json(key))
            except Exception:
                pass

        cortex_context = [r.rule_text for r in rules]
        episode_context = [
            json.dumps(ep.get("actual", ep), default=str) for ep in recent_episodes
        ]

        prompt = f"""Answer using consolidated rules AND recent episodic anomalies.
Recent episodes may contain context not yet vectorized into long-term memory.

Consolidated rules:
{chr(10).join(f"- {c}" for c in cortex_context) or "(none)"}

Recent hippocampus episodes:
{chr(10).join(f"- {c}" for c in episode_context) or "(none)"}

Question: {question}
"""
        answer = self.aws.reason(prompt)
        sources = [r.to_dict() for r in rules] + [
            {"episode": ep.get("episode_id", "unknown")} for ep in recent_episodes
        ]
        return RAGResponse(query=question, answer=answer, sources=sources, method="somnus")

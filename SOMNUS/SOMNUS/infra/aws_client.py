"""AWS wrapper: lazy clients, paginated S3, batched deletes, parallel reads."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from infra.config import CONFIG, EMBEDDING_DIMENSION, EMBEDDING_MODEL_ID, REASONING_MODEL_ID

logger = logging.getLogger(__name__)


class AWSClient:
    """S3 + Bedrock. Clients are built on first use, not in __init__.

    Eager construction cost ~100-300ms per client, and both the MCP server and
    the dashboard instantiate this class repeatedly.
    """

    def __init__(self, region: str | None = None, s3_bucket: str | None = None) -> None:
        self.region = region or CONFIG.region
        self.s3_bucket = s3_bucket or CONFIG.s3_bucket
        self.__s3: Any = None
        self.__bedrock: Any = None

    def _client(self, service: str) -> Any:
        import boto3

        return boto3.client(service, region_name=self.region)

    @property
    def s3(self) -> Any:
        if self.__s3 is None:
            self.__s3 = self._client("s3")
        return self.__s3

    @property
    def bedrock(self) -> Any:
        if self.__bedrock is None:
            self.__bedrock = self._client("bedrock-runtime")
        return self.__bedrock

    # --- S3 archive ----------------------------------------------------
    def write_json(self, key: str, data: dict[str, Any]) -> str:
        if not self.s3_bucket:
            raise ValueError("S3_ARCHIVE_BUCKET is not configured")
        self.s3.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=json.dumps(data, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        return key

    def list_keys(self, prefix: str = "archive/") -> list[str]:
        """Paginated. ``list_objects_v2`` silently caps at 1000 keys."""
        if not self.s3_bucket:
            return []
        keys: list[str] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    def read_json(self, key: str) -> dict[str, Any]:
        body = self.s3.get_object(Bucket=self.s3_bucket, Key=key)["Body"].read()
        return json.loads(body.decode("utf-8"))

    def read_many(self, keys: list[str], workers: int = 16) -> list[dict[str, Any]]:
        """Parallel reads. S3 GETs are IO-bound at 50-150ms each; serial reads
        of 100 objects take ~10s, parallel takes ~1s."""
        if not keys:
            return []
        out: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(self._safe_read, keys):
                if result is not None:
                    out.append(result)
        return out

    def _safe_read(self, key: str) -> dict[str, Any] | None:
        try:
            return self.read_json(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", key, exc)
            return None

    def delete_keys(self, keys: list[str]) -> int:
        """Batched: 1000 per call instead of one API call per object."""
        deleted = 0
        for i in range(0, len(keys), 1000):
            chunk = keys[i : i + 1000]
            self.s3.delete_objects(
                Bucket=self.s3_bucket,
                Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
            )
            deleted += len(chunk)
        return deleted

    # --- Bedrock -------------------------------------------------------
    def embed_text(self, text: str) -> list[float]:
        body = json.dumps(
            {"inputText": text, "dimensions": EMBEDDING_DIMENSION, "normalize": True}
        )
        response = self.bedrock.invoke_model(
            modelId=EMBEDDING_MODEL_ID, body=body, contentType="application/json", accept="application/json"
        )
        return json.loads(response["body"].read())["embedding"]

    def reason(self, prompt: str, max_tokens: int = 512) -> str:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        response = self.bedrock.invoke_model(
            modelId=REASONING_MODEL_ID, body=body, contentType="application/json", accept="application/json"
        )
        return json.loads(response["body"].read())["content"][0]["text"]

    def health_check(self) -> dict[str, Any]:
        status: dict[str, Any] = {"s3": False, "bedrock": False, "errors": []}
        try:
            if self.s3_bucket:
                self.s3.head_bucket(Bucket=self.s3_bucket)
                status["s3"] = True
            else:
                status["errors"].append("S3_ARCHIVE_BUCKET unset")
        except Exception as exc:  # noqa: BLE001
            status["errors"].append(f"s3: {exc}")
        try:
            vec = self.embed_text("health check")
            status["bedrock"] = len(vec) == EMBEDDING_DIMENSION
            if not status["bedrock"]:
                status["errors"].append(f"embedding dim {len(vec)} != {EMBEDDING_DIMENSION}")
        except Exception as exc:  # noqa: BLE001
            status["errors"].append(f"bedrock: {exc}")
        return status


class BedrockEmbedder:
    """Embedder adapter for infra.embeddings.ObservationEncoder."""

    def __init__(self, client: AWSClient | None = None) -> None:
        self._client = client or AWSClient()
        self.dimension = EMBEDDING_DIMENSION

    def embed(self, text: str) -> list[float]:
        return self._client.embed_text(text)

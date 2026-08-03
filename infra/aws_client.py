"""Centralized AWS wrapper for S3 and Bedrock."""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
REASONING_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
EMBEDDING_DIMENSION = 1536


class AWSClient:
    """Boto3 clients for S3 (reads/writes) and Bedrock (Titan + Claude)."""

    def __init__(
        self,
        region: str | None = None,
        s3_bucket: str | None = None,
    ) -> None:
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.s3_bucket = s3_bucket or os.getenv("S3_HIPPOCAMPUS_BUCKET", "")
        self._s3 = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self._bedrock = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    def write_json(self, key: str, data: dict[str, Any]) -> str:
        """Write JSON payload to the hippocampus S3 bucket."""
        if not self.s3_bucket:
            raise ValueError("S3_HIPPOCAMPUS_BUCKET is not configured")
        body = json.dumps(data, default=str)
        self._s3.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        return key

    def list_episodes(self, prefix: str = "episodes/") -> list[str]:
        """List episode keys under a prefix."""
        if not self.s3_bucket:
            return []
        response = self._s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def read_json(self, key: str) -> dict[str, Any]:
        """Read and parse a JSON object from S3."""
        response = self._s3.get_object(Bucket=self.s3_bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))

    def delete_object(self, key: str) -> None:
        """Delete an object from S3."""
        self._s3.delete_object(Bucket=self.s3_bucket, Key=key)

    def embed_text(self, text: str) -> list[float]:
        """Generate a vector embedding via Bedrock Titan."""
        body = json.dumps({"inputText": text, "dimensions": EMBEDDING_DIMENSION})
        response = self._bedrock.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["embedding"]

    def reason(self, prompt: str, max_tokens: int = 1024) -> str:
        """Invoke Claude on Bedrock for reasoning / summarization."""
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        response = self._bedrock.invoke_model(
            modelId=REASONING_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["content"][0]["text"]

    def health_check(self) -> dict[str, bool]:
        """Validate S3 bucket access and Bedrock availability."""
        status = {"s3": False, "bedrock": False}
        try:
            if self.s3_bucket:
                self._s3.head_bucket(Bucket=self.s3_bucket)
                status["s3"] = True
        except ClientError:
            pass
        try:
            self.embed_text("health check")
            status["bedrock"] = True
        except ClientError:
            pass
        return status

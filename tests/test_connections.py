"""Validate CockroachDB and AWS Bedrock/S3 credentials."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("COCKROACH_DB_URL") or not os.getenv("AWS_ACCESS_KEY_ID"),
    reason="Live credentials not configured",
)


def test_cockroachdb_connection() -> None:
    from memory.cortex import Cortex

    cortex = Cortex()
    assert cortex.health_check() is True
    cortex.close()


def test_aws_s3_and_bedrock() -> None:
    from infra.aws_client import AWSClient

    client = AWSClient()
    status = client.health_check()
    assert status["s3"] is True
    assert status["bedrock"] is True

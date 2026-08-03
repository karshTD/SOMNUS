"""Unit tests for sleep cycle: S3 -> Bedrock -> CockroachDB transformation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_episodes() -> list[dict]:
    return [
        {
            "episode_id": "ep-001",
            "actual": {"cpu_percent": 95, "rps": 1100, "network_mbps": 500},
            "decision": {"action": "rate_limit"},
        },
        {
            "episode_id": "ep-002",
            "actual": {"cpu_percent": 88, "rps": 950, "network_mbps": 420},
            "decision": {"action": "scale_service"},
        },
    ]


def test_consolidation_flow(mock_episodes: list[dict]) -> None:
    from sleep_cycle.lambda_handler import consolidate_episodes

    mock_aws = MagicMock()
    mock_aws.list_episodes.return_value = ["episodes/20260101/ep-001.json"]
    mock_aws.read_json.return_value = mock_episodes[0]
    mock_aws.reason.return_value = "Apply rate limiting when RPS exceeds 800 during CPU spikes."
    mock_aws.embed_text.return_value = [0.1] * 1536

    mock_cortex = MagicMock()
    mock_cortex.insert_rule.return_value = 42

    result = consolidate_episodes(aws=mock_aws, cortex=mock_cortex)

    assert result["consolidated"] == 1
    assert result["rule_id"] == 42
    assert "rate limiting" in result["rule_text"].lower()
    mock_aws.reason.assert_called_once()
    mock_aws.embed_text.assert_called_once()
    mock_cortex.insert_rule.assert_called_once()
    mock_aws.delete_object.assert_called()


def test_consolidation_empty_bucket() -> None:
    from sleep_cycle.lambda_handler import consolidate_episodes

    mock_aws = MagicMock()
    mock_aws.list_episodes.return_value = []

    result = consolidate_episodes(aws=mock_aws, cortex=MagicMock())

    assert result["consolidated"] == 0
    assert "No episodes" in result["message"]


def test_lambda_handler_success() -> None:
    from sleep_cycle.lambda_handler import handler

    with patch("sleep_cycle.lambda_handler.consolidate_episodes") as mock_consolidate:
        mock_consolidate.return_value = {"consolidated": 1, "rule_id": 1}
        response = handler({}, None)

    assert response["statusCode"] == 200
    assert "consolidated" in response["body"]

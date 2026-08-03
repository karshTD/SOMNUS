"""Unit tests for neuromodulation / prediction error."""

from core.neuromod import compute_prediction_error


def test_no_surprise_when_close() -> None:
    predicted = {"cpu_percent": 35.0, "rps": 120.0, "network_mbps": 50.0}
    actual = {"cpu_percent": 36.0, "rps": 118.0, "network_mbps": 52.0}
    result = compute_prediction_error(predicted, actual)
    assert result.error_score < 0.35
    assert result.is_surprised is False


def test_surprise_on_anomaly() -> None:
    predicted = {"cpu_percent": 35.0, "rps": 120.0, "network_mbps": 50.0}
    actual = {"cpu_percent": 95.0, "rps": 1100.0, "network_mbps": 500.0, "anomaly": True}
    result = compute_prediction_error(predicted, actual)
    assert result.is_surprised is True
    assert result.error_score > 0.35

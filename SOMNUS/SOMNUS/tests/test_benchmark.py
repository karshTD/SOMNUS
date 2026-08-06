"""The headline claim must actually hold, and must hold across seeds."""

import pytest

from eval.benchmark import run_benchmark


@pytest.fixture(scope="module")
def result() -> dict:
    return run_benchmark(seed=7, verbose=False)


def _by_name(result: dict) -> dict:
    return {r["name"]: r for r in result["results"]}


def test_control_forgets_catastrophically(result) -> None:
    assert _by_name(result)["control"]["forgetting"] > 0.3


def test_somnus_forgets_far_less_than_control(result) -> None:
    arms = _by_name(result)
    assert arms["somnus"]["forgetting"] < arms["control"]["forgetting"] * 0.6


def test_all_arms_master_task_a_equally(result) -> None:
    """If the arms differ before the shift, the comparison is unfair."""
    learned = [r["task_a_learned"] for r in result["results"]]
    assert max(learned) - min(learned) < 0.05


def test_somnus_recovers_after_re_entry(result) -> None:
    arms = _by_name(result)
    assert arms["somnus"]["task_a_recovered"] < arms["somnus"]["task_a_reentry"]


def test_result_is_stable_across_seeds() -> None:
    margins = []
    for seed in (1, 2, 3, 4, 5):
        arms = _by_name(run_benchmark(seed=seed, verbose=False))
        margins.append(arms["control"]["forgetting"] - arms["somnus"]["forgetting"])
    assert all(m > 0.15 for m in margins), f"unstable across seeds: {margins}"

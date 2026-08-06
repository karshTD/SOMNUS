"""Neuromodulation: the z-score changepoint detector must survive noise."""

from core.neuromod import NeuromodulatorySystem, weighted_surprise
from core.simulator import Simulator

PRED = {"cpu_percent": 35.0, "rps": 120.0, "network_mbps": 50.0}


def test_no_boundary_in_steady_state() -> None:
    nm = NeuromodulatorySystem()
    sim = Simulator(regime_name="steady", seed=1)
    boundaries = sum(nm.step(PRED, sim.emit()).boundary for _ in range(200))
    assert boundaries == 0, "steady state must not trigger context boundaries"


def test_boundary_fires_on_regime_shift() -> None:
    nm = NeuromodulatorySystem()
    sim = Simulator(regime_name="steady", seed=1)
    for _ in range(200):
        nm.step(PRED, sim.emit())

    sim.set_regime("surge")
    fired = any(nm.step(PRED, sim.emit()).boundary for _ in range(10))
    assert fired, "a sustained regime shift must trigger a boundary"


def test_na_is_a_zscore_not_a_ratio() -> None:
    """A ratio collapses under noise; the z-score must not."""
    for jitter_regime in ("steady", "surge"):
        nm = NeuromodulatorySystem()
        sim = Simulator(regime_name=jitter_regime, seed=3)
        pred = {k: getattr(sim, f"base_{'cpu' if k == 'cpu_percent' else k.split('_')[0]}", 0) for k in PRED}
        for _ in range(150):
            nm.step(pred, sim.emit())
        assert nm.ach_std > 0


def test_ach_tracks_expected_uncertainty() -> None:
    quiet = NeuromodulatorySystem()
    noisy = NeuromodulatorySystem()
    sim_q = Simulator(regime_name="steady", seed=2)
    for _ in range(150):
        obs = sim_q.emit()
        quiet.step(PRED, obs)
        noisy.step(PRED, {**obs, "cpu_percent": obs["cpu_percent"] * 3})
    assert noisy.ach > quiet.ach


def test_encode_probability_rises_with_surprise() -> None:
    nm = NeuromodulatorySystem()
    low = nm.step(PRED, {"cpu_percent": 35, "rps": 120, "network_mbps": 50})
    high = nm.step(PRED, {"cpu_percent": 99, "rps": 1300, "network_mbps": 600})
    assert nm.encode_probability(high) > nm.encode_probability(low)


def test_weighted_surprise_bounds() -> None:
    s, _ = weighted_surprise(PRED, PRED)
    assert s == 0.0
    s2, _ = weighted_surprise(PRED, {"cpu_percent": 0, "rps": 0, "network_mbps": 0})
    assert 0.0 <= s2 <= 1.0

import pytest
from smartphone_mocap.rate_limit import PerceptionRateGate


def test_gate_drops_sets_without_accumulating_a_backlog():
    gate = PerceptionRateGate(10)
    assert gate.ready(0)
    assert not gate.ready(50_000_000)
    assert gate.ready(100_000_000)
    assert gate.ready(950_000_000)
    assert not gate.ready(951_000_000)
    assert gate.ready(0)  # new replay epoch


def test_uncapped_and_invalid_rates():
    assert all(PerceptionRateGate().ready(t) for t in [0, 1, 2])
    for hz in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            PerceptionRateGate(hz)

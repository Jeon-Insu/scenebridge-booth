"""Optional scheduling of fresh synchronized observations, not pose deadbands."""
from __future__ import annotations

import math


class PerceptionRateGate:
    """0 Hz means uncapped. Call with monotonic time (or replay media time).

    Callers discard skipped input sets, not queue them or relabel old results.
    A backwards clock starts a new epoch, as when a recording is replayed.
    """

    def __init__(self, hz: float = 0.0):
        hz = float(hz)
        if not math.isfinite(hz) or hz < 0:
            raise ValueError("perception_max_hz must be finite and nonnegative")
        self.interval_ns = round(1e9 / hz) if hz else 0
        self.last_ns: int | None = None

    def ready(self, now_ns: int) -> bool:
        if (self.last_ns is None or now_ns < self.last_ns
                or now_ns - self.last_ns >= self.interval_ns):
            self.last_ns = now_ns
            return True
        return False

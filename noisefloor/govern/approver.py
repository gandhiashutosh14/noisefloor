"""A scripted on-call engineer who approves or refuses failover requests.

It reads the clean error-rate channel (a human looking at the real dashboard) and approves only when
errors have been above 2% for the last 2 steps. The decision arrives one step after the request, so
an agent that asks for failover keeps operating on its best reversible action in the meantime.

The pre-registered approver (v0.1) looks at errors alone, so it also approves a failover during a
flash crowd, where the region is healthy and a failover cannot help but still costs 20. With
``require_degraded_health=True`` it also requires the region's health to be below 1 (the operator
checks the region status page before approving). Both variants are kept: the first is the
registered environment, the second is used in the post-hoc analysis.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np


class Approver:
    def __init__(self, E: int, threshold: float = 0.02, consecutive: int = 2, require_degraded_health: bool = False):
        self.threshold = threshold
        self.consecutive = consecutive
        self.require_degraded_health = require_degraded_health
        self.history: List[Deque[float]] = [deque(maxlen=consecutive) for _ in range(E)]
        self.health = np.ones(E)
        self.pending = np.zeros(E, dtype=bool)
        self.requests = np.zeros(E, dtype=np.int64)
        self.approvals = np.zeros(E, dtype=np.int64)

    def observe(self, error_rate: np.ndarray, health: Optional[np.ndarray] = None) -> None:
        for i, e in enumerate(error_rate):
            self.history[i].append(float(e))
        if health is not None:
            self.health = np.asarray(health, dtype=float)

    def request(self, i: int) -> None:
        if not self.pending[i]:
            self.pending[i] = True
            self.requests[i] += 1

    def decide(self) -> Dict[int, bool]:
        """Decisions for last step's requests: {env: approved}."""
        out: Dict[int, bool] = {}
        for i in np.flatnonzero(self.pending):
            h = self.history[i]
            ok = len(h) == self.consecutive and all(e > self.threshold for e in h)
            if self.require_degraded_health:
                ok = ok and bool(self.health[i] < 1.0)
            out[int(i)] = ok
            self.approvals[i] += int(ok)
            self.pending[i] = False
        return out

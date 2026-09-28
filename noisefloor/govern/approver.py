"""A scripted on-call engineer who approves or refuses failover requests.

It reads the clean error-rate channel (a human looking at the real dashboard) and approves only when
errors have been above 2% for the last 2 steps. The decision arrives one step after the request, so
an agent that asks for failover keeps operating on its best reversible action in the meantime.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

import numpy as np


class Approver:
    def __init__(self, E: int, threshold: float = 0.02, consecutive: int = 2):
        self.threshold = threshold
        self.consecutive = consecutive
        self.history: List[Deque[float]] = [deque(maxlen=consecutive) for _ in range(E)]
        self.pending = np.zeros(E, dtype=bool)
        self.requests = np.zeros(E, dtype=np.int64)
        self.approvals = np.zeros(E, dtype=np.int64)

    def observe(self, error_rate: np.ndarray) -> None:
        for i, e in enumerate(error_rate):
            self.history[i].append(float(e))

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
            out[int(i)] = ok
            self.approvals[i] += int(ok)
            self.pending[i] = False
        return out

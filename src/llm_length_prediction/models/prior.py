from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same size")
    return sum(a * b for a, b in zip(left, right, strict=True))


def lognormal_mean(mu: float, variance: float) -> float:
    if variance < 0:
        raise ValueError("variance must be non-negative")
    return math.exp(mu + 0.5 * variance)


@dataclass(frozen=True)
class LinearLogNormalPrior:
    """ALPS-style linear probe with a log-normal output prior."""

    weights: tuple[float, ...]
    bias: float
    log_variance: float

    def predict_mu(self, hidden_state: Sequence[float]) -> float:
        return dot(self.weights, hidden_state) + self.bias

    def predict_mean_length(self, hidden_state: Sequence[float]) -> float:
        return lognormal_mean(self.predict_mu(hidden_state), self.log_variance)

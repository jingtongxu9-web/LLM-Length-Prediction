from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicFeatures:
    prior_length: float
    step: int
    entropy: float
    entropy_mean: float
    entropy_slope: float
    eos_probability: float


def scheduled_gamma(step: int, midpoint: float = 64.0, scale: float = 24.0) -> float:
    """Weight that gradually transfers trust from the prior to decode evidence."""

    if step < 0 or scale <= 0:
        raise ValueError("step must be non-negative and scale must be positive")
    return 1.0 / (1.0 + math.exp(-(step - midpoint) / scale))


def hybrid_total_length(
    prior_length: float,
    step: int,
    predicted_remaining: float,
    gamma: float,
) -> float:
    if prior_length < 0 or step < 0 or predicted_remaining < 0:
        raise ValueError("length values must be non-negative")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    dynamic_total = step + predicted_remaining
    return (1.0 - gamma) * prior_length + gamma * dynamic_total

from __future__ import annotations

import math
from collections.abc import Sequence


def _paired(actual: Sequence[float], predicted: Sequence[float]) -> list[tuple[float, float]]:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same size")
    if not actual:
        raise ValueError("at least one observation is required")
    return list(zip(actual, predicted, strict=True))


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    pairs = _paired(actual, predicted)
    return sum(abs(a - p) for a, p in pairs) / len(pairs)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    pairs = _paired(actual, predicted)
    return math.sqrt(sum((a - p) ** 2 for a, p in pairs) / len(pairs))


def severe_underestimation_rate(
    actual: Sequence[float], predicted: Sequence[float], threshold: float = 100.0
) -> float:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    pairs = _paired(actual, predicted)
    severe = sum((a - p) > threshold for a, p in pairs)
    return severe / len(pairs)

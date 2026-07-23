from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


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


def log1p_prior_metrics(
    actual: Sequence[int],
    predicted_mean: Sequence[float],
    predicted_mu: Sequence[float],
    residual_variance: float,
) -> dict[str, int | float]:
    """Evaluate the shifted-log-normal prior in token and transformed spaces."""

    _paired(actual, predicted_mean)
    _paired(actual, predicted_mu)
    if residual_variance < 0:
        raise ValueError("residual_variance must be non-negative")

    target = np.log1p(np.asarray(actual, dtype=np.float64))
    mu = np.asarray(predicted_mu, dtype=np.float64)
    denominator = float(np.square(target - target.mean()).sum())
    r_squared = (
        0.0 if denominator == 0.0 else 1.0 - float(np.square(target - mu).sum()) / denominator
    )
    safe_variance = max(residual_variance, 1e-12)
    negative_log_likelihood = float(
        np.mean(
            0.5 * math.log(2.0 * math.pi * safe_variance)
            + np.square(target - mu) / (2.0 * safe_variance)
            + target
        )
    )
    radius = 1.959963984540054 * math.sqrt(safe_variance)
    lower = np.maximum(0.0, np.expm1(mu - radius))
    upper = np.expm1(mu + radius)
    actual_array = np.asarray(actual)
    coverage = float(np.mean((actual_array >= lower) & (actual_array <= upper)))
    return {
        "count": len(actual),
        "mae_tokens": mae(actual, predicted_mean),
        "rmse_tokens": rmse(actual, predicted_mean),
        "r_squared_log1p": r_squared,
        "negative_log_likelihood": negative_log_likelihood,
        "interval_95_coverage": coverage,
        "residual_variance_mle": residual_variance,
    }

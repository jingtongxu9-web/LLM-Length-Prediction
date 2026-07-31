from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from llm_length_prediction.evaluation.metrics import log1p_prior_metrics
from llm_length_prediction.models.dynamic import ProgressiveSample

PROGRESS_BINS = (
    ("0-10%", 0.0, 0.10),
    ("10-25%", 0.10, 0.25),
    ("25-50%", 0.25, 0.50),
    ("50-75%", 0.50, 0.75),
    ("75-100%", 0.75, 1.0),
)


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.square(actual - actual.mean()).sum())
    if denominator == 0.0:
        return 0.0
    return 1.0 - float(np.square(actual - predicted).sum()) / denominator


def progressive_metrics(
    samples: Sequence[ProgressiveSample],
    predicted_remaining: Sequence[float],
    predicted_mu: Sequence[float],
    residual_variance: float,
) -> dict[str, int | float]:
    if len(samples) != len(predicted_remaining) or len(samples) != len(predicted_mu):
        raise ValueError("samples and predictions must have the same size")
    if not samples:
        raise ValueError("at least one progressive sample is required")
    actual = [sample.remaining_tokens for sample in samples]
    metrics = log1p_prior_metrics(
        actual,
        predicted_remaining,
        predicted_mu,
        residual_variance,
    )
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted_remaining, dtype=np.float64)
    trace_keys = [(sample.prompt_id, sample.seed) for sample in samples]
    points_per_trace = Counter(trace_keys)
    sequence_weights = np.asarray(
        [1.0 / points_per_trace[key] for key in trace_keys],
        dtype=np.float64,
    )
    sequence_weights /= sequence_weights.sum()
    errors = predicted_array - actual_array
    mu_array = np.asarray(predicted_mu, dtype=np.float64)
    weighted_actual_mean = float(np.sum(sequence_weights * actual_array))
    weighted_denominator = float(
        np.sum(sequence_weights * np.square(actual_array - weighted_actual_mean))
    )
    weighted_r_squared = (
        0.0
        if weighted_denominator == 0.0
        else 1.0
        - float(np.sum(sequence_weights * np.square(errors))) / weighted_denominator
    )
    safe_variance = max(residual_variance, 1e-12)
    log_actual = np.log1p(actual_array)
    per_point_nll = (
        0.5 * math.log(2.0 * math.pi * safe_variance)
        + np.square(log_actual - mu_array) / (2.0 * safe_variance)
        + log_actual
    )
    radius = 1.959963984540054 * math.sqrt(safe_variance)
    lower = np.maximum(0.0, np.expm1(mu_array - radius))
    upper = np.expm1(mu_array + radius)
    covered = (actual_array >= lower) & (actual_array <= upper)
    return {
        **metrics,
        "r_squared_tokens": _r_squared(actual_array, predicted_array),
        "mean_error_tokens": float(np.mean(errors)),
        "trace_count": len(points_per_trace),
        "sequence_balanced_mae_tokens": float(
            np.sum(sequence_weights * np.abs(errors))
        ),
        "sequence_balanced_rmse_tokens": float(
            np.sqrt(np.sum(sequence_weights * np.square(errors)))
        ),
        "sequence_balanced_mean_error_tokens": float(
            np.sum(sequence_weights * errors)
        ),
        "sequence_balanced_r_squared_tokens": weighted_r_squared,
        "sequence_balanced_negative_log_likelihood": float(
            np.sum(sequence_weights * per_point_nll)
        ),
        "sequence_balanced_interval_95_coverage": float(
            np.sum(sequence_weights * covered)
        ),
    }


def progress_breakdown(
    samples: Sequence[ProgressiveSample],
    predicted_remaining: Sequence[float],
    predicted_mu: Sequence[float],
    residual_variance: float,
) -> list[dict[str, int | float | str]]:
    breakdown = []
    for label, lower, upper in PROGRESS_BINS:
        indices = [
            index
            for index, sample in enumerate(samples)
            if lower <= sample.step / sample.output_tokens
            and (
                sample.step / sample.output_tokens < upper
                or (upper == 1.0 and sample.step / sample.output_tokens <= upper)
            )
        ]
        if not indices:
            continue
        subset = [samples[index] for index in indices]
        metrics = progressive_metrics(
            subset,
            [predicted_remaining[index] for index in indices],
            [predicted_mu[index] for index in indices],
            residual_variance,
        )
        breakdown.append(
            {
                "decode_progress": label,
                "lower_fraction": lower,
                "upper_fraction": upper,
                **metrics,
            }
        )
    return breakdown

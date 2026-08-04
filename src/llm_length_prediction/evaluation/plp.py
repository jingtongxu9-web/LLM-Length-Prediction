from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from llm_length_prediction.evaluation.progressive import PROGRESS_BINS
from llm_length_prediction.models.plp import HiddenStatePLPSample


def _r_squared(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.sum(weights * actual) / np.sum(weights))
    denominator = float(np.sum(weights * np.square(actual - mean)))
    if denominator == 0.0:
        return 0.0
    return 1.0 - float(np.sum(weights * np.square(actual - predicted))) / denominator


def hidden_state_plp_metrics(
    samples: Sequence[HiddenStatePLPSample], predictions: Sequence[float]
) -> dict[str, int | float]:
    if not samples or len(samples) != len(predictions):
        raise ValueError("samples and predictions must be non-empty and aligned")
    actual = np.asarray([sample.remaining_tokens for sample in samples], dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if np.any(~np.isfinite(predicted)):
        raise ValueError("predictions must be finite")
    errors = predicted - actual
    ordinary = np.full(len(samples), 1.0 / len(samples), dtype=np.float64)
    trace_keys = [(sample.prompt_id, sample.seed) for sample in samples]
    counts = Counter(trace_keys)
    balanced = np.asarray([1.0 / counts[key] for key in trace_keys], dtype=np.float64)
    balanced /= balanced.sum()

    def summarize(weights: np.ndarray, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}mae_tokens": float(np.sum(weights * np.abs(errors))),
            f"{prefix}rmse_tokens": float(np.sqrt(np.sum(weights * np.square(errors)))),
            f"{prefix}mean_error_tokens": float(np.sum(weights * errors)),
            f"{prefix}r_squared_tokens": _r_squared(actual, predicted, weights),
        }

    return {
        "count": len(samples),
        "trace_count": len(counts),
        **summarize(ordinary, ""),
        **summarize(balanced, "sequence_balanced_"),
    }


def hidden_state_plp_progress_breakdown(
    samples: Sequence[HiddenStatePLPSample], predictions: Sequence[float]
) -> list[dict[str, int | float | str]]:
    rows = []
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
        rows.append(
            {
                "decode_progress": label,
                "lower_fraction": lower,
                "upper_fraction": upper,
                **hidden_state_plp_metrics(
                    subset, [predictions[index] for index in indices]
                ),
            }
        )
    return rows


def hidden_state_plp_group_breakdown(
    samples: Sequence[HiddenStatePLPSample],
    predictions: Sequence[float],
    *,
    group_by: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Evaluate point predictions by pre-declared metadata without using it as input."""

    allowed = {"task", "intended_length", "seed"}
    if not group_by or not set(group_by).issubset(allowed):
        raise ValueError(f"group_by must use one or more of {sorted(allowed)}")
    if not samples or len(samples) != len(predictions):
        raise ValueError("samples and predictions must be non-empty and aligned")

    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, sample in enumerate(samples):
        key = tuple(getattr(sample, attribute) for attribute in group_by)
        groups.setdefault(key, []).append(index)

    rows = []
    for key in sorted(groups, key=lambda values: tuple(str(value) for value in values)):
        indices = groups[key]
        subset = [samples[index] for index in indices]
        rows.append(
            {
                **dict(zip(group_by, key, strict=True)),
                **hidden_state_plp_metrics(
                    subset, [predictions[index] for index in indices]
                ),
            }
        )
    return rows

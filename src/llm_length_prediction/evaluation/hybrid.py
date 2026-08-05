"""Family-grouped evaluation and uncertainty estimates for Hybrid v3."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from llm_length_prediction.models.hybrid import HybridSample


def task_stratified_family_folds(
    samples: Sequence[HybridSample], *, folds: int, seed: int
) -> dict[str, int]:
    """Assign whole families to folds while balancing each task stratum."""

    if folds < 2:
        raise ValueError("folds must be at least two")
    family_tasks: dict[str, str] = {}
    for sample in samples:
        previous = family_tasks.setdefault(sample.prompt_family_id, sample.task)
        if previous != sample.task:
            raise ValueError("one prompt family cannot span multiple tasks")
    by_task: dict[str, list[str]] = defaultdict(list)
    for family, task in family_tasks.items():
        by_task[task].append(family)
    rng = np.random.default_rng(seed)
    assignments: dict[str, int] = {}
    for task in sorted(by_task):
        families = sorted(by_task[task])
        rng.shuffle(families)
        for index, family in enumerate(families):
            assignments[family] = index % folds
    if any(sum(value == fold for value in assignments.values()) == 0 for fold in range(folds)):
        raise ValueError("every fold must contain at least one family")
    return assignments


def sequence_balanced_metrics(
    samples: Sequence[HybridSample], predictions: Sequence[float]
) -> dict[str, float | int]:
    if not samples or len(samples) != len(predictions):
        raise ValueError("samples and predictions must be non-empty and aligned")
    actual = np.asarray([sample.remaining_tokens for sample in samples], dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if np.any(~np.isfinite(predicted)):
        raise ValueError("predictions must be finite")
    weights = np.asarray([sample.sequence_weight for sample in samples], dtype=np.float64)
    weights /= weights.sum()
    error = predicted - actual
    under = np.maximum(actual - predicted, 0.0)
    positive_under = under[under > 0]
    actual_mean = float(np.sum(weights * actual))
    denominator = float(np.sum(weights * (actual - actual_mean) ** 2))
    top_threshold = float(np.quantile(actual, 0.9))
    top_mask = actual >= top_threshold
    top_weights = weights[top_mask] / weights[top_mask].sum()
    return {
        "point_count": len(samples),
        "trace_count": len({sample.trace_key for sample in samples}),
        "sequence_balanced_mae_tokens": float(np.sum(weights * np.abs(error))),
        "sequence_balanced_rmse_tokens": float(np.sqrt(np.sum(weights * np.square(error)))),
        "sequence_balanced_mean_error_tokens": float(np.sum(weights * error)),
        "sequence_balanced_r_squared_tokens": (
            0.0 if denominator == 0 else 1.0 - float(np.sum(weights * error**2)) / denominator
        ),
        "underprediction_rate": float(np.sum(weights * (error < 0))),
        "p95_positive_underprediction_tokens": (
            float(np.quantile(positive_under, 0.95)) if len(positive_under) else 0.0
        ),
        "top_10_percent_remaining_mae_tokens": float(np.sum(top_weights * np.abs(error[top_mask]))),
    }


def family_metric_rows(
    samples: Sequence[HybridSample], predictions: Sequence[float]
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample.prompt_family_id].append(index)
    rows = []
    for family in sorted(groups):
        indices = groups[family]
        subset = [samples[index] for index in indices]
        rows.append(
            {
                "prompt_family_id": family,
                "task": subset[0].task,
                **sequence_balanced_metrics(subset, [predictions[index] for index in indices]),
            }
        )
    return rows


def family_macro_metrics(
    samples: Sequence[HybridSample], predictions: Sequence[float]
) -> dict[str, Any]:
    rows = family_metric_rows(samples, predictions)
    metric_names = (
        "sequence_balanced_mae_tokens",
        "sequence_balanced_rmse_tokens",
        "sequence_balanced_mean_error_tokens",
        "underprediction_rate",
        "p95_positive_underprediction_tokens",
        "top_10_percent_remaining_mae_tokens",
    )
    return {
        **sequence_balanced_metrics(samples, predictions),
        "family_count": len(rows),
        **{
            f"family_macro_{name}": float(np.mean([row[name] for row in rows]))
            for name in metric_names
        },
    }


def family_bootstrap_interval(
    values: Mapping[str, float], *, replicates: int, confidence: float, seed: int
) -> dict[str, float | int]:
    if not values or replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap inputs")
    families = sorted(values)
    vector = np.asarray([values[family] for family in families], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(vector), size=(replicates, len(vector)))
    statistics = vector[draws].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "estimate": float(vector.mean()),
        "lower": float(np.quantile(statistics, alpha / 2.0)),
        "upper": float(np.quantile(statistics, 1.0 - alpha / 2.0)),
        "confidence_level": confidence,
        "replicates": replicates,
        "family_count": len(vector),
    }


def paired_family_mae_difference(
    samples: Sequence[HybridSample],
    first: Sequence[float],
    second: Sequence[float],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    first_rows = {
        row["prompt_family_id"]: float(row["sequence_balanced_mae_tokens"])
        for row in family_metric_rows(samples, first)
    }
    second_rows = {
        row["prompt_family_id"]: float(row["sequence_balanced_mae_tokens"])
        for row in family_metric_rows(samples, second)
    }
    if first_rows.keys() != second_rows.keys():
        raise ValueError("paired methods do not cover identical families")
    differences = {family: first_rows[family] - second_rows[family] for family in first_rows}
    return family_bootstrap_interval(
        differences, replicates=replicates, confidence=confidence, seed=seed
    )


def absolute_step_breakdown(
    samples: Sequence[HybridSample],
    predictions: Sequence[float],
    *,
    boundaries: Sequence[int],
) -> list[dict[str, Any]]:
    if list(boundaries) != sorted(set(boundaries)) or any(value <= 0 for value in boundaries):
        raise ValueError("absolute-step boundaries must be unique positive sorted integers")
    rows = []
    lower = 1
    for upper in [*boundaries, None]:
        indices = [
            index
            for index, sample in enumerate(samples)
            if sample.step >= lower and (upper is None or sample.step <= upper)
        ]
        if indices:
            rows.append(
                {
                    "step_range": f"{lower}+" if upper is None else f"{lower}-{upper}",
                    **sequence_balanced_metrics(
                        [samples[index] for index in indices],
                        [predictions[index] for index in indices],
                    ),
                }
            )
        if upper is not None:
            lower = upper + 1
    return rows

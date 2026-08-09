"""Diagnostics for progress-gated residual length prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from llm_length_prediction.evaluation.hybrid import sequence_balanced_metrics
from llm_length_prediction.models.hybrid import HybridSample


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_quantiles(
    values: Sequence[float], weights: Sequence[float], quantiles: Sequence[float]
) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    requested = np.asarray(quantiles, dtype=np.float64)
    if (
        vector.ndim != 1
        or weight.shape != vector.shape
        or not len(vector)
        or np.any(~np.isfinite(vector))
        or np.any(weight < 0)
        or weight.sum() <= 0
        or np.any((requested < 0) | (requested > 1))
    ):
        raise ValueError("invalid weighted-quantile inputs")
    order = np.argsort(vector)
    ordered = vector[order]
    cumulative = np.cumsum(weight[order])
    cumulative = (cumulative - 0.5 * weight[order]) / weight.sum()
    result = np.interp(requested, cumulative, ordered)
    return {
        f"p{int(round(value * 100)):02d}": float(result[index])
        for index, value in enumerate(requested)
    }


def _weighted_pearson(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> float:
    first_mean = _weighted_mean(first, weights)
    second_mean = _weighted_mean(second, weights)
    first_centered = first - first_mean
    second_centered = second - second_mean
    denominator = np.sqrt(
        np.sum(weights * first_centered**2) * np.sum(weights * second_centered**2)
    )
    return (
        0.0
        if denominator == 0
        else float(np.sum(weights * first_centered * second_centered) / denominator)
    )


def _method_metrics(
    samples: Sequence[HybridSample],
    indices: Sequence[int],
    predictions: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    subset = [samples[index] for index in indices]
    return {
        method: sequence_balanced_metrics(subset, values[np.asarray(indices, dtype=int)])
        for method, values in predictions.items()
    }


def _subset_report(
    samples: Sequence[HybridSample],
    indices: Sequence[int],
    *,
    predictions: Mapping[str, np.ndarray],
    candidate_id: str,
    alps_id: str,
    concat_id: str,
    scalar_id: str,
    actual: np.ndarray,
    weights: np.ndarray,
    needed: np.ndarray,
    applied: np.ndarray,
    gate: np.ndarray,
    gate_confidence: np.ndarray,
    bounded: np.ndarray,
    bounds: np.ndarray,
    saturation_ratio: float,
    direction_tolerance: float,
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=int)
    subset_weights = weights[selected]
    candidate_error = np.abs(predictions[candidate_id][selected] - actual[selected])
    alps_error = np.abs(predictions[alps_id][selected] - actual[selected])
    concat_error = np.abs(predictions[concat_id][selected] - actual[selected])
    direction_mask = np.abs(needed[selected]) > direction_tolerance
    report: dict[str, Any] = {
        "point_count": len(selected),
        "trace_count": len({samples[index].trace_key for index in selected}),
        "sequence_balanced_mean_gate": _weighted_mean(gate[selected], subset_weights),
        "sequence_balanced_mean_learned_gate_confidence": _weighted_mean(
            gate_confidence[selected], subset_weights
        ),
        "sequence_balanced_mean_applied_correction_tokens": _weighted_mean(
            applied[selected], subset_weights
        ),
        "sequence_balanced_mean_absolute_applied_correction_tokens": _weighted_mean(
            np.abs(applied[selected]), subset_weights
        ),
        "sequence_balanced_mean_needed_correction_tokens": _weighted_mean(
            needed[selected], subset_weights
        ),
        "sequence_balanced_mean_absolute_needed_correction_tokens": _weighted_mean(
            np.abs(needed[selected]), subset_weights
        ),
        "sequence_balanced_correction_success_rate": _weighted_mean(
            (candidate_error < alps_error).astype(np.float64), subset_weights
        ),
        "sequence_balanced_mean_mae_improvement_over_alps_tokens": _weighted_mean(
            alps_error - candidate_error, subset_weights
        ),
        "sequence_balanced_mean_mae_improvement_over_concat_v1_tokens": _weighted_mean(
            concat_error - candidate_error, subset_weights
        ),
        "sequence_balanced_bound_saturation_rate": _weighted_mean(
            (np.abs(bounded[selected]) >= saturation_ratio * bounds[selected]).astype(np.float64),
            subset_weights,
        ),
        "methods": _method_metrics(
            samples,
            selected,
            {
                alps_id: predictions[alps_id],
                concat_id: predictions[concat_id],
                scalar_id: predictions[scalar_id],
                candidate_id: predictions[candidate_id],
            },
        ),
    }
    if np.any(direction_mask):
        direction_weights = subset_weights[direction_mask]
        report["sequence_balanced_correction_direction_agreement_rate"] = _weighted_mean(
            (
                np.sign(applied[selected][direction_mask])
                == np.sign(needed[selected][direction_mask])
            ).astype(np.float64),
            direction_weights,
        )
    else:
        report["sequence_balanced_correction_direction_agreement_rate"] = None
    return report


def _binned_reports(
    values: np.ndarray,
    boundaries: Sequence[float],
    labels: Sequence[str],
    build: Any,
) -> list[dict[str, Any]]:
    if len(boundaries) != len(labels) + 1 or list(boundaries) != sorted(boundaries):
        raise ValueError("bin boundaries and labels do not align")
    rows = []
    for index, label in enumerate(labels):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (values >= lower) & (
            (values <= upper) if index == len(labels) - 1 else (values < upper)
        )
        indices = np.flatnonzero(mask).tolist()
        if indices:
            rows.append({"group": label, "lower": lower, "upper": upper, **build(indices)})
    return rows


def gated_residual_diagnostics(
    samples: Sequence[HybridSample],
    predictions: Mapping[str, np.ndarray],
    *,
    candidate_id: str,
    alps_id: str,
    concat_id: str,
    scalar_id: str,
    applied: np.ndarray,
    gate: np.ndarray,
    gate_confidence: np.ndarray,
    bounded: np.ndarray,
    terminal_probability: np.ndarray,
    progress: np.ndarray,
    bounds: np.ndarray,
    family_folds: Mapping[str, int],
    settings: Mapping[str, Any],
    terminal_threshold: float,
) -> dict[str, Any]:
    """Explain when the gate opens and whether opening improves predictions."""

    count = len(samples)
    arrays = [
        *predictions.values(),
        applied,
        gate,
        gate_confidence,
        bounded,
        terminal_probability,
        progress,
        bounds,
    ]
    if not count or any(np.asarray(value).shape != (count,) for value in arrays):
        raise ValueError("gated diagnostics inputs must be non-empty and aligned")
    if any(np.any(~np.isfinite(value)) for value in arrays):
        raise ValueError("gated diagnostics inputs must be finite")
    actual = np.asarray([sample.remaining_tokens for sample in samples], dtype=np.float64)
    weights = np.asarray([sample.sequence_weight for sample in samples], dtype=np.float64)
    alps = predictions[alps_id]
    needed = actual - alps
    saturation_ratio = float(settings["bound_saturation_ratio"])
    direction_tolerance = float(settings["direction_tolerance_tokens"])

    def build(indices: Sequence[int]) -> dict[str, Any]:
        return _subset_report(
            samples,
            indices,
            predictions=predictions,
            candidate_id=candidate_id,
            alps_id=alps_id,
            concat_id=concat_id,
            scalar_id=scalar_id,
            actual=actual,
            weights=weights,
            needed=needed,
            applied=applied,
            gate=gate,
            gate_confidence=gate_confidence,
            bounded=bounded,
            bounds=bounds,
            saturation_ratio=saturation_ratio,
            direction_tolerance=direction_tolerance,
        )

    all_indices = list(range(count))
    overall = build(all_indices)
    thresholds = [float(value) for value in settings["gate_thresholds"]]
    overall["gate_quantiles"] = weighted_quantiles(
        gate, weights, [float(value) for value in settings["gate_quantiles"]]
    )
    overall["learned_gate_confidence_quantiles"] = weighted_quantiles(
        gate_confidence,
        weights,
        [float(value) for value in settings["gate_quantiles"]],
    )
    overall["sequence_balanced_gate_rates"] = {
        f"at_most_{threshold:g}": _weighted_mean((gate <= threshold).astype(float), weights)
        for threshold in thresholds
    } | {
        f"at_least_{threshold:g}": _weighted_mean((gate >= threshold).astype(float), weights)
        for threshold in thresholds
    }
    overall["sequence_balanced_learned_gate_confidence_rates"] = {
        f"at_most_{threshold:g}": _weighted_mean(
            (gate_confidence <= threshold).astype(float), weights
        )
        for threshold in thresholds
    } | {
        f"at_least_{threshold:g}": _weighted_mean(
            (gate_confidence >= threshold).astype(float), weights
        )
        for threshold in thresholds
    }
    improvement = np.abs(alps - actual) - np.abs(predictions[candidate_id] - actual)
    overall["weighted_pearson_gate_vs_mae_improvement_over_alps"] = _weighted_pearson(
        gate, improvement, weights
    )
    overall[
        "weighted_pearson_learned_gate_confidence_vs_mae_improvement_over_alps"
    ] = _weighted_pearson(gate_confidence, improvement, weights)
    overall["weighted_pearson_absolute_correction_vs_absolute_need"] = _weighted_pearson(
        np.abs(applied), np.abs(needed), weights
    )

    terminal_actual = actual == 0
    terminal_predicted = terminal_probability >= terminal_threshold
    tp = int(np.sum(terminal_actual & terminal_predicted))
    fp = int(np.sum(~terminal_actual & terminal_predicted))
    fn = int(np.sum(terminal_actual & ~terminal_predicted))
    tn = int(np.sum(~terminal_actual & ~terminal_predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    terminal = {
        "threshold": terminal_threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
        "sequence_balanced_predicted_terminal_rate": _weighted_mean(
            terminal_predicted.astype(float), weights
        ),
        "sequence_balanced_actual_terminal_rate": _weighted_mean(
            terminal_actual.astype(float), weights
        ),
    }

    def grouped_rows(labels: Sequence[str]) -> list[dict[str, Any]]:
        return [
            {"group": label, **build([i for i, value in enumerate(labels) if value == label])}
            for label in sorted(set(labels))
        ]

    task_labels = [sample.task for sample in samples]
    length_labels = [sample.intended_length for sample in samples]
    task_length_labels = [f"{sample.task}__{sample.intended_length}" for sample in samples]
    fold_labels = [str(family_folds[sample.prompt_family_id]) for sample in samples]
    terminal_status_labels = [
        "terminal" if sample.remaining_tokens == 0 else "nonterminal" for sample in samples
    ]
    return {
        "overall": overall,
        "terminal_classification": terminal,
        "by_decode_progress": _binned_reports(
            progress,
            settings["decode_progress_boundaries"],
            settings["decode_progress_labels"],
            build,
        ),
        "by_gate_band": _binned_reports(
            gate,
            settings["gate_boundaries"],
            settings["gate_labels"],
            build,
        ),
        "by_task": grouped_rows(task_labels),
        "by_intended_length": grouped_rows(length_labels),
        "by_task_and_intended_length": grouped_rows(task_length_labels),
        "by_outer_fold": grouped_rows(fold_labels),
        "by_terminal_status": grouped_rows(terminal_status_labels),
    }

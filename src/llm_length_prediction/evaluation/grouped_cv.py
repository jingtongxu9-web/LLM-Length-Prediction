from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagnosticRow:
    prompt_id: str
    prompt_family_id: str
    task_type: str
    intended_length: str
    prompt_tokens: int
    output_tokens: int
    hidden_state: tuple[float, ...]


@dataclass(frozen=True)
class FittedLogModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float
    residual_variance: float

    def predict_mu(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_scale
        return self.bias + standardized @ self.weights

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        return np.maximum(
            0.0,
            np.expm1(self.predict_mu(features) + 0.5 * self.residual_variance),
        )


def grouped_folds(groups: Sequence[str], n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic folds with whole prompt families kept together."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    unique_groups = sorted(set(groups))
    if len(unique_groups) < n_splits:
        raise ValueError("n_splits cannot exceed the number of prompt families")

    counts = Counter(groups)
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for group in sorted(unique_groups, key=lambda item: (-counts[item], item)):
        fold_index = min(range(n_splits), key=lambda index: (fold_sizes[index], index))
        fold_groups[fold_index].append(group)
        fold_sizes[fold_index] += counts[group]

    group_array = np.asarray(groups, dtype=object)
    folds = []
    for held_out in fold_groups:
        validation = np.flatnonzero(np.isin(group_array, held_out))
        training = np.flatnonzero(~np.isin(group_array, held_out))
        folds.append((training, validation))
    return folds


def one_hot_schema(rows: Sequence[DiagnosticRow]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tasks = tuple(sorted({row.task_type for row in rows}))
    lengths = tuple(sorted({row.intended_length for row in rows}))
    return tasks, lengths


def feature_matrix(
    rows: Sequence[DiagnosticRow],
    model_name: str,
    *,
    schema: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
) -> np.ndarray:
    """Build leakage-safe features for one diagnostic baseline."""

    if model_name == "global_mean":
        return np.zeros((len(rows), 0), dtype=np.float64)
    if model_name == "prompt_tokens":
        return np.asarray([[row.prompt_tokens] for row in rows], dtype=np.float64)
    if model_name == "alps_hidden":
        return np.asarray([row.hidden_state for row in rows], dtype=np.float64)
    if model_name not in {"metadata", "metadata_prompt_tokens"}:
        raise ValueError(f"unknown diagnostic model: {model_name}")

    tasks, lengths = schema or one_hot_schema(rows)
    values: list[list[float]] = []
    for row in rows:
        encoded = [float(row.task_type == value) for value in tasks]
        encoded.extend(float(row.intended_length == value) for value in lengths)
        if model_name == "metadata_prompt_tokens":
            encoded.append(float(row.prompt_tokens))
        values.append(encoded)
    return np.asarray(values, dtype=np.float64)


def fit_log_ridge(features: np.ndarray, output_tokens: np.ndarray, alpha: float) -> FittedLogModel:
    """Fit a standardized Ridge model on log1p(output_tokens)."""

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if features.ndim != 2 or output_tokens.ndim != 1:
        raise ValueError("features must be 2-D and output_tokens must be 1-D")
    if len(features) != len(output_tokens) or len(features) == 0:
        raise ValueError("features and output_tokens must be non-empty and aligned")

    target = np.log1p(output_tokens.astype(np.float64))
    bias = float(target.mean())
    if features.shape[1] == 0:
        mean = np.empty(0, dtype=np.float64)
        scale = np.empty(0, dtype=np.float64)
        weights = np.empty(0, dtype=np.float64)
        fitted = np.full(len(target), bias, dtype=np.float64)
    else:
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale == 0.0] = 1.0
        standardized = (features - mean) / scale
        centered = target - bias
        observations, dimensions = standardized.shape
        if dimensions <= observations:
            system = standardized.T @ standardized + alpha * np.eye(dimensions)
            weights = np.linalg.solve(system, standardized.T @ centered)
        else:
            system = standardized @ standardized.T + alpha * np.eye(observations)
            weights = standardized.T @ np.linalg.solve(system, centered)
        fitted = bias + standardized @ weights
    variance = float(np.mean(np.square(target - fitted)))
    return FittedLogModel(mean, scale, weights, bias, variance)


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.square(actual - actual.mean()).sum())
    if denominator == 0.0:
        return 0.0
    return 1.0 - float(np.square(actual - predicted).sum()) / denominator


def point_metrics(actual: np.ndarray, predicted: np.ndarray, mu: np.ndarray) -> dict[str, float]:
    log_actual = np.log1p(actual)
    return {
        "mae_tokens": float(np.mean(np.abs(actual - predicted))),
        "rmse_tokens": float(np.sqrt(np.mean(np.square(actual - predicted)))),
        "r_squared_tokens": _r_squared(actual, predicted),
        "rmse_log1p": float(np.sqrt(np.mean(np.square(log_actual - mu)))),
        "r_squared_log1p": _r_squared(log_actual, mu),
    }


def prompt_mean_metrics(
    rows: Sequence[DiagnosticRow], predicted: np.ndarray, mu: np.ndarray
) -> dict[str, float]:
    by_prompt: dict[str, list[int]] = {}
    prediction_by_prompt: dict[str, list[float]] = {}
    mu_by_prompt: dict[str, list[float]] = {}
    for row, prediction, predicted_mu in zip(rows, predicted, mu, strict=True):
        by_prompt.setdefault(row.prompt_id, []).append(row.output_tokens)
        prediction_by_prompt.setdefault(row.prompt_id, []).append(float(prediction))
        mu_by_prompt.setdefault(row.prompt_id, []).append(float(predicted_mu))
    prompt_ids = sorted(by_prompt)
    actual_mean = np.asarray([np.mean(by_prompt[key]) for key in prompt_ids])
    predicted_mean = np.asarray([np.mean(prediction_by_prompt[key]) for key in prompt_ids])
    predicted_mu = np.asarray([np.mean(mu_by_prompt[key]) for key in prompt_ids])
    return {
        "prompt_count": len(prompt_ids),
        **point_metrics(actual_mean, predicted_mean, predicted_mu),
    }


def interval_metrics(
    actual: np.ndarray, mu: np.ndarray, variances: np.ndarray
) -> dict[str, float]:
    safe = np.maximum(variances, 1e-12)
    target = np.log1p(actual)
    radius = 1.959963984540054 * np.sqrt(safe)
    lower = np.maximum(0.0, np.expm1(mu - radius))
    upper = np.expm1(mu + radius)
    nll = np.mean(
        0.5 * np.log(2.0 * math.pi * safe)
        + np.square(target - mu) / (2.0 * safe)
        + target
    )
    return {
        "negative_log_likelihood": float(nll),
        "interval_95_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "interval_95_mean_width": float(np.mean(upper - lower)),
    }


def cross_validate(
    rows: Sequence[DiagnosticRow],
    *,
    model_name: str,
    alpha: float,
    n_splits: int = 5,
) -> dict[str, object]:
    """Produce family-grouped out-of-fold predictions and metrics."""

    if not rows:
        raise ValueError("rows cannot be empty")
    schema = one_hot_schema(rows)
    all_features = feature_matrix(rows, model_name, schema=schema)
    actual = np.asarray([row.output_tokens for row in rows], dtype=np.float64)
    groups = [row.prompt_family_id for row in rows]
    predictions = np.empty(len(rows), dtype=np.float64)
    mus = np.empty(len(rows), dtype=np.float64)
    variances = np.empty(len(rows), dtype=np.float64)
    fold_ids = np.empty(len(rows), dtype=np.int64)

    for fold_id, (train_indices, validation_indices) in enumerate(
        grouped_folds(groups, n_splits)
    ):
        fitted = fit_log_ridge(
            all_features[train_indices],
            actual[train_indices],
            alpha,
        )
        validation_features = all_features[validation_indices]
        mus[validation_indices] = fitted.predict_mu(validation_features)
        predictions[validation_indices] = fitted.predict_mean(validation_features)
        variances[validation_indices] = fitted.residual_variance
        fold_ids[validation_indices] = fold_id

    prediction_rows = [
        {
            "prompt_id": row.prompt_id,
            "prompt_family_id": row.prompt_family_id,
            "fold": int(fold_id),
            "actual_output_tokens": row.output_tokens,
            "predicted_log1p_mu": float(mu),
            "predicted_mean_output_tokens": float(prediction),
            "training_residual_variance": float(variance),
        }
        for row, fold_id, mu, prediction, variance in zip(
            rows, fold_ids, mus, predictions, variances, strict=True
        )
    ]
    return {
        "model": model_name,
        "alpha": alpha,
        "n_splits": n_splits,
        "family_count": len(set(groups)),
        "rollout_metrics": {
            **point_metrics(actual, predictions, mus),
            **interval_metrics(actual, mus, variances),
        },
        "prompt_mean_metrics": prompt_mean_metrics(rows, predictions, mus),
        "predictions": prediction_rows,
    }


def select_families(
    rows: Sequence[DiagnosticRow], fraction: float, *, repeat: int
) -> list[DiagnosticRow]:
    """Select a deterministic rotating subset for a family-level learning curve."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    families = sorted({row.prompt_family_id for row in rows})
    count = max(2, round(len(families) * fraction))
    offset = (repeat * max(1, len(families) // 5)) % len(families)
    rotated = families[offset:] + families[:offset]
    selected = set(rotated[:count])
    return [row for row in rows if row.prompt_family_id in selected]


def flatten_results(results: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for result in results:
        rollout = result["rollout_metrics"]
        prompt = result["prompt_mean_metrics"]
        assert isinstance(rollout, dict) and isinstance(prompt, dict)
        rows.append(
            {
                "model": result["model"],
                "alpha": result["alpha"],
                "n_splits": result["n_splits"],
                "family_count": result["family_count"],
                **{f"rollout_{key}": value for key, value in rollout.items()},
                **{f"prompt_mean_{key}": value for key, value in prompt.items()},
            }
        )
    return rows

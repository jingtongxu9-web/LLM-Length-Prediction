from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same size")
    return sum(a * b for a, b in zip(left, right, strict=True))


def shifted_lognormal_mean(mu: float, variance: float) -> float:
    """Return E[L] when log(1 + L) is Normal(mu, variance)."""

    if variance < 0:
        raise ValueError("variance must be non-negative")
    return max(0.0, math.expm1(mu + 0.5 * variance))


@dataclass(frozen=True)
class LinearLogNormalPrior:
    """ALPS probe where log(1 + output_tokens) follows a Normal distribution."""

    weights: tuple[float, ...]
    bias: float
    log_variance: float

    def predict_mu(self, hidden_state: Sequence[float]) -> float:
        return dot(self.weights, hidden_state) + self.bias

    def predict_mean_length(self, hidden_state: Sequence[float]) -> float:
        return shifted_lognormal_mean(self.predict_mu(hidden_state), self.log_variance)


@dataclass(frozen=True)
class StandardizedRidgeLogNormalPrior:
    """Fitted ALPS v1 prior with train-only scaling and residual variance."""

    weights: tuple[float, ...]
    bias: float
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    residual_variance: float
    ridge_alpha: float = 1.0
    target: str = "log1p_output_tokens"
    residual_variance_estimator: str = "maximum_likelihood"

    def predict_mu(self, hidden_state: Sequence[float]) -> float:
        if len(hidden_state) != len(self.weights):
            raise ValueError("hidden state has the wrong dimension")
        standardized = (
            (value - mean) / scale
            for value, mean, scale in zip(
                hidden_state, self.feature_mean, self.feature_scale, strict=True
            )
        )
        return dot(self.weights, tuple(standardized)) + self.bias

    def predict_mean_length(self, hidden_state: Sequence[float]) -> float:
        return shifted_lognormal_mean(self.predict_mu(hidden_state), self.residual_variance)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_type": "standardized_ridge_shifted_lognormal",
            "target": self.target,
            "weights": list(self.weights),
            "bias": self.bias,
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "residual_variance": self.residual_variance,
            "residual_variance_estimator": self.residual_variance_estimator,
            "ridge_alpha": self.ridge_alpha,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StandardizedRidgeLogNormalPrior:
        if payload.get("target") != "log1p_output_tokens":
            raise ValueError("prior target must be log1p_output_tokens")
        return cls(
            weights=tuple(float(value) for value in payload["weights"]),  # type: ignore[arg-type]
            bias=float(payload["bias"]),
            feature_mean=tuple(float(value) for value in payload["feature_mean"]),  # type: ignore[arg-type]
            feature_scale=tuple(float(value) for value in payload["feature_scale"]),  # type: ignore[arg-type]
            residual_variance=float(payload["residual_variance"]),
            ridge_alpha=float(payload.get("ridge_alpha", 1.0)),
            residual_variance_estimator=str(
                payload.get("residual_variance_estimator", "maximum_likelihood")
            ),
        )


def fit_log1p_ridge_prior(
    hidden_states: Sequence[Sequence[float]],
    output_tokens: Sequence[int],
    *,
    alpha: float = 1.0,
) -> StandardizedRidgeLogNormalPrior:
    """Fit Ridge on log1p(output_tokens) and estimate MLE residual variance."""

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    features = np.asarray(hidden_states, dtype=np.float64)
    lengths = np.asarray(output_tokens, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("hidden_states must be a non-empty two-dimensional matrix")
    if lengths.ndim != 1 or lengths.shape[0] != features.shape[0]:
        raise ValueError("output_tokens must contain one value per hidden state")
    if np.any(lengths < 0):
        raise ValueError("output token counts must be non-negative")

    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    standardized = (features - feature_mean) / feature_scale
    target = np.log1p(lengths)
    bias = float(target.mean())
    centered_target = target - bias
    observation_count, feature_count = standardized.shape
    if alpha == 0.0:
        weights = np.linalg.lstsq(standardized, centered_target, rcond=None)[0]
    elif feature_count <= observation_count:
        system = standardized.T @ standardized + alpha * np.eye(feature_count)
        weights = np.linalg.solve(system, standardized.T @ centered_target)
    else:
        # ALPS has many more hidden dimensions than pilot observations. The dual solve
        # avoids a slower feature_count x feature_count system while remaining exact.
        system = standardized @ standardized.T + alpha * np.eye(observation_count)
        weights = standardized.T @ np.linalg.solve(system, centered_target)
    residuals = target - (bias + standardized @ weights)
    residual_variance = float(np.mean(np.square(residuals)))

    return StandardizedRidgeLogNormalPrior(
        weights=tuple(float(value) for value in weights),
        bias=bias,
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        residual_variance=residual_variance,
        ridge_alpha=alpha,
    )


def grouped_fold_ids(
    groups: Sequence[Hashable],
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Assign every member of a family/group to one deterministic fold."""

    if folds < 2:
        raise ValueError("folds must be at least two")
    if len(groups) == 0:
        raise ValueError("at least one group is required")
    unique_groups = sorted(set(groups), key=str)
    if len(unique_groups) < folds:
        raise ValueError("the number of unique groups must be at least the fold count")
    generator = np.random.default_rng(seed)
    shuffled = list(unique_groups)
    generator.shuffle(shuffled)
    group_to_fold = {group: index % folds for index, group in enumerate(shuffled)}
    return np.asarray([group_to_fold[group] for group in groups], dtype=np.int32)


def fit_grouped_oof_log1p_prior(
    hidden_states: Sequence[Sequence[float]],
    output_tokens: Sequence[int],
    groups: Sequence[Hashable],
    *,
    folds: int = 5,
    alpha: float = 1.0,
    seed: int = 42,
    fold_ids: Sequence[int] | None = None,
) -> tuple[StandardizedRidgeLogNormalPrior, np.ndarray, np.ndarray]:
    """Fit the final Ridge while calibrating variance from grouped OOF residuals.

    Returns ``(full_train_prior, oof_mu, resolved_fold_ids)``. The fitted Ridge weights
    use all supplied training rows, while ``residual_variance`` is replaced by the MLE
    over predictions made without each row's family.
    """

    features = np.asarray(hidden_states, dtype=np.float64)
    lengths = np.asarray(output_tokens, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("hidden_states must be a non-empty matrix")
    if lengths.shape != (len(features),) or len(groups) != len(features):
        raise ValueError("lengths and groups must align with hidden_states")
    if np.any(~np.isfinite(features)) or np.any(~np.isfinite(lengths)):
        raise ValueError("training inputs must be finite")
    if np.any(lengths < 1):
        raise ValueError("Bayesian ALPS output lengths must include at least EOS")
    if fold_ids is None:
        resolved_folds = grouped_fold_ids(groups, folds=folds, seed=seed)
    else:
        resolved_folds = np.asarray(fold_ids, dtype=np.int32)
        if resolved_folds.shape != (len(features),):
            raise ValueError("fold_ids must align with hidden_states")
        unique_folds = sorted(set(int(value) for value in resolved_folds))
        if unique_folds != list(range(folds)):
            raise ValueError("fold_ids must cover contiguous values 0..folds-1")
        group_folds: dict[Hashable, int] = {}
        for group, fold in zip(groups, resolved_folds, strict=True):
            previous = group_folds.setdefault(group, int(fold))
            if previous != int(fold):
                raise ValueError("one group cannot appear in multiple folds")

    oof_mu = np.full(len(features), np.nan, dtype=np.float64)
    for fold in range(folds):
        validation_mask = resolved_folds == fold
        training_mask = ~validation_mask
        if not np.any(validation_mask) or not np.any(training_mask):
            raise ValueError("each fold needs both training and validation rows")
        fold_prior = fit_log1p_ridge_prior(
            features[training_mask],
            lengths[training_mask].astype(np.int64),
            alpha=alpha,
        )
        oof_mu[validation_mask] = np.asarray(
            [fold_prior.predict_mu(row) for row in features[validation_mask]],
            dtype=np.float64,
        )
    if np.any(~np.isfinite(oof_mu)):
        raise RuntimeError("grouped OOF prior predictions are incomplete")
    residuals = np.log1p(lengths) - oof_mu
    oof_variance = float(np.mean(np.square(residuals)))
    full_prior = fit_log1p_ridge_prior(features, lengths.astype(np.int64), alpha=alpha)
    calibrated = StandardizedRidgeLogNormalPrior(
        weights=full_prior.weights,
        bias=full_prior.bias,
        feature_mean=full_prior.feature_mean,
        feature_scale=full_prior.feature_scale,
        residual_variance=oof_variance,
        ridge_alpha=full_prior.ridge_alpha,
        residual_variance_estimator="family_grouped_oof_log1p_residual_mle",
    )
    return calibrated, oof_mu, resolved_folds

from __future__ import annotations

import math
from collections.abc import Sequence
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
            "residual_variance_estimator": "maximum_likelihood",
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

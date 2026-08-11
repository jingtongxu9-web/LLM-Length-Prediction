"""Numerically stable discrete Bayesian filtering for remaining output length."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _as_finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if np.any(np.isnan(vector)) or np.any(np.isposinf(vector)):
        raise ValueError(f"{name} cannot contain NaN or positive infinity")
    return vector


def logsumexp(values: np.ndarray) -> float:
    """Return log(sum(exp(values))) without overflow."""

    vector = _as_finite_vector(values, name="values")
    maximum = float(np.max(vector))
    if math.isinf(maximum) and maximum < 0:
        return -math.inf
    return maximum + math.log(float(np.exp(vector - maximum).sum()))


def normalize_log_probabilities(log_probabilities: np.ndarray) -> np.ndarray:
    """Normalize a log-mass vector and reject an empty support."""

    vector = _as_finite_vector(log_probabilities, name="log_probabilities")
    normalizer = logsumexp(vector)
    if not math.isfinite(normalizer):
        raise ValueError("probability support has zero total mass")
    normalized = vector - normalizer
    if not np.all(np.isfinite(normalized[np.isfinite(vector)])):
        raise ValueError("probability normalization failed")
    return normalized


def probabilities_to_log(probabilities: np.ndarray) -> np.ndarray:
    vector = np.asarray(probabilities, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or np.any(~np.isfinite(vector)):
        raise ValueError("probabilities must be a finite non-empty vector")
    if np.any(vector < 0):
        raise ValueError("probabilities cannot be negative")
    total = float(vector.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("probabilities must have positive finite mass")
    normalized = vector / total
    with np.errstate(divide="ignore"):
        return np.log(normalized)


def log_to_probabilities(log_probabilities: np.ndarray) -> np.ndarray:
    return np.exp(normalize_log_probabilities(log_probabilities))


def shifted_lognormal_cdf(value: float, mu: float, variance: float) -> float:
    """CDF of L when log(1 + L) follows Normal(mu, variance)."""

    if not math.isfinite(value) or not math.isfinite(mu):
        raise ValueError("value and mu must be finite")
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and non-negative")
    if value <= -1.0:
        return 0.0
    if variance == 0.0:
        return float(math.log1p(value) >= mu)
    z_score = (math.log1p(value) - mu) / math.sqrt(variance)
    return 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))


@dataclass(frozen=True)
class DiscreteLengthPrior:
    """Integer total-length probabilities plus an explicit upper-tail state."""

    probabilities: np.ndarray
    exact_max_total_tokens: int
    lower_tail_mass: float
    upper_tail_mass: float

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities, dtype=np.float64)
        expected_size = self.exact_max_total_tokens + 2
        if self.exact_max_total_tokens < 1 or values.shape != (expected_size,):
            raise ValueError("prior must contain exact states 0..max plus overflow")
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("prior probabilities must be finite and non-negative")
        if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-10):
            raise ValueError("prior probabilities must sum to one")
        if not 0.0 <= self.lower_tail_mass <= 1.0:
            raise ValueError("lower_tail_mass must be a probability")
        if not 0.0 <= self.upper_tail_mass <= 1.0:
            raise ValueError("upper_tail_mass must be a probability")

    @property
    def overflow_index(self) -> int:
        return self.exact_max_total_tokens + 1

    @property
    def log_probabilities(self) -> np.ndarray:
        return probabilities_to_log(self.probabilities)


def shifted_lognormal_integer_prior(
    mu: float,
    variance: float,
    *,
    max_total_tokens: int,
    min_total_tokens: int = 1,
) -> DiscreteLengthPrior:
    """Discretize an ALPS shifted log-normal prior using half-integer CDF bins.

    The returned vector is indexed by exact total length ``0..max_total_tokens``;
    its final entry is an overflow state for longer completions. Invalid mass below
    ``min_total_tokens - 0.5`` is removed and all remaining mass is normalized.
    """

    if max_total_tokens < 1:
        raise ValueError("max_total_tokens must be positive")
    if not 1 <= min_total_tokens <= max_total_tokens:
        raise ValueError("min_total_tokens must lie inside the exact support")
    if not math.isfinite(mu) or not math.isfinite(variance) or variance < 0:
        raise ValueError("mu and variance must be finite; variance cannot be negative")

    lower_boundary = max(0.5, min_total_tokens - 0.5)
    lower_tail = shifted_lognormal_cdf(lower_boundary, mu, variance)
    upper_boundary = max_total_tokens + 0.5
    upper_tail = max(
        0.0,
        1.0 - shifted_lognormal_cdf(upper_boundary, mu, variance),
    )
    exact_mass = np.zeros(max_total_tokens + 1, dtype=np.float64)
    for length in range(min_total_tokens, max_total_tokens + 1):
        lower = max(lower_boundary, length - 0.5)
        upper = length + 0.5
        exact_mass[length] = max(
            0.0,
            shifted_lognormal_cdf(upper, mu, variance)
            - shifted_lognormal_cdf(lower, mu, variance),
        )

    retained_mass = float(exact_mass.sum()) + upper_tail
    if retained_mass <= 0.0 or not math.isfinite(retained_mass):
        raise ValueError("shifted-lognormal prior has no mass on or above the support")
    probabilities = np.concatenate((exact_mass, np.asarray([upper_tail]))) / retained_mass
    return DiscreteLengthPrior(
        probabilities=probabilities,
        exact_max_total_tokens=max_total_tokens,
        lower_tail_mass=lower_tail,
        upper_tail_mass=upper_tail,
    )


def transition_log_posterior(
    log_posterior: np.ndarray,
    delta: int,
    *,
    has_overflow: bool = True,
) -> np.ndarray:
    """Count down by ``delta`` tokens and condition on survival to the new step."""

    previous = _as_finite_vector(log_posterior, name="log_posterior")
    if delta <= 0:
        raise ValueError("delta must be positive")
    finite_size = previous.size - int(has_overflow)
    if finite_size <= delta:
        raise ValueError("delta removes the entire exact remaining-length support")
    shifted_exact = previous[delta:finite_size]
    shifted = (
        np.concatenate((shifted_exact, previous[-1:]))
        if has_overflow
        else shifted_exact
    )
    return normalize_log_probabilities(shifted)


def bayesian_log_update(log_predictive: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Apply incremental log-likelihood-ratio scores to a predictive distribution."""

    predictive = _as_finite_vector(log_predictive, name="log_predictive")
    likelihood_scores = np.asarray(scores, dtype=np.float64)
    if likelihood_scores.shape != predictive.shape or np.any(~np.isfinite(likelihood_scores)):
        raise ValueError("scores must be finite and align with log_predictive")
    return normalize_log_probabilities(predictive + likelihood_scores)


@dataclass(frozen=True)
class CredibleInterval:
    level: float
    lower: float
    upper: float


@dataclass(frozen=True)
class PosteriorSummary:
    mean_remaining_lower_bound: float
    median_remaining: float
    mode_remaining: float
    variance_lower_bound: float
    entropy: float
    overflow_probability: float
    credible_intervals: tuple[CredibleInterval, ...]


def _posterior_quantile(
    probabilities: np.ndarray,
    quantile: float,
    *,
    has_overflow: bool,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    cumulative = np.cumsum(probabilities)
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    if has_overflow and index == probabilities.size - 1:
        return math.inf
    return float(min(index, probabilities.size - 1))


def summarize_posterior(
    probabilities: np.ndarray,
    *,
    has_overflow: bool = True,
    interval_levels: tuple[float, ...] = (0.5, 0.9, 0.95),
) -> PosteriorSummary:
    """Summarize a posterior without pretending the overflow state is exact."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("probabilities must be a finite one-dimensional vector")
    if np.any(values < 0) or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-8):
        raise ValueError("probabilities must be non-negative and sum to one")
    finite_size = values.size - int(has_overflow)
    if finite_size <= 0:
        raise ValueError("posterior has no exact states")
    overflow_probability = float(values[-1]) if has_overflow else 0.0
    representatives = np.arange(values.size, dtype=np.float64)
    if has_overflow:
        representatives[-1] = float(finite_size)
    mean_lower = float(np.sum(representatives * values))
    variance_lower = float(np.sum(np.square(representatives - mean_lower) * values))
    positive = values > 0
    entropy = -float(np.sum(values[positive] * np.log(values[positive])))
    mode_index = int(np.argmax(values))
    mode = math.inf if has_overflow and mode_index == values.size - 1 else float(mode_index)
    intervals = []
    for level in interval_levels:
        if not 0.0 < level < 1.0:
            raise ValueError("credible interval levels must lie in (0, 1)")
        tail = (1.0 - level) / 2.0
        lower = _posterior_quantile(values, tail, has_overflow=has_overflow)
        upper = _posterior_quantile(values, 1.0 - tail, has_overflow=has_overflow)
        intervals.append(CredibleInterval(level=level, lower=lower, upper=upper))
    return PosteriorSummary(
        mean_remaining_lower_bound=mean_lower,
        median_remaining=_posterior_quantile(values, 0.5, has_overflow=has_overflow),
        mode_remaining=mode,
        variance_lower_bound=variance_lower,
        entropy=entropy,
        overflow_probability=overflow_probability,
        credible_intervals=tuple(intervals),
    )


@dataclass(frozen=True)
class BayesianFilterState:
    """One request-local posterior. Model parameters live outside this state."""

    step: int
    log_probabilities: np.ndarray
    has_overflow: bool = True

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step cannot be negative")
        normalized = normalize_log_probabilities(self.log_probabilities)
        if not np.allclose(
            np.exp(normalized),
            np.exp(self.log_probabilities),
            atol=1e-8,
            rtol=1e-8,
        ):
            raise ValueError("log_probabilities must already be normalized")
        if self.has_overflow and normalized.size < 3:
            raise ValueError("overflow posterior requires exact states and one tail state")

    @classmethod
    def from_probabilities(
        cls,
        probabilities: np.ndarray,
        *,
        step: int = 0,
        has_overflow: bool = True,
    ) -> BayesianFilterState:
        return cls(
            step=step,
            log_probabilities=probabilities_to_log(probabilities),
            has_overflow=has_overflow,
        )

    @property
    def probabilities(self) -> np.ndarray:
        return np.exp(self.log_probabilities)

    @property
    def exact_max_remaining(self) -> int:
        return self.log_probabilities.size - 1 - int(self.has_overflow)

    def predict_to(self, step: int) -> BayesianFilterState:
        if step <= self.step:
            raise ValueError("new step must be greater than the current step")
        return BayesianFilterState(
            step=step,
            log_probabilities=transition_log_posterior(
                self.log_probabilities,
                step - self.step,
                has_overflow=self.has_overflow,
            ),
            has_overflow=self.has_overflow,
        )

    def update(self, scores: np.ndarray) -> BayesianFilterState:
        return BayesianFilterState(
            step=self.step,
            log_probabilities=bayesian_log_update(self.log_probabilities, scores),
            has_overflow=self.has_overflow,
        )

    def summary(self) -> PosteriorSummary:
        return summarize_posterior(
            self.probabilities,
            has_overflow=self.has_overflow,
        )

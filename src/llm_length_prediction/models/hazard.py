"""Conversions between a discrete remaining-length posterior and stopping hazards."""

from __future__ import annotations

import math

import numpy as np


def posterior_to_hazard(
    probabilities: np.ndarray,
    *,
    has_overflow: bool = True,
) -> np.ndarray:
    """Return q(r) = P(R=r | R>=r) for every finite remaining length."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("probabilities must be a finite one-dimensional vector")
    if np.any(values < 0) or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-8):
        raise ValueError("probabilities must be non-negative and sum to one")
    finite_size = values.size - int(has_overflow)
    if finite_size <= 0:
        raise ValueError("posterior must contain at least one finite state")
    survival = np.cumsum(values[::-1])[::-1]
    hazards = np.divide(
        values[:finite_size],
        survival[:finite_size],
        out=np.zeros(finite_size, dtype=np.float64),
        where=survival[:finite_size] > 0,
    )
    return np.clip(hazards, 0.0, 1.0)


def hazard_to_posterior(
    hazards: np.ndarray,
    *,
    include_survival_tail: bool = True,
) -> np.ndarray:
    """Reconstruct exact masses and, optionally, the unresolved overflow mass."""

    values = np.asarray(hazards, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("hazards must be a finite non-empty vector")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("hazards must lie in [0, 1]")
    masses = np.zeros_like(values)
    survival = 1.0
    for index, hazard in enumerate(values):
        masses[index] = survival * hazard
        survival *= 1.0 - hazard
    if include_survival_tail:
        posterior = np.concatenate((masses, np.asarray([survival])))
    else:
        if survival > 1e-8:
            raise ValueError("hazards leave unresolved survival mass")
        posterior = masses
    total = float(posterior.sum())
    if total <= 0.0:
        raise ValueError("hazards imply zero probability mass")
    return posterior / total


def expected_stop_within(
    probabilities: np.ndarray,
    horizon: int,
    *,
    has_overflow: bool = True,
) -> float:
    """Return P(R <= horizon) without assigning overflow to a fake token count."""

    values = np.asarray(probabilities, dtype=np.float64)
    finite_size = values.size - int(has_overflow)
    if horizon < 0 or finite_size <= 0:
        raise ValueError("horizon must be non-negative and support must be non-empty")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-8):
        raise ValueError("probabilities must sum to one")
    return float(values[: min(horizon + 1, finite_size)].sum())

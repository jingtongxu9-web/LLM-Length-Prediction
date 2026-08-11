"""Probability, point, convergence, and runtime metrics for Bayesian sequences."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from llm_length_prediction.data.sequential import BayesianSequence
from llm_length_prediction.models.bayesian_filter import (
    BayesianFilterState,
    PosteriorSummary,
)
from llm_length_prediction.models.bayesian_scorer import HIDDEN_DELTA_METHOD_ID


def posterior_nll(
    probabilities: np.ndarray,
    true_remaining: int,
    *,
    has_overflow: bool = True,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    finite_size = values.size - int(has_overflow)
    if true_remaining < 0 or true_remaining >= finite_size:
        raise ValueError("true_remaining lies outside exact support")
    probability = float(values[true_remaining])
    return -math.log(max(probability, np.finfo(np.float64).tiny))


def right_censored_posterior_nll(
    probabilities: np.ndarray,
    censored_after_remaining: int,
    *,
    has_overflow: bool = True,
) -> float:
    """Return -log P(R > censor boundary), including the explicit overflow state."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("probabilities must be a finite one-dimensional vector")
    if np.any(values < 0) or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-8):
        raise ValueError("probabilities must be non-negative and sum to one")
    finite_size = values.size - int(has_overflow)
    if censored_after_remaining < 0:
        raise ValueError("censoring threshold cannot be negative")
    survival = float(values[censored_after_remaining + 1 : finite_size].sum())
    if has_overflow:
        survival += float(values[-1])
    return -math.log(max(survival, np.finfo(np.float64).tiny))


def discrete_crps(
    probabilities: np.ndarray,
    true_remaining: int,
    *,
    has_overflow: bool = True,
) -> float:
    """Discrete CRPS with overflow represented at its conservative boundary value."""

    values = np.asarray(probabilities, dtype=np.float64)
    finite_size = values.size - int(has_overflow)
    if true_remaining < 0 or true_remaining >= finite_size:
        raise ValueError("true_remaining lies outside exact support")
    cumulative = np.cumsum(values)
    support = np.arange(values.size)
    observation_cdf = (support >= true_remaining).astype(np.float64)
    return float(np.square(cumulative - observation_cdf).sum())


@dataclass(frozen=True)
class PosteriorObservation:
    prompt_id: str
    prompt_family_id: str
    seed: int
    step: int
    probabilities: np.ndarray
    summary: PosteriorSummary
    true_remaining: int | None
    censored_after_remaining: int | None
    terminal_observed: bool
    update_wall_time_ms: float
    predictor_state_bytes: int = 0
    has_overflow: bool = True

    @property
    def sequence_id(self) -> tuple[str, int]:
        return self.prompt_id, self.seed

    @property
    def true_total_tokens(self) -> int | None:
        if self.true_remaining is None:
            return None
        return self.step + self.true_remaining

    @property
    def predicted_total_tokens_lower_bound(self) -> float:
        return self.step + self.summary.mean_remaining_lower_bound


def run_bayesian_sequence(
    sequence: BayesianSequence,
    scorer: Any,
    *,
    device: str = "cpu",
) -> list[PosteriorObservation]:
    """Run one frozen scorer without changing its model parameters."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Bayesian scorer inference requires PyTorch") from error
    sequence.validate()
    spec = getattr(scorer, "bayesian_spec", {})
    method_id = spec.get("method_id")
    if method_id not in {
        "bayesian_entropy_scalar_v1",
        HIDDEN_DELTA_METHOD_ID,
    }:
        raise ValueError("scorer is missing a recognized Bayesian method_id")
    scorer.eval()
    scorer_bytes = sum(
        value.numel() * value.element_size()
        for value in (*tuple(scorer.parameters()), *tuple(scorer.buffers()))
    )
    state = BayesianFilterState(
        step=0,
        log_probabilities=np.asarray(sequence.initial_log_probabilities, dtype=np.float64),
        has_overflow=sequence.has_overflow,
    )
    observations = []
    with torch.inference_mode():
        for evidence_step in sequence.steps:
            started = time.perf_counter()
            predictive = state.predict_to(evidence_step.step)
            evidence = torch.as_tensor(
                evidence_step.scalar_features[None, :],
                dtype=torch.float32,
                device=device,
            )
            candidates = torch.as_tensor(
                evidence_step.candidate_features[None, :, :],
                dtype=torch.float32,
                device=device,
            )
            hidden = None
            if method_id == HIDDEN_DELTA_METHOD_ID:
                if evidence_step.hidden_delta is None:
                    raise ValueError("hidden-delta scorer received a scalar-only sequence")
                hidden = torch.as_tensor(
                    evidence_step.hidden_delta[None, :],
                    dtype=torch.float32,
                    device=device,
                )
            scores = scorer(evidence, candidates, hidden).squeeze(0).cpu().numpy()
            state = predictive.update(scores)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            observations.append(
                PosteriorObservation(
                    prompt_id=sequence.prompt_id,
                    prompt_family_id=sequence.prompt_family_id,
                    seed=sequence.seed,
                    step=evidence_step.step,
                    probabilities=state.probabilities,
                    summary=state.summary(),
                    true_remaining=evidence_step.true_remaining,
                    censored_after_remaining=evidence_step.censored_after_remaining,
                    terminal_observed=evidence_step.terminal_observed,
                    update_wall_time_ms=elapsed_ms,
                    predictor_state_bytes=(
                        scorer_bytes
                        + state.log_probabilities.nbytes
                        + np.asarray(scores).nbytes
                    ),
                    has_overflow=sequence.has_overflow,
                )
            )
    return observations


def stable_time_to_relative_error(
    observations: Sequence[PosteriorObservation],
    *,
    threshold: float = 0.05,
) -> int | None:
    """First saved step whose error and every later saved error stay in tolerance."""

    if not observations or not 0.0 < threshold < 1.0:
        raise ValueError("observations are required and threshold must lie in (0, 1)")
    ordered = sorted(observations, key=lambda observation: observation.step)
    if any(observation.true_total_tokens is None for observation in ordered):
        return None
    within = []
    for observation in ordered:
        truth = int(observation.true_total_tokens)
        relative_error = abs(observation.predicted_total_tokens_lower_bound - truth) / max(
            truth,
            1,
        )
        within.append(relative_error <= threshold)
    suffix_all = True
    first_stable = None
    for index in range(len(ordered) - 1, -1, -1):
        suffix_all = suffix_all and within[index]
        if suffix_all:
            first_stable = ordered[index].step
    return first_stable


def _raw_r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.square(actual - actual.mean()).sum())
    if denominator == 0.0:
        return 0.0
    return 1.0 - float(np.square(actual - predicted).sum()) / denominator


def evaluate_posterior_observations(
    observations: Sequence[PosteriorObservation],
) -> dict[str, int | float]:
    """Compute rollout-balanced and family-macro frozen Bayesian metrics."""

    if not observations:
        raise ValueError("at least one posterior observation is required")
    grouped: dict[tuple[str, int], list[PosteriorObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.sequence_id].append(observation)

    per_sequence_nll = []
    per_sequence_crps = []
    per_sequence_mae = []
    per_sequence_squared_error = []
    per_sequence_bias = []
    per_sequence_variance = []
    per_sequence_entropy = []
    per_sequence_coverages: dict[float, list[float]] = defaultdict(list)
    per_sequence_widths: dict[float, list[float]] = defaultdict(list)
    family_sequence_nll: dict[str, list[float]] = defaultdict(list)
    terminal_nll = []
    nonterminal_nll = []
    censored_nll = []
    stable_times = []
    stable_successes = 0
    all_actual = []
    all_predicted = []
    for sequence_observations in grouped.values():
        point_nll = []
        point_crps = []
        point_errors = []
        point_variances = []
        point_entropies = []
        point_coverages: dict[float, list[float]] = defaultdict(list)
        point_widths: dict[float, list[float]] = defaultdict(list)
        for observation in sequence_observations:
            if observation.true_remaining is not None:
                current_nll = posterior_nll(
                    observation.probabilities,
                    observation.true_remaining,
                    has_overflow=observation.has_overflow,
                )
                point_nll.append(current_nll)
                if observation.terminal_observed:
                    terminal_nll.append(current_nll)
                else:
                    nonterminal_nll.append(current_nll)
                point_crps.append(
                    discrete_crps(
                        observation.probabilities,
                        observation.true_remaining,
                        has_overflow=observation.has_overflow,
                    )
                )
                error = (
                    observation.summary.mean_remaining_lower_bound
                    - observation.true_remaining
                )
                point_errors.append(error)
                point_variances.append(observation.summary.variance_lower_bound)
                point_entropies.append(observation.summary.entropy)
                all_actual.append(float(observation.true_remaining))
                all_predicted.append(observation.summary.mean_remaining_lower_bound)
                for interval in observation.summary.credible_intervals:
                    covered = interval.lower <= observation.true_remaining <= interval.upper
                    point_coverages[interval.level].append(float(covered))
                    point_widths[interval.level].append(interval.upper - interval.lower)
            else:
                current_nll = right_censored_posterior_nll(
                    observation.probabilities,
                    int(observation.censored_after_remaining),
                    has_overflow=observation.has_overflow,
                )
                point_nll.append(current_nll)
                censored_nll.append(current_nll)
        sequence_nll = float(np.mean(point_nll))
        per_sequence_nll.append(sequence_nll)
        family_sequence_nll[sequence_observations[0].prompt_family_id].append(sequence_nll)
        if point_crps:
            per_sequence_crps.append(float(np.mean(point_crps)))
            errors = np.asarray(point_errors, dtype=np.float64)
            per_sequence_mae.append(float(np.mean(np.abs(errors))))
            per_sequence_squared_error.append(float(np.mean(np.square(errors))))
            per_sequence_bias.append(float(np.mean(errors)))
            per_sequence_variance.append(float(np.mean(point_variances)))
            per_sequence_entropy.append(float(np.mean(point_entropies)))
            for level, values in point_coverages.items():
                per_sequence_coverages[level].append(float(np.mean(values)))
            for level, values in point_widths.items():
                per_sequence_widths[level].append(float(np.mean(values)))
            stable = stable_time_to_relative_error(sequence_observations)
            if stable is not None:
                stable_successes += 1
                stable_times.append(stable)

    exact_sequence_count = len(per_sequence_mae)
    metrics: dict[str, int | float] = {
        "observation_count": len(observations),
        "sequence_count": len(grouped),
        "family_count": len(family_sequence_nll),
        "family_macro_sequence_balanced_posterior_nll": float(
            np.mean(
                [np.mean(values) for values in family_sequence_nll.values()]
            )
        ),
        "sequence_balanced_posterior_nll": float(np.mean(per_sequence_nll)),
        "mean_update_wall_time_ms": float(
            np.mean([observation.update_wall_time_ms for observation in observations])
        ),
        "peak_predictor_state_bytes": max(
            observation.predictor_state_bytes for observation in observations
        ),
        "mean_overflow_probability": float(
            np.mean(
                [observation.summary.overflow_probability for observation in observations]
            )
        ),
        "terminal_posterior_nll": (
            float(np.mean(terminal_nll)) if terminal_nll else math.nan
        ),
        "nonterminal_posterior_nll": (
            float(np.mean(nonterminal_nll)) if nonterminal_nll else math.nan
        ),
        "right_censored_posterior_nll": (
            float(np.mean(censored_nll)) if censored_nll else math.nan
        ),
    }
    if exact_sequence_count:
        actual = np.asarray(all_actual, dtype=np.float64)
        predicted = np.asarray(all_predicted, dtype=np.float64)
        metrics.update(
            {
                "sequence_balanced_crps": float(np.mean(per_sequence_crps)),
                "sequence_balanced_mae_tokens": float(np.mean(per_sequence_mae)),
                "sequence_balanced_rmse_tokens": float(
                    math.sqrt(float(np.mean(per_sequence_squared_error)))
                ),
                "sequence_balanced_bias_tokens": float(np.mean(per_sequence_bias)),
                "sequence_balanced_posterior_variance_lower_bound": float(
                    np.mean(per_sequence_variance)
                ),
                "sequence_balanced_posterior_entropy": float(
                    np.mean(per_sequence_entropy)
                ),
                "raw_r_squared_tokens": _raw_r_squared(actual, predicted),
                "severe_underestimation_rate_100_tokens": float(
                    np.mean((actual - predicted) > 100.0)
                ),
                "stable_time_to_5pct_success_rate": stable_successes
                / exact_sequence_count,
                "stable_time_to_5pct_mean_tokens_on_success": (
                    float(np.mean(stable_times)) if stable_times else math.nan
                ),
            }
        )
        for level, values in sorted(per_sequence_coverages.items()):
            label = int(round(level * 100))
            metrics[f"sequence_balanced_interval_{label}_coverage"] = float(
                np.mean(values)
            )
            metrics[f"sequence_balanced_interval_{label}_mean_width"] = float(
                np.mean(per_sequence_widths[level])
            )
    return metrics

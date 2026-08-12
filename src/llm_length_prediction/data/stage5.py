"""Stage-5 views over the unified Bayesian trace without copying raw data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from llm_length_prediction.data.bayesian_trace import (
    BayesianTraceV1,
    sequential_raw_trace_from_collected,
)
from llm_length_prediction.data.sequential import BayesianSequence, build_bayesian_sequence
from llm_length_prediction.models.hybrid import HybridSample
from llm_length_prediction.models.prior import shifted_lognormal_mean


def rolling_mean_slope(values: np.ndarray, *, window: int) -> tuple[float, float]:
    if window <= 1:
        raise ValueError("rolling window must exceed one")
    recent = np.asarray(values[-window:], dtype=np.float64)
    mean = float(recent.mean())
    if len(recent) <= 1:
        return mean, 0.0
    positions = np.arange(len(recent), dtype=np.float64)
    centered = positions - positions.mean()
    denominator = float(np.square(centered).sum())
    slope = float(np.sum(centered * (recent - mean)) / denominator)
    return mean, slope


def stage5_hybrid_samples(
    trace: BayesianTraceV1,
    *,
    entropy_window: int = 20,
) -> list[HybridSample]:
    """Build PLP/concat controls from the same unified saved points."""

    trace.validate()
    weight = 1.0 / len(trace.saved_steps)
    samples = []
    for index, raw_step in enumerate(trace.saved_steps):
        step = int(raw_step)
        entropy_mean, entropy_slope = rolling_mean_slope(
            trace.token_entropies[:step], window=entropy_window
        )
        samples.append(
            HybridSample(
                prompt_id=trace.prompt_id,
                prompt_family_id=trace.prompt_family_id,
                task=trace.task,
                intended_length=trace.intended_length,
                temperature=trace.temperature,
                seed=trace.seed,
                step=step,
                output_tokens=trace.observed_tokens,
                remaining_tokens=trace.observed_tokens - step,
                prior_feature=trace.prior_feature,
                prompt_feature=trace.prompt_feature,
                decode_feature=trace.decode_hidden_states[index],
                dynamic_features=(
                    float(step),
                    float(trace.token_entropies[step - 1]),
                    entropy_mean,
                    entropy_slope,
                    float(trace.token_eos_probabilities[step - 1]),
                ),
                sequence_weight=weight,
            )
        )
    return samples


def stage5_bayesian_sequence(
    trace: BayesianTraceV1,
    *,
    prior_mu: float,
    prior_log_variance: float,
) -> BayesianSequence:
    raw = sequential_raw_trace_from_collected(
        trace,
        prior_mu=prior_mu,
        prior_log_variance=prior_log_variance,
        prior_mean_total_tokens=shifted_lognormal_mean(
            prior_mu, prior_log_variance
        ),
    )
    return build_bayesian_sequence(raw)


def stage5_prior_summary_matrix(
    samples: Sequence[HybridSample],
    *,
    trace_mu: dict[tuple[str, float, int], float],
    variance: float,
) -> np.ndarray:
    if variance < 0:
        raise ValueError("prior variance cannot be negative")
    rows = []
    for sample in samples:
        mu = trace_mu[sample.trace_key]
        mean_total = shifted_lognormal_mean(mu, variance)
        median_total = max(0.0, float(np.expm1(mu)))
        rows.append(
            (
                mu,
                variance,
                mean_total,
                max(mean_total - sample.step, 0.0),
                max(median_total - sample.step, 0.0),
            )
        )
    return np.asarray(rows, dtype=np.float32)

import math

import numpy as np

from llm_length_prediction.models.bayesian_filter import (
    BayesianFilterState,
    bayesian_log_update,
    shifted_lognormal_integer_prior,
    summarize_posterior,
    transition_log_posterior,
)
from llm_length_prediction.models.hazard import (
    hazard_to_posterior,
    posterior_to_hazard,
)


def test_shifted_lognormal_integer_prior_has_exact_degenerate_mass() -> None:
    prior = shifted_lognormal_integer_prior(
        math.log1p(7),
        0.0,
        max_total_tokens=10,
    )
    assert np.isclose(prior.probabilities.sum(), 1.0)
    assert prior.probabilities[7] == 1.0
    assert prior.probabilities[-1] == 0.0


def test_shifted_lognormal_preserves_upper_tail_as_overflow() -> None:
    prior = shifted_lognormal_integer_prior(
        math.log1p(7),
        0.0,
        max_total_tokens=5,
    )
    assert prior.overflow_index == 6
    assert prior.probabilities[-1] == 1.0
    assert prior.upper_tail_mass == 1.0


def test_transition_counts_down_and_conditions_on_survival() -> None:
    previous = np.asarray([0.0, 0.1, 0.2, 0.3, 0.3, 0.1])
    with np.errstate(divide="ignore"):
        log_previous = np.log(previous)
    transitioned = np.exp(transition_log_posterior(log_previous, 2, has_overflow=True))
    assert np.allclose(transitioned, np.asarray([0.2, 0.3, 0.3, 0.1]) / 0.9)


def test_extreme_likelihood_scores_remain_normalized() -> None:
    predictive = np.log(np.asarray([0.2, 0.3, 0.5]))
    posterior = bayesian_log_update(
        predictive,
        np.asarray([-10000.0, 0.0, 10000.0]),
    )
    assert np.isclose(np.exp(posterior).sum(), 1.0)
    assert np.argmax(posterior) == 2
    assert not np.any(np.isnan(posterior))


def test_positive_incremental_evidence_moves_mass_toward_its_candidate() -> None:
    predictive_probability = np.asarray([0.2, 0.5, 0.3])
    posterior = np.exp(
        bayesian_log_update(
            np.log(predictive_probability),
            np.asarray([0.0, 2.0, 0.0]),
        )
    )
    assert posterior[1] > predictive_probability[1]
    assert posterior[0] < predictive_probability[0]
    assert posterior[2] < predictive_probability[2]


def test_summary_does_not_turn_overflow_into_a_false_finite_interval() -> None:
    summary = summarize_posterior(np.asarray([0.01, 0.04, 0.05, 0.90]))
    interval_95 = next(
        interval for interval in summary.credible_intervals if interval.level == 0.95
    )
    assert summary.overflow_probability == 0.90
    assert math.isinf(summary.median_remaining)
    assert math.isinf(summary.mode_remaining)
    assert math.isinf(interval_95.upper)


def test_hazard_round_trip_retains_unresolved_tail_mass() -> None:
    posterior = np.asarray([0.1, 0.2, 0.3, 0.4])
    hazards = posterior_to_hazard(posterior, has_overflow=True)
    restored = hazard_to_posterior(hazards, include_survival_tail=True)
    assert np.allclose(restored, posterior)


def test_filter_state_updates_request_posterior_without_parameters() -> None:
    state = BayesianFilterState.from_probabilities(
        np.asarray([0.0, 0.2, 0.3, 0.4, 0.1]),
        has_overflow=True,
    )
    predictive = state.predict_to(1)
    updated = predictive.update(np.asarray([-5.0, -1.0, 2.0, 0.0]))
    assert predictive.step == updated.step == 1
    assert np.isclose(updated.probabilities.sum(), 1.0)
    assert updated.exact_max_remaining == 2

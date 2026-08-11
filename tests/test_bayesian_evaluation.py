import math

import numpy as np

from llm_length_prediction.data.sequential import build_synthetic_sequence
from llm_length_prediction.evaluation.sequential import (
    PosteriorObservation,
    discrete_crps,
    evaluate_posterior_observations,
    posterior_nll,
    right_censored_posterior_nll,
    run_bayesian_sequence,
    stable_time_to_relative_error,
)
from llm_length_prediction.models.bayesian_filter import summarize_posterior
from llm_length_prediction.models.bayesian_scorer import (
    SCALAR_METHOD_ID,
    build_likelihood_ratio_scorer,
    fit_scorer_standardization,
)


def _observation(step: int, true_total: int, predicted_remaining: int) -> PosteriorObservation:
    exact_max = true_total + 2
    probabilities = np.zeros(exact_max + 2)
    probabilities[predicted_remaining] = 1.0
    return PosteriorObservation(
        prompt_id="p",
        prompt_family_id="f",
        seed=42,
        step=step,
        probabilities=probabilities,
        summary=summarize_posterior(probabilities),
        true_remaining=true_total - step,
        censored_after_remaining=None,
        terminal_observed=step == true_total,
        update_wall_time_ms=1.0,
    )


def test_probability_metrics_reward_the_true_state_and_tail() -> None:
    exact = np.asarray([0.1, 0.7, 0.1, 0.1])
    assert math.isclose(posterior_nll(exact, 1), -math.log(0.7))
    assert discrete_crps(exact, 1) >= 0.0
    censored = np.asarray([0.2, 0.2, 0.1, 0.5])
    assert math.isclose(right_censored_posterior_nll(censored, 2), -math.log(0.5))


def test_stable_time_requires_all_later_predictions_to_stay_accurate() -> None:
    observations = [
        _observation(1, 10, 9),
        _observation(5, 10, 5),
        _observation(8, 10, 4),
        _observation(10, 10, 0),
    ]
    assert stable_time_to_relative_error(observations) == 10


def test_frozen_scorer_runs_sequence_and_reports_metrics() -> None:
    sequence = build_synthetic_sequence(output_tokens=8, max_new_tokens=12)
    standardization = fit_scorer_standardization([sequence])
    scorer = build_likelihood_ratio_scorer(
        SCALAR_METHOD_ID,
        standardization=standardization,
        hidden_dim=16,
        dropout=0.0,
    )
    observations = run_bayesian_sequence(sequence, scorer)
    metrics = evaluate_posterior_observations(observations)
    assert len(observations) == len(sequence.steps)
    assert metrics["sequence_count"] == 1
    assert metrics["family_count"] == 1
    assert metrics["sequence_balanced_posterior_nll"] >= 0.0
    assert 0.0 <= metrics["stable_time_to_5pct_success_rate"] <= 1.0

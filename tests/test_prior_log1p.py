import math

import numpy as np

from llm_length_prediction.evaluation.metrics import log1p_prior_metrics
from llm_length_prediction.models.prior import (
    StandardizedRidgeLogNormalPrior,
    fit_grouped_oof_log1p_prior,
    fit_log1p_ridge_prior,
    grouped_fold_ids,
    shifted_lognormal_mean,
)


def test_shifted_lognormal_mean() -> None:
    assert math.isclose(shifted_lognormal_mean(math.log1p(9), 0.0), 9.0)


def test_fit_uses_log1p_target_and_mle_residual_variance() -> None:
    features = [[0.0], [1.0], [2.0], [3.0]]
    lengths = [0, 2, 8, 26]
    prior = fit_log1p_ridge_prior(features, lengths, alpha=0.0)
    mus = np.asarray([prior.predict_mu(row) for row in features])
    residuals = np.log1p(lengths) - mus
    assert math.isclose(prior.residual_variance, float(np.mean(residuals**2)))
    assert prior.target == "log1p_output_tokens"

    restored = StandardizedRidgeLogNormalPrior.from_dict(prior.to_dict())
    assert restored == prior

    predicted = [prior.predict_mean_length(row) for row in features]
    metrics = log1p_prior_metrics(lengths, predicted, mus, prior.residual_variance)
    assert metrics["count"] == 4
    assert metrics["mae_tokens"] >= 0
    assert 0.0 <= metrics["interval_95_coverage"] <= 1.0


def test_grouped_oof_variance_never_predicts_a_family_with_itself() -> None:
    features = [[0.0], [0.2], [1.0], [1.2], [2.0], [2.2], [3.0], [3.2]]
    lengths = [2, 3, 5, 6, 12, 13, 25, 26]
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    folds = grouped_fold_ids(groups, folds=2, seed=7)
    for group in set(groups):
        indices = [index for index, value in enumerate(groups) if value == group]
        assert len(set(folds[indices])) == 1
    prior, oof_mu, resolved = fit_grouped_oof_log1p_prior(
        features,
        lengths,
        groups,
        folds=2,
        alpha=1.0,
        fold_ids=folds,
    )
    expected_variance = float(np.mean(np.square(np.log1p(lengths) - oof_mu)))
    assert np.isclose(prior.residual_variance, expected_variance)
    assert prior.residual_variance_estimator == "family_grouped_oof_log1p_residual_mle"
    assert np.array_equal(resolved, folds)


def test_grouped_oof_rejects_family_leakage_in_supplied_folds() -> None:
    with np.testing.assert_raises_regex(ValueError, "multiple folds"):
        fit_grouped_oof_log1p_prior(
            [[0.0], [0.1], [1.0], [1.1]],
            [2, 3, 5, 6],
            ["a", "a", "b", "b"],
            folds=2,
            fold_ids=[0, 1, 0, 1],
        )

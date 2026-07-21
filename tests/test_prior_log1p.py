import math

import numpy as np

from llm_length_prediction.models.prior import (
    StandardizedRidgeLogNormalPrior,
    fit_log1p_ridge_prior,
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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llm_length_prediction.models.prompt_token_baseline import (
    METHOD_ID,
    fit_prompt_token_ridge,
    predict_prompt_token_countdown,
)


def test_prompt_token_ridge_predicts_total_then_counts_down() -> None:
    model = fit_prompt_token_ridge(
        [10, 20, 30, 40], [20, 40, 60, 80], alpha=1.0
    )
    prediction = predict_prompt_token_countdown(model, [20, 20], [1, 10])
    assert prediction.shape == (2,)
    assert prediction[0] > prediction[1]
    np.testing.assert_allclose(prediction[0] - prediction[1], 9.0)
    assert np.all(prediction >= 0)


def test_main_comparison_protocol_freezes_four_methods_and_no_test() -> None:
    protocol = json.loads(
        Path("configs/experiments/alps_plp_main_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(protocol["methods"]) == (
        METHOD_ID,
        "alps_countdown",
        "plp_terminal_zero_v3",
        "alps_plp_concat_v1",
    )
    assert protocol["data_policy"]["test_access"] == (
        "forbidden_until_a_new_hybrid_holdout_is_authored"
    )
    assert protocol["evaluation"]["selection_status"] == (
        "concat_v1_was_selected_before_this_supplemental_baseline"
    )

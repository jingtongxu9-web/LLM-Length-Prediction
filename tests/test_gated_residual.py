from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm_length_prediction.evaluation.gated_residual import (
    gated_residual_diagnostics,
    weighted_quantiles,
)
from llm_length_prediction.models.gated_residual import (
    METHOD_ID,
    build_gated_residual_head,
    correction_bounds,
    fit_scalar_residual_ridge,
    predict_gated_residual_head,
    predict_scalar_residual_ridge,
    progress_values,
)
from llm_length_prediction.models.hybrid import HybridSample


def sample(step: int, remaining: int) -> HybridSample:
    return HybridSample(
        prompt_id="qa_001_short",
        prompt_family_id="qa_001",
        task="qa",
        intended_length="short",
        seed=42,
        step=step,
        output_tokens=step + remaining,
        remaining_tokens=remaining,
        prior_feature=np.asarray([1.0, 2.0], dtype=np.float32),
        prompt_feature=np.asarray([1.0, 2.0], dtype=np.float32),
        decode_feature=np.asarray([3.0, 4.0], dtype=np.float32),
        dynamic_features=(float(step), 1.5, 1.25, -0.1, 0.05),
        sequence_weight=0.5,
    )


def test_progress_and_correction_bounds_are_conservative() -> None:
    samples = [sample(5, 15), sample(15, 5)]
    base = np.asarray([15.0, 5.0], dtype=np.float32)
    progress = progress_values(samples, base)
    bounds = correction_bounds(base, fraction=0.5, minimum_tokens=16.0)
    np.testing.assert_allclose(progress, [0.25, 0.75])
    np.testing.assert_allclose(bounds, [16.0, 16.0])


def test_zero_initialized_gated_head_starts_as_alps() -> None:
    pytest.importorskip("torch")
    head = build_gated_residual_head(
        10, hidden_dim=4, dropout=0.0, gate_initial_bias=-3.0
    )
    base = np.asarray([12.0, 7.0], dtype=np.float32)
    progress = np.asarray([0.25, 0.75], dtype=np.float32)
    predicted, diagnostics = predict_gated_residual_head(
        head,
        np.zeros((2, 10), dtype=np.float32),
        base,
        progress,
        correction_bound_fraction=0.5,
        minimum_correction_bound_tokens=16.0,
        terminal_threshold=0.5,
        batch_size=2,
        device="cpu",
    )
    np.testing.assert_allclose(predicted, base)
    np.testing.assert_allclose(diagnostics["applied_correction"], 0.0)
    assert diagnostics["gate"][0] < diagnostics["gate"][1]
    np.testing.assert_allclose(
        diagnostics["gate_confidence"][0], diagnostics["gate_confidence"][1]
    )
    assert np.all(diagnostics["terminal_probability"] < 0.5)


def test_scalar_residual_ridge_returns_nonnegative_remaining_length() -> None:
    samples = [sample(5, 15), sample(10, 10), sample(15, 5), sample(20, 0)]
    prior = np.asarray(
        [
            [4.0, 0.2, 25.0, 20.0, 19.0],
            [4.0, 0.2, 25.0, 15.0, 14.0],
            [4.0, 0.2, 25.0, 10.0, 9.0],
            [4.0, 0.2, 25.0, 5.0, 4.0],
        ],
        dtype=np.float32,
    )
    model = fit_scalar_residual_ridge(samples, prior, alpha=1.0)
    predicted = predict_scalar_residual_ridge(model, samples, prior)
    assert predicted.shape == (4,)
    assert np.all(np.isfinite(predicted))
    assert np.all(predicted >= 0)


def test_v2_1_protocol_is_supplemental_and_does_not_reuse_test() -> None:
    path = Path("configs/experiments/alps_plp_gated_residual_v2_1.json")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["method_id"] == METHOD_ID
    assert protocol["source_oof"]["protocol_id"] == (
        "alps-plp-hybrid-v1-v2-development-2026"
    )
    assert protocol["data_policy"]["reuse_exact_source_family_folds"] is True
    assert protocol["data_policy"]["test_access"] == (
        "forbidden_until_a_new_hybrid_holdout_is_authored"
    )
    assert protocol["method"]["hidden_dim"] == 512
    assert protocol["scalar_residual_ridge"]["alpha"] == 1.0
    assert protocol["training"]["weight_decay"] == 0.0001
    diagnostics = protocol["evaluation"]["gate_diagnostics"]
    assert diagnostics["decode_progress_labels"] == [
        "0-10%",
        "10-25%",
        "25-50%",
        "50-75%",
        "75-100%",
    ]
    assert diagnostics["gate_thresholds"] == [0.05, 0.1, 0.25, 0.5]


def test_weighted_quantiles_respect_sequence_weights() -> None:
    result = weighted_quantiles([0.0, 0.5, 1.0], [8.0, 1.0, 1.0], [0.5, 0.9])
    assert result["p50"] < 0.2
    assert result["p90"] == pytest.approx(0.75)


def test_gate_diagnostics_expose_usefulness_bins_and_terminal_errors() -> None:
    samples = [sample(1, 9), sample(5, 5), sample(9, 1), sample(10, 0)]
    alps = np.asarray([8.0, 7.0, 2.0, 1.0])
    candidate = np.asarray([9.0, 5.5, 1.0, 0.0])
    predictions = {
        "alps_countdown": alps,
        "alps_plp_concat_v1": np.asarray([9.5, 5.0, 1.5, 0.0]),
        "alps_scalar_residual_ridge": np.asarray([8.5, 6.0, 1.0, 0.0]),
        METHOD_ID: candidate,
    }
    report = gated_residual_diagnostics(
        samples,
        predictions,
        candidate_id=METHOD_ID,
        alps_id="alps_countdown",
        concat_id="alps_plp_concat_v1",
        scalar_id="alps_scalar_residual_ridge",
        applied=np.asarray([1.0, -1.5, -1.0, -1.0]),
        gate=np.asarray([0.02, 0.08, 0.3, 0.8]),
        gate_confidence=np.asarray([0.2, 0.16, 1.0 / 3.0, 0.8]),
        bounded=np.asarray([2.0, -3.0, -4.0, -16.0]),
        terminal_probability=np.asarray([0.01, 0.02, 0.8, 0.9]),
        progress=np.asarray([0.1, 0.5, 0.9, 1.0]),
        bounds=np.asarray([16.0, 16.0, 16.0, 16.0]),
        family_folds={"qa_001": 0},
        settings={
            "decode_progress_boundaries": [0.0, 0.25, 0.75, 1.0],
            "decode_progress_labels": ["early", "middle", "late"],
            "gate_boundaries": [0.0, 0.05, 0.25, 1.0],
            "gate_labels": ["closed", "partial", "open"],
            "gate_thresholds": [0.05, 0.5],
            "gate_quantiles": [0.5, 0.9],
            "bound_saturation_ratio": 0.95,
            "direction_tolerance_tokens": 1.0,
        },
        terminal_threshold=0.5,
    )
    assert len(report["by_decode_progress"]) == 3
    assert len(report["by_gate_band"]) == 3
    assert len(report["by_task_and_intended_length"]) == 1
    assert len(report["by_terminal_status"]) == 2
    assert report["terminal_classification"]["true_positive"] == 1
    assert report["terminal_classification"]["false_positive"] == 1
    assert report["overall"]["sequence_balanced_correction_success_rate"] > 0.5
    assert report["overall"]["gate_quantiles"]["p50"] > 0.05

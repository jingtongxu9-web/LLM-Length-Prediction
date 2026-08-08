from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm_length_prediction.models.hybrid import HybridSample
from llm_length_prediction.models.hybrid_versions import (
    METHOD_IDS,
    build_residual_head,
    fit_control_scaler,
    predict_residual_head,
    residual_control_matrix,
    residual_feature_matrix,
)


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


def test_residual_features_keep_alps_countdown_explicit() -> None:
    samples = [sample(5, 10), sample(10, 5)]
    prior = np.asarray(
        [[4.0, 0.2, 20.0, 15.0, 14.0], [4.0, 0.2, 20.0, 10.0, 9.0]],
        dtype=np.float32,
    )
    controls = residual_control_matrix(samples, prior)
    assert controls.shape == (2, 6)
    np.testing.assert_allclose(controls[:, 0], [15.0, 10.0])
    scaler = fit_control_scaler(controls, np.asarray([0.5, 0.5], dtype=np.float32))
    features = residual_feature_matrix(samples, prior, scaler=scaler)
    assert features.shape == (2, 10)
    np.testing.assert_allclose(features[:, :4], [[1.0, 2.0, 3.0, 4.0]] * 2)


def test_zero_initialized_residual_head_starts_as_alps() -> None:
    pytest.importorskip("torch")
    head = build_residual_head(3, hidden_dim=4, dropout=0.0)
    base = np.asarray([12.0, 7.0], dtype=np.float32)
    predicted, correction, terminal_probability = predict_residual_head(
        head,
        np.zeros((2, 3), dtype=np.float32),
        base,
        residual_scale=20.0,
        terminal_threshold=0.5,
        batch_size=2,
        device="cpu",
    )
    np.testing.assert_allclose(correction, 0.0)
    np.testing.assert_allclose(predicted, base)
    assert np.all(terminal_probability < 0.5)


def test_protocol_matches_implementation_and_forbids_old_test() -> None:
    path = Path("configs/experiments/alps_plp_hybrid_versions.json")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(protocol["methods"]) == METHOD_IDS
    assert protocol["data_policy"]["test_access"] == (
        "forbidden_until_a_new_hybrid_holdout_is_authored"
    )

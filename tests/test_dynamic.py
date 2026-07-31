import math

import pytest

from llm_length_prediction.data.schema import GenerationTrace, TracePoint
from llm_length_prediction.evaluation.progressive import (
    progress_breakdown,
    progressive_metrics,
)
from llm_length_prediction.models.dynamic import (
    PLP_FEATURE_NAMES,
    StandardizedMLPRemainingLength,
    build_progressive_samples,
)


def _trace() -> GenerationTrace:
    return GenerationTrace(
        prompt_id="p1",
        task="qa",
        prompt_tokens=12,
        output_tokens=20,
        temperature=0.7,
        seed=42,
        stop_reason="eos",
        points=[
            TracePoint(5, 2.0, 0.01, 15, entropy_mean=2.1, entropy_slope=-0.1),
            TracePoint(10, 1.8, 0.05, 10, entropy_mean=1.9, entropy_slope=-0.05),
            TracePoint(15, 1.2, 0.20, 5, entropy_mean=1.5, entropy_slope=-0.1),
            TracePoint(20, 0.2, 0.90, 0, entropy_mean=0.8, entropy_slope=-0.2),
        ],
    )


def test_progressive_samples_exclude_terminal_and_balance_sequence() -> None:
    samples = build_progressive_samples(_trace(), prompt_family_id="family-1")
    assert [sample.step for sample in samples] == [5, 10, 15]
    assert [sample.remaining_tokens for sample in samples] == [15, 10, 5]
    assert all(len(sample.features) == len(PLP_FEATURE_NAMES) for sample in samples)
    assert math.isclose(sum(sample.sequence_weight for sample in samples), 1.0)


def test_dynamic_model_round_trip_and_progress_metrics() -> None:
    model = StandardizedMLPRemainingLength(
        feature_names=PLP_FEATURE_NAMES,
        feature_mean=(0.0, 0.0, 0.0, 0.0, 0.0),
        feature_scale=(1.0, 1.0, 1.0, 1.0, 1.0),
        layer_weights=(((0.0, 0.0, 0.0, 0.0, 0.0),),),
        layer_biases=((math.log1p(10.0),),),
        residual_variance=0.0,
        hidden_sizes=(),
        dropout=0.0,
    )
    restored = StandardizedMLPRemainingLength.from_dict(model.to_dict())
    samples = build_progressive_samples(_trace(), prompt_family_id="family-1")
    features = [sample.features for sample in samples]
    predicted_mu = restored.predict_mu_many(features).tolist()
    predicted = restored.predict_remaining_many(features).tolist()

    assert all(math.isclose(value, 10.0) for value in predicted)
    metrics = progressive_metrics(samples, predicted, predicted_mu, 0.0)
    assert metrics["count"] == 3
    assert metrics["trace_count"] == 1
    assert len(progress_breakdown(samples, predicted, predicted_mu, 0.0)) == 3


def test_dynamic_model_rejects_malformed_layer_shapes() -> None:
    with pytest.raises(ValueError, match="wrong input dimension"):
        StandardizedMLPRemainingLength(
            feature_names=PLP_FEATURE_NAMES,
            feature_mean=(0.0, 0.0, 0.0, 0.0, 0.0),
            feature_scale=(1.0, 1.0, 1.0, 1.0, 1.0),
            layer_weights=(((0.0, 0.0),),),
            layer_biases=((0.0,),),
            residual_variance=0.0,
            hidden_sizes=(),
            dropout=0.0,
        )

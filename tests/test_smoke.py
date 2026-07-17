import math

from llm_length_prediction.data.schema import GenerationTrace, TracePoint
from llm_length_prediction.evaluation.metrics import mae, rmse, severe_underestimation_rate
from llm_length_prediction.models.dynamic import hybrid_total_length, scheduled_gamma
from llm_length_prediction.models.prior import LinearLogNormalPrior
from llm_length_prediction.serving.simulator import length_bucket


def test_prior_and_hybrid_prediction() -> None:
    prior = LinearLogNormalPrior(weights=(0.5, -0.25), bias=2.0, log_variance=0.2)
    predicted = prior.predict_mean_length((1.0, 2.0))
    assert math.isclose(predicted, math.exp(2.1))

    gamma = scheduled_gamma(step=64)
    assert math.isclose(gamma, 0.5)
    assert hybrid_total_length(predicted, 64, 40, gamma) > 0


def test_trace_and_metrics_contracts() -> None:
    trace = GenerationTrace(
        prompt_id="p1",
        task="qa",
        prompt_tokens=10,
        output_tokens=120,
        temperature=0.7,
        seed=42,
        stop_reason="eos",
        points=[TracePoint(20, 2.1, 0.05, 100)],
    )
    trace.validate()

    actual = [100.0, 200.0]
    predicted = [90.0, 50.0]
    assert mae(actual, predicted) == 80.0
    assert rmse(actual, predicted) > mae(actual, predicted)
    assert severe_underestimation_rate(actual, predicted, threshold=100.0) == 0.5


def test_length_buckets() -> None:
    assert length_bucket(50, (64, 256, 1024)) == 0
    assert length_bucket(200, (64, 256, 1024)) == 1
    assert length_bucket(4096, (64, 256, 1024)) == 3

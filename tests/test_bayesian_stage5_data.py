from __future__ import annotations

import math

import numpy as np

from llm_length_prediction.data.bayesian_trace import BayesianTraceV1
from llm_length_prediction.data.stage5 import (
    rolling_mean_slope,
    stage5_bayesian_sequence,
    stage5_hybrid_samples,
)
from llm_length_prediction.evaluation.sequential import run_bayesian_sequence
from llm_length_prediction.evaluation.stage5 import (
    compact_posterior_rows,
    paired_family_nll_difference,
    posterior_metrics,
)
from llm_length_prediction.models.bayesian_scorer import (
    SCALAR_METHOD_ID,
    build_likelihood_ratio_scorer,
    fit_scorer_standardization,
)


def _trace(
    prompt_id: str = "qa_001_short",
    family: str = "qa_001",
    temperature: float = 0.7,
) -> BayesianTraceV1:
    observed = 7
    hidden_size = 4
    steps = np.asarray([1, 5, 7], dtype=np.int32)
    return BayesianTraceV1(
        prompt_id=prompt_id,
        prompt_family_id=family,
        task="qa",
        intended_length="short",
        split="train",
        prompt_tokens=8,
        observed_tokens=observed,
        max_new_tokens=20,
        temperature=temperature,
        top_p=0.95,
        seed=42,
        stop_reason="eos",
        eos_token_ids=(3,),
        prior_feature=np.arange(hidden_size, dtype=np.float32),
        prompt_feature=np.arange(hidden_size, dtype=np.float32) + 1,
        initial_decode_hidden_state=np.zeros(hidden_size, dtype=np.float32),
        decode_hidden_states=np.stack(
            [np.full(hidden_size, step / observed, dtype=np.float32) for step in steps]
        ),
        saved_steps=steps,
        generated_token_ids=np.asarray([2, 2, 2, 2, 2, 2, 3], dtype=np.int32),
        token_entropies=np.asarray([3.0, 2.8, 2.4, 2.0, 1.5, 0.8, 0.1], dtype=np.float32),
        token_eos_probabilities=np.asarray(
            [0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 0.9], dtype=np.float32
        ),
        model_name="Qwen/Qwen2.5-7B-Instruct",
        model_revision="revision",
        tokenizer_revision="revision",
        duration_ms=10.0,
        metadata={},
    )


def test_stage5_views_preserve_temperature_terminal_and_non_overlapping_sequence() -> None:
    trace = _trace(temperature=1.0)
    samples = stage5_hybrid_samples(trace, entropy_window=3)
    sequence = stage5_bayesian_sequence(
        trace, prior_mu=math.log1p(7), prior_log_variance=0.2
    )
    assert [sample.step for sample in samples] == [1, 5, 7]
    assert [sample.remaining_tokens for sample in samples] == [6, 2, 0]
    assert {sample.temperature for sample in samples} == {1.0}
    assert sequence.sequence_id == (trace.prompt_id, 1.0, 42)
    assert [step.delta for step in sequence.steps] == [1, 4, 2]
    assert sequence.steps[-1].terminal_observed is True


def test_rolling_features_match_the_frozen_causal_prefix() -> None:
    mean, slope = rolling_mean_slope(np.asarray([1.0, 2.0, 3.0, 4.0]), window=3)
    assert mean == 3.0
    assert slope == 1.0


def test_compact_stage5_metrics_and_paired_selection_use_family_unit() -> None:
    sequences = [
        stage5_bayesian_sequence(
            _trace(prompt_id=f"p-{family}", family=family),
            prior_mu=math.log1p(7),
            prior_log_variance=0.2,
        )
        for family in ("a", "b")
    ]
    standardization = fit_scorer_standardization(sequences)
    scorer = build_likelihood_ratio_scorer(
        SCALAR_METHOD_ID,
        standardization=standardization,
        hidden_dim=8,
        dropout=0.0,
    )
    rows = []
    for sequence in sequences:
        rows.extend(
            compact_posterior_rows(
                SCALAR_METHOD_ID,
                run_bayesian_sequence(sequence, scorer),
                outer_fold=0,
            )
        )
    metrics = posterior_metrics(rows)
    paired = paired_family_nll_difference(
        rows, rows, replicates=100, confidence=0.95, seed=1
    )
    assert metrics["sequence_count"] == 2
    assert metrics["family_count"] == 2
    assert paired["estimate"] == 0.0
    assert paired["upper"] == 0.0

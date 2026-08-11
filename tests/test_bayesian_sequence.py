import numpy as np
import pytest

from llm_length_prediction.data.sequential import (
    SequentialRawTrace,
    build_synthetic_sequence,
    scheduled_update_steps,
)


def test_synthetic_sequence_uses_non_overlapping_frozen_schedule() -> None:
    sequence = build_synthetic_sequence(
        output_tokens=12,
        max_new_tokens=20,
        hidden_size=4,
    )
    assert [step.step for step in sequence.steps] == [1, 5, 10, 12]
    assert [step.delta for step in sequence.steps] == [1, 4, 5, 2]
    assert [step.true_remaining for step in sequence.steps] == [11, 7, 2, 0]
    assert [step.terminal_observed for step in sequence.steps] == [False, False, False, True]
    assert all(step.hidden_delta.shape == (4,) for step in sequence.steps)
    assert len(sequence.steps[-1].candidate_features) == 10


def test_censored_sequence_targets_survival_beyond_the_cap() -> None:
    sequence = build_synthetic_sequence(
        output_tokens=30,
        max_new_tokens=20,
        censored=True,
    )
    assert [step.step for step in sequence.steps] == [1, 5, 10, 15, 20]
    assert all(step.true_remaining is None for step in sequence.steps)
    assert sequence.steps[-1].censored_after_remaining == 0
    assert sequence.steps[-1].terminal_observed is False
    assert len(sequence.steps[-1].candidate_features) == 2
    assert sequence.steps[-1].candidate_features[-1, -1] == 1.0


def test_schedule_adds_non_stride_terminal_once() -> None:
    assert scheduled_update_steps(7, terminal_observed=True).tolist() == [1, 5, 7]
    assert scheduled_update_steps(5, terminal_observed=True).tolist() == [1, 5]
    assert scheduled_update_steps(7, terminal_observed=False).tolist() == [1, 5]


def test_trace_rejects_overlapping_or_unregistered_update_points() -> None:
    trace = SequentialRawTrace(
        prompt_id="p",
        prompt_family_id="f",
        task="qa",
        intended_length="short",
        temperature=0.7,
        seed=42,
        stop_reason="eos",
        observed_tokens=7,
        max_new_tokens=20,
        prior_mu=2.0,
        prior_log_variance=0.2,
        prior_mean_total_tokens=8.0,
        token_entropies=np.ones(7),
        token_eos_probabilities=np.linspace(0.01, 0.9, 7),
        saved_steps=np.asarray([1, 4, 7]),
    )
    with pytest.raises(ValueError, match="frozen update schedule"):
        trace.validate()

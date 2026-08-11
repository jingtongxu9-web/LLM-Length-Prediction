from pathlib import Path

import numpy as np

from llm_length_prediction.data.sequential import build_synthetic_sequence
from llm_length_prediction.models.bayesian_scorer import (
    HIDDEN_DELTA_METHOD_ID,
    SCALAR_METHOD_ID,
    bayesian_sequence_loss,
    build_likelihood_ratio_scorer,
    fit_bayesian_scorer,
    fit_scorer_standardization,
    load_bayesian_checkpoint,
    make_bayesian_checkpoint,
    restore_bayesian_scorer,
    save_bayesian_checkpoint,
    validate_bayesian_checkpoint,
)


def test_scalar_scorer_is_shared_across_candidate_lengths() -> None:
    import torch

    sequence = build_synthetic_sequence(output_tokens=8, max_new_tokens=12)
    standardization = fit_scorer_standardization([sequence])
    scorer = build_likelihood_ratio_scorer(
        SCALAR_METHOD_ID,
        standardization=standardization,
        hidden_dim=16,
        dropout=0.0,
    )
    step = sequence.steps[0]
    scores = scorer(
        torch.from_numpy(step.scalar_features[None, :]).float(),
        torch.from_numpy(step.candidate_features[None, :, :]).float(),
    )
    assert scores.shape == (1, len(step.candidate_features))
    assert torch.allclose(scores.mean(dim=-1), torch.zeros(1), atol=1e-6)


def test_hidden_projection_is_frozen_and_hidden_delta_changes_scores() -> None:
    import torch

    sequence = build_synthetic_sequence(
        output_tokens=8,
        max_new_tokens=12,
        hidden_size=6,
    )
    standardization = fit_scorer_standardization([sequence])
    scorer = build_likelihood_ratio_scorer(
        HIDDEN_DELTA_METHOD_ID,
        standardization=standardization,
        hidden_input_dim=6,
        hidden_projection_dim=4,
        hidden_dim=16,
        dropout=0.0,
    )
    parameter_names = {name for name, _ in scorer.named_parameters()}
    buffer_names = {name for name, _ in scorer.named_buffers()}
    assert "hidden_projection" not in parameter_names
    assert "hidden_projection" in buffer_names
    step = sequence.steps[0]
    evidence = torch.from_numpy(step.scalar_features[None, :]).float()
    candidates = torch.from_numpy(step.candidate_features[None, :, :]).float()
    first = scorer(evidence, candidates, torch.zeros(1, 6))
    second = scorer(evidence, candidates, torch.ones(1, 6))
    assert not torch.allclose(first, second)


def test_sequence_loss_is_finite_for_exact_and_censored_rollouts() -> None:
    exact = build_synthetic_sequence(output_tokens=8, max_new_tokens=12)
    censored = build_synthetic_sequence(
        output_tokens=18,
        max_new_tokens=12,
        censored=True,
    )
    standardization = fit_scorer_standardization([exact, censored])
    scorer = build_likelihood_ratio_scorer(
        SCALAR_METHOD_ID,
        standardization=standardization,
        hidden_dim=16,
        dropout=0.0,
    )
    loss, components = bayesian_sequence_loss(scorer, [exact, censored])
    loss.backward()
    assert np.isfinite(float(loss.detach().numpy()))
    assert set(components) == {
        "posterior_nll",
        "terminal_bce",
        "posterior_total_variation",
    }


def test_small_scorer_fit_and_checkpoint_round_trip(tmp_path: Path) -> None:
    import torch

    sequence = build_synthetic_sequence(output_tokens=6, max_new_tokens=10)
    scorer, report = fit_bayesian_scorer(
        [sequence],
        method_id=SCALAR_METHOD_ID,
        hidden_dim=16,
        dropout=0.0,
        epochs=2,
        device="cpu",
    )
    payload = make_bayesian_checkpoint(
        scorer,
        contract_sha256="0" * 64,
        training_report=report,
    )
    validate_bayesian_checkpoint(payload)
    checkpoint_path = save_bayesian_checkpoint(payload, tmp_path / "scorer.pt")
    loaded = load_bayesian_checkpoint(checkpoint_path)
    restored = restore_bayesian_scorer(loaded)
    step = sequence.steps[0]
    evidence = torch.from_numpy(step.scalar_features[None, :]).float()
    candidates = torch.from_numpy(step.candidate_features[None, :, :]).float()
    with torch.inference_mode():
        original = scorer(evidence, candidates)
        recovered = restored(evidence, candidates)
    assert torch.allclose(original, recovered)
    assert report["rollout_count"] == 1

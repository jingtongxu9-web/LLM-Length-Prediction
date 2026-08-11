import json
from copy import deepcopy
from pathlib import Path

import pytest

from llm_length_prediction.bayesian_contract import (
    BayesianContractError,
    load_bayesian_contract,
    validate_bayesian_contract,
)
from llm_length_prediction.experiment import file_sha256

CONTRACT_PATH = Path("configs/experiments/bayesian_sequential_v1.json")


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_loader_validates_the_frozen_file() -> None:
    contract = load_bayesian_contract(CONTRACT_PATH)
    assert contract["method_id"] == "bayesian-sequential-v1"


def test_contract_validator_rejects_online_parameter_updates() -> None:
    contract = deepcopy(_load_contract())
    contract["scientific_identity"]["updates_model_parameters_online"] = True
    with pytest.raises(BayesianContractError, match="online model-parameter"):
        validate_bayesian_contract(contract, repository_root=Path("."))


def test_authoritative_reference_is_pinned() -> None:
    contract = _load_contract()
    reference = contract["authoritative_reference"]
    reference_path = Path(reference["path"])
    assert reference_path.is_file()
    assert file_sha256(reference_path) == reference["sha256"]
    assert reference["sha256"] == (
        "b3c39e785a4a9217fd5d3c580b127c5a25093314f82bacb86f64f98feed02e62"
    )
    assert Path(reference["mathematical_contract"]).is_file()


def test_latent_state_and_transition_are_unambiguous() -> None:
    contract = _load_contract()
    latent = contract["latent_state"]
    updates = contract["updates"]
    assert latent["definition"] == "R_t = output_tokens - step"
    assert latent["support"] == "integer_tokens"
    assert latent["minimum"] == 0
    assert latent["terminal_state"] == 0
    assert latent["output_length_includes_eos"] is True
    assert updates["nominal_stride"] == 5
    assert updates["include_first_point"] is True
    assert updates["include_terminal_point"] is True
    assert "r + delta" in updates["transition_formula"]
    assert updates["posterior_operations"] == "log_space"
    assert updates["normalization"] == "logsumexp"


def test_prior_is_alps_with_oof_variance_calibration() -> None:
    prior = _load_contract()["prior"]
    assert prior["source"] == "alps_layer14_last_prompt_token"
    assert prior["target"] == "log1p_output_tokens"
    assert prior["distribution"] == "shifted_lognormal"
    assert prior["feature_layer_zero_based"] == 14
    assert prior["ridge"] == {"standardize": True, "alpha": 1.0}
    assert prior["variance_source"] == "family_grouped_oof_log1p_residual_mle"
    assert prior["training_features"] == "inner_family_grouped_cross_fitted"
    assert prior["validation_features"] == "fit_on_outer_train_families_only"
    assert prior["discretization"]["silent_tail_mass_drop_forbidden"] is True


def test_evidence_is_incremental_and_cannot_double_count_history() -> None:
    evidence = _load_contract()["evidence"]
    assert evidence["unit"] == "non_overlapping_new_token_block_since_previous_update"
    assert evidence["overlapping_windows_as_independent_likelihood_forbidden"] is True
    assert evidence["full_causal_hidden_state_repeated_multiplication_forbidden"] is True
    assert "terminal_observed" in evidence["block_features"]
    assert "delta_entropy" in evidence["block_features"]
    assert "delta_eos_probability" in evidence["block_features"]


def test_contract_distinguishes_parameters_from_request_posterior() -> None:
    contract = _load_contract()
    identity = contract["scientific_identity"]
    training = contract["training"]
    assert identity["updates_model_parameters_online"] is False
    assert identity["updates_request_posterior_online"] is True
    assert training["online_parameter_updates"] is False
    assert contract["likelihood_ratio_head"]["direct_point_regression"] is False
    assert contract["likelihood_ratio_head"]["direct_posterior_without_prior"] is False
    assert contract["posterior_update"]["derive_discrete_hazard_from_posterior"] is True


def test_only_two_predeclared_bayesian_candidates_can_be_selected() -> None:
    contract = _load_contract()
    assert set(contract["candidate_models"]) == {
        "bayesian_entropy_scalar_v1",
        "bayesian_entropy_hidden_delta_v1",
    }
    selection = contract["selection"]
    assert selection["select_hidden_only_if_paired_nll_ci_entirely_below_zero"] is True
    assert selection["otherwise_select"] == "bayesian_entropy_scalar_v1"
    assert selection["final_holdout_selects_nothing"] is True


def test_final_holdout_is_still_forbidden() -> None:
    contract = _load_contract()
    data_policy = contract["data_policy"]
    assert contract["status"] == "phase1_frozen_preimplementation"
    assert data_policy["new_final_holdout_required"] is True
    assert data_policy["new_final_holdout_status"] == "not_authored_not_opened"
    assert data_policy["old_alps_plp_test_role"] == "historical_evidence_only"
    assert data_policy["same_family_all_tasks_lengths_seeds_temperatures_and_steps_same_fold"]


def test_metrics_and_temperature_robustness_are_predeclared() -> None:
    contract = _load_contract()
    evaluation = contract["evaluation"]
    generation = contract["generation"]
    assert evaluation["model_selection_primary_metric"] == (
        "family_macro_sequence_balanced_posterior_nll"
    )
    assert evaluation["convergence_metric"] == {
        "name": "stable_time_to_relative_error_5pct",
        "threshold": 0.05,
        "requires_all_later_saved_points_within_threshold": True,
        "never_reached": "censored_failure",
    }
    assert evaluation["uncertainty_cone_quantiles"] == [0.025, 0.5, 0.975]
    assert generation["primary_temperature"] == 0.7
    assert generation["robustness_temperatures"] == [0.3, 1.0]
    assert generation["temperature_is_model_input"] is False
    assert generation["robustness_refit_forbidden"] is True


def test_censored_rollouts_cannot_become_terminal_zero() -> None:
    censoring = _load_contract()["censoring"]
    assert censoring["type"] == "right_censored"
    assert censoring["impute_unknown_remaining_as_zero"] is False
    assert censoring["silent_drop_forbidden"] is True
    assert censoring["primary_likelihood"] == "right_censored_survival_contribution"

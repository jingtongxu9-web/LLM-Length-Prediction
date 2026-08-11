from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_length_prediction.experiment import file_sha256


class BayesianContractError(ValueError):
    """Raised when the frozen Bayesian Sequential contract is inconsistent."""


EXPECTED_CANDIDATES = {
    "bayesian_entropy_scalar_v1",
    "bayesian_entropy_hidden_delta_v1",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BayesianContractError(message)


def validate_bayesian_contract(
    contract: dict[str, Any], *, repository_root: Path
) -> None:
    _require(contract.get("schema_version") == 1, "unsupported schema_version")
    _require(
        contract.get("contract_kind") == "preimplementation_scientific_contract",
        "contract_kind must remain preimplementation_scientific_contract",
    )
    _require(contract.get("method_id") == "bayesian-sequential-v1", "wrong method_id")
    _require(
        contract.get("status")
        in {"phase1_frozen_preimplementation", "phase1_approved_for_implementation"},
        "unsupported phase-one status",
    )

    reference = contract["authoritative_reference"]
    reference_path = repository_root / reference["path"]
    mathematical_contract = repository_root / reference["mathematical_contract"]
    _require(reference_path.is_file(), f"missing authoritative reference: {reference_path}")
    _require(
        file_sha256(reference_path) == reference["sha256"],
        "authoritative reference SHA-256 does not match",
    )
    _require(mathematical_contract.is_file(), "missing mathematical contract")

    identity = contract["scientific_identity"]
    _require(identity["role"] == "proposed_method", "Bayesian method role changed")
    _require(identity["qwen_weights_frozen"] is True, "Qwen weights must stay frozen")
    _require(
        identity["updates_model_parameters_online"] is False,
        "online model-parameter updates are forbidden",
    )
    _require(
        identity["updates_request_posterior_online"] is True,
        "request posterior must update online",
    )

    latent = contract["latent_state"]
    _require(latent["definition"] == "R_t = output_tokens - step", "latent definition changed")
    _require(latent["support"] == "integer_tokens", "latent support must be integer tokens")
    _require(latent["minimum"] == 0, "remaining length cannot be negative")
    _require(latent["terminal_state"] == 0, "terminal remaining length must be zero")
    _require(latent["output_length_includes_eos"] is True, "length must include EOS")

    prior = contract["prior"]
    _require(prior["target"] == "log1p_output_tokens", "prior target must be log1p length")
    _require(prior["distribution"] == "shifted_lognormal", "prior distribution changed")
    _require(prior["feature_layer_zero_based"] == 14, "ALPS prior layer must remain 14")
    _require(prior["ridge"] == {"standardize": True, "alpha": 1.0}, "Ridge changed")
    _require(
        prior["variance_source"] == "family_grouped_oof_log1p_residual_mle",
        "prior variance must use OOF residuals",
    )
    _require(
        prior["training_features"] == "inner_family_grouped_cross_fitted",
        "training prior features must be cross-fitted",
    )
    _require(
        prior["validation_features"] == "fit_on_outer_train_families_only",
        "validation prior leaked outside outer train",
    )

    updates = contract["updates"]
    _require(updates["nominal_stride"] == 5, "Bayesian update stride changed")
    _require(updates["include_first_point"] is True, "first update point is required")
    _require(updates["include_terminal_point"] is True, "terminal update point is required")
    _require(updates["posterior_operations"] == "log_space", "posterior must use log space")
    _require(updates["normalization"] == "logsumexp", "posterior must use logsumexp")

    evidence = contract["evidence"]
    _require(
        evidence["unit"] == "non_overlapping_new_token_block_since_previous_update",
        "evidence must be non-overlapping",
    )
    _require(
        evidence["overlapping_windows_as_independent_likelihood_forbidden"] is True,
        "overlapping evidence likelihood is forbidden",
    )
    _require(
        evidence["full_causal_hidden_state_repeated_multiplication_forbidden"] is True,
        "repeated cumulative hidden-state multiplication is forbidden",
    )

    _require(
        set(contract["candidate_models"]) == EXPECTED_CANDIDATES,
        "only the two preregistered Bayesian candidates are allowed",
    )
    likelihood_head = contract["likelihood_ratio_head"]
    _require(likelihood_head["direct_point_regression"] is False, "point regression is not Bayes")
    _require(
        likelihood_head["direct_posterior_without_prior"] is False,
        "posterior cannot ignore the ALPS prior",
    )

    posterior = contract["posterior_update"]
    _require(posterior["nonnegative_support_only"] is True, "negative support is forbidden")
    _require(
        posterior["mask_total_length_less_than_observed_step"] is True,
        "impossible total lengths must be masked",
    )
    _require(
        0.0 < float(posterior["probability_sum_tolerance"]) <= 1e-4,
        "invalid posterior probability tolerance",
    )
    _require(
        posterior["derive_discrete_hazard_from_posterior"] is True,
        "v1 hazard must be derived from posterior",
    )

    training = contract["training"]
    _require(training["online_parameter_updates"] is False, "online training is forbidden")
    _require(training["soft_labels_enabled"] is False, "soft labels require a new method ID")

    generation = contract["generation"]
    _require(generation["primary_temperature"] == 0.7, "primary temperature changed")
    _require(
        generation["primary_temperature"] not in generation["robustness_temperatures"],
        "primary temperature must not be duplicated in robustness temperatures",
    )
    _require(generation["temperature_is_model_input"] is False, "temperature input needs new ID")
    _require(generation["robustness_refit_forbidden"] is True, "robustness refit is forbidden")

    data_policy = contract["data_policy"]
    _require(data_policy["group_unit"] == "prompt_family_id", "family must be group unit")
    _require(
        data_policy["same_family_all_tasks_lengths_seeds_temperatures_and_steps_same_fold"]
        is True,
        "family members must stay in one fold",
    )
    _require(data_policy["new_final_holdout_required"] is True, "new final holdout is required")
    _require(
        data_policy["new_final_holdout_status"] == "not_authored_not_opened",
        "final holdout must remain unopened in phase one",
    )
    _require(
        data_policy["old_alps_plp_test_role"] == "historical_evidence_only",
        "old Test cannot select Bayesian methods",
    )

    censoring = contract["censoring"]
    _require(censoring["type"] == "right_censored", "max-token stops are right censored")
    _require(censoring["impute_unknown_remaining_as_zero"] is False, "censoring cannot be zero")
    _require(censoring["silent_drop_forbidden"] is True, "censored rows cannot be silently dropped")

    selection = contract["selection"]
    _require(selection["final_holdout_selects_nothing"] is True, "final Test cannot select")
    _require(
        selection["otherwise_select"] == "bayesian_entropy_scalar_v1",
        "selection fallback must remain the simpler candidate",
    )


def load_bayesian_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    repository_root = path.resolve().parents[2]
    validate_bayesian_contract(contract, repository_root=repository_root)
    return contract

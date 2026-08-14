from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage8_benchmark import (
    _convergence_with_censoring,
    _sequence_point_metrics,
)
from llm_length_prediction.stage8_freeze import (
    MODEL_FILES,
    build_checkpoint_registry,
    final_holdout_gate_report,
    load_stage8a_config,
    validate_checkpoint_registry,
)
from llm_length_prediction.stage8_holdout import load_final_holdout_contract

CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")
REPORT_SCHEMA = Path("configs/reports/bayesian_sequential_stage8_final_report_schema.json")


def test_stage8a_freezes_selection_training_and_unopened_holdout() -> None:
    config = load_stage8a_config(CONFIG)
    holdout = config["final_holdout_plan"]
    assert config["stage5"]["selected_method"] == "bayesian_entropy_scalar_v1"
    assert config["final_training"]["temperature"] == 0.7
    assert config["final_training"]["trace_count"] == 540
    assert config["final_training"]["family_count"] == 60
    assert config["final_training"]["final_holdout_access"] == "forbidden"
    assert holdout["status"] == "not_authored_not_opened"
    assert holdout["new_family_count"] == 12
    assert holdout["prompt_count"] == 36
    assert holdout["expected_rollout_count"] == 324
    assert config["benchmark"]["final_holdout_selects_nothing"] is True


def test_stage8_report_schema_forbids_final_selection_and_tuning() -> None:
    schema = json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert properties["primary_method"]["const"] == "bayesian_entropy_scalar_v1"
    assert properties["model_selection_performed"]["const"] is False
    assert properties["threshold_tuning_performed"]["const"] is False
    assert properties["final_holdout_selects_nothing"]["const"] is True
    assert properties["comparison_list"]["minItems"] == 7
    assert properties["comparison_list"]["maxItems"] == 7


def test_final_holdout_gate_fails_closed_before_models_and_lock(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["holdout_gate"]["benchmark_lock"] = str(tmp_path / "absent-lock.json")
    config_path = tmp_path / "stage8a.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    report = final_holdout_gate_report(config_path, model_root=tmp_path / "models")
    assert report["ready"] is False
    assert report["status"] == "blocked"
    assert report["final_holdout_opened"] is False
    assert any("registry" in failure for failure in report["failures"])
    assert any("lock" in failure for failure in report["failures"])


def test_checkpoint_registry_detects_any_final_model_change(tmp_path: Path) -> None:
    config = load_stage8a_config(CONFIG)
    config["_config_sha256"] = file_sha256(CONFIG)
    for name in MODEL_FILES:
        (tmp_path / name).write_bytes(f"frozen:{name}".encode())
    report_name = config["outputs"]["training_report"]
    (tmp_path / report_name).write_text('{"status":"pass"}\n', encoding="utf-8")
    registry = build_checkpoint_registry(config, output_dir=tmp_path)
    registry_path = tmp_path / config["outputs"]["checkpoint_registry"]
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    assert validate_checkpoint_registry(config, output_dir=tmp_path) == registry
    (tmp_path / MODEL_FILES[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest changed"):
        validate_checkpoint_registry(config, output_dir=tmp_path)


def test_final_holdout_loader_does_not_read_manifest_before_ready_lock(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["holdout_gate"]["benchmark_lock"] = str(tmp_path / "absent-lock.json")
    config_path = tmp_path / "stage8a.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must remain unopened"):
        load_final_holdout_contract(config_path)


def test_stage8b_template_is_deliberately_not_ready() -> None:
    template = json.loads(
        Path("configs/experiments/bayesian_sequential_stage8b_lock_v1.template.json").read_text(
            encoding="utf-8"
        )
    )
    assert template["status"] == "template_not_ready_do_not_open_holdout"
    assert template["prompt_semantic_overlap_review_complete"] is False
    assert template["final_holdout_opened"] is False
    assert template["final_holdout_selects_nothing"] is True


def test_final_point_metrics_exclude_censoring_instead_of_imputing_zero() -> None:
    rows = [
        {
            "prompt_id": "exact",
            "prompt_family_id": "family-exact",
            "temperature": 0.7,
            "seed": 42,
            "true_remaining": 10,
            "prediction": 12.0,
        },
        {
            "prompt_id": "censored",
            "prompt_family_id": "family-censored",
            "temperature": 0.7,
            "seed": 42,
            "true_remaining": None,
            "prediction": 0.0,
        },
    ]
    metrics = _sequence_point_metrics(rows, "prediction")
    assert metrics["exact_sequence_count"] == 1
    assert metrics["censored_sequence_count"] == 1
    assert metrics["sequence_balanced_mae_tokens"] == 2.0
    assert metrics["censored_point_targets_excluded_not_imputed"] is True


def test_final_convergence_counts_right_censoring_as_failure() -> None:
    exact = [
        {
            "prompt_id": "exact",
            "temperature": 0.7,
            "seed": 42,
            "step": 1,
            "true_remaining": 9,
            "predicted_remaining": 9.0,
        }
    ]
    censored = {
        "prompt_id": "censored",
        "temperature": 0.7,
        "seed": 42,
        "step": 1,
        "true_remaining": None,
        "predicted_remaining": 100.0,
    }
    report = _convergence_with_censoring(
        exact,
        [*exact, censored],
        group=lambda row: "all",
    )["all"]
    assert report["sequence_count"] == 2
    assert report["success_count"] == 1
    assert report["failure_count"] == 1
    assert report["success_rate"] == 0.5
    assert report["right_censored_counted_as_failure"] == 1

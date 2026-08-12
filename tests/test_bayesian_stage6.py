from __future__ import annotations

from pathlib import Path

from llm_length_prediction.stage6_analysis import (
    Stage6Sources,
    convergence_metrics,
    load_stage6_config,
    serving_replay,
    uncertainty_curve_rows,
    uncertainty_findings,
)

CONFIG = Path("configs/experiments/bayesian_sequential_stage6_analysis_v1.json")


def _row(
    *,
    prompt: str,
    family: str,
    step: int,
    remaining: int,
    predicted: float,
    temperature: float = 0.7,
) -> dict[str, object]:
    return {
        "method_id": "bayesian_entropy_scalar_v1",
        "prompt_id": prompt,
        "prompt_family_id": family,
        "task": "qa",
        "intended_length": "long",
        "temperature": temperature,
        "seed": 42,
        "outer_fold": 0,
        "step": step,
        "true_remaining": remaining,
        "predicted_remaining": predicted,
        "posterior_nll": 1.0,
        "crps": 2.0,
        "error_tokens": predicted - remaining,
        "posterior_variance_lower_bound": float(remaining + 1),
        "posterior_entropy": 1.0,
        "interval_50_coverage": 1.0,
        "interval_50_width": 10.0,
        "interval_90_coverage": 1.0,
        "interval_90_width": 20.0,
        "interval_95_coverage": 1.0,
        "interval_95_width": 30.0,
        "update_wall_time_ms": 1.0,
        "predictor_state_bytes": 100,
    }


def test_stage6_config_freezes_no_refit_and_kv_bytes() -> None:
    config = load_stage6_config(CONFIG)
    assert config["data_policy"]["model_refit"] is False
    assert config["data_policy"]["threshold_tuning"] is False
    assert "forbidden" in config["data_policy"]["new_final_holdout_access"]
    kv = config["serving_replay"]["kv_model_contract"]
    assert kv["bytes_per_output_token"] == 57344
    assert kv["source_url"].endswith("/config.json")
    assert config["stage5"]["selected_method"] == "bayesian_entropy_scalar_v1"


def test_uncertainty_curves_balance_sequences_before_progress_groups() -> None:
    config = load_stage6_config(CONFIG)
    first = _row(prompt="p1", family="f1", step=1, remaining=99, predicted=99.0)
    second = _row(prompt="p2", family="f2", step=1, remaining=99, predicted=109.0)
    repeated = [dict(first, step=step, true_remaining=100 - step) for step in (2, 3, 4)]
    rows = [first, second, *repeated]
    curves = uncertainty_curve_rows({"bayesian_entropy_scalar_v1": rows}, config)
    early = next(
        row
        for row in curves
        if row["temperature"] == 0.7 and row["progress_bin"] == "0-10%"
    )
    # p1 has four saved points and p2 one; sequence balancing gives (0 + 10) / 2.
    assert early["sequence_balanced_absolute_error_tokens"] == 5.0
    assert early["sequence_count"] == 2


def test_stable_time_requires_every_later_saved_point() -> None:
    rows = [
        _row(prompt="p", family="f", step=10, remaining=90, predicted=90.0),
        _row(prompt="p", family="f", step=20, remaining=80, predicted=90.0),
        _row(prompt="p", family="f", step=50, remaining=50, predicted=50.0),
        _row(prompt="p", family="f", step=100, remaining=0, predicted=0.0),
    ]
    metrics = convergence_metrics(rows, threshold=0.05, group=lambda row: "all")["all"]
    assert metrics["success_rate"] == 1.0
    assert metrics["stable_step_mean_on_success"] == 50.0


def test_uncertainty_findings_do_not_equate_variance_drop_with_success() -> None:
    config = load_stage6_config(CONFIG)
    rows = []
    for index, label in enumerate(config["analysis"]["progress_bin_labels"]):
        rows.append(
            {
                "method_id": "bayesian_entropy_scalar_v1",
                "temperature": 0.7,
                "progress_bin": label,
                "sequence_balanced_posterior_variance_lower_bound": 100.0 - index,
                "sequence_balanced_posterior_entropy": 5.0 - index * 0.1,
                "sequence_balanced_interval_50_coverage": 0.5,
                "sequence_balanced_interval_90_coverage": 0.7,
                "sequence_balanced_interval_95_coverage": 0.8,
            }
        )
    findings = uncertainty_findings(rows, config)
    assert findings["variance_decreased_first_to_last"] is True
    assert findings["coverage"]["90"]["all_progress_bins_within_tolerance"] is False
    assert findings["uncertainty_success_requires_joint_width_and_coverage_interpretation"]


def test_serving_replay_uses_real_duration_and_q975_allocation(tmp_path: Path) -> None:
    config = load_stage6_config(CONFIG)
    identities = [("p1", 0.7, 42), ("p2", 0.7, 42)]
    stage4_rows = tuple(
        {
            "prompt_id": prompt,
            "temperature": temperature,
            "seed": seed,
            "observed_tokens": actual,
            "duration_ms": duration,
        }
        for (prompt, temperature, seed), actual, duration in zip(
            identities, (100, 200), (10.0, 20.0), strict=True
        )
    )
    sources = Stage6Sources(
        config=config,
        stage4_root=tmp_path,
        stage5_root=tmp_path,
        stage5_report={},
        selection={},
        stage4_rows=stage4_rows,
    )
    scalar = [
        _row(prompt="p1", family="f1", step=1, remaining=99, predicted=49.0),
        _row(prompt="p2", family="f2", step=1, remaining=199, predicted=99.0),
    ]
    alps = [dict(row, method_id="alps_countdown") for row in scalar]
    baselines = [
        {
            **row,
            "plp_terminal_zero_v3": row["predicted_remaining"],
            "alps_plp_concat_v1": row["predicted_remaining"],
        }
        for row in scalar
    ]
    cone = [
        {
            **row,
            "posterior_q975_remaining": true_remaining + 20,
        }
        for row, true_remaining in zip(scalar, (99, 199), strict=True)
    ]
    report = serving_replay(
        sources=sources,
        posterior_methods={
            "alps_countdown": alps,
            "bayesian_entropy_scalar_v1": scalar,
        },
        baseline_rows=baselines,
        cone_rows=cone,
    )
    mean = report["metrics"]["bayesian_entropy_scalar_v1_mean"]
    q975 = report["metrics"]["bayesian_entropy_scalar_v1_q975"]
    assert mean["underallocation_rate"] == 1.0
    assert q975["underallocation_rate"] == 0.0
    assert q975["kv_overreservation_bytes"] > 0
    assert report["serving_superiority_claimed"] is False

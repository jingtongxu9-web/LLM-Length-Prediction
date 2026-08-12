from __future__ import annotations

from pathlib import Path

from llm_length_prediction.stage7_error_feedback import (
    _direction_changes,
    _repetition_metrics,
    build_review_queue,
    load_stage7_config,
    summarize_audit,
)

CONFIG = Path("configs/experiments/bayesian_sequential_stage7_error_feedback_v1.json")


def _row(*, absolute: bool, worst: bool, rebound: bool = False) -> dict[str, object]:
    labels = {
        "entropy_rebound": rebound,
        "entropy_oscillation": False,
        "sampling_divergence": False,
        "repetition": False,
        "early_stop": False,
        "posterior_variance_increase": False,
        "posterior_premature_collapse": False,
        "posterior_oscillation": False,
    }
    return {
        "prompt_id": "qa_001_x_long",
        "prompt_family_id": "qa_001_x",
        "task": "qa",
        "intended_length": "long",
        "temperature": 0.7,
        "seed": 42,
        "outer_fold": 0,
        "sequence_mae_tokens": 120.0,
        "sequence_bias_tokens": -110.0,
        "maximum_absolute_error_tokens": 150.0,
        "maximum_underestimation_tokens": 150.0,
        "maximum_overestimation_tokens": 10.0,
        "absolute_error_cohort": absolute,
        "worst_fraction_cohort": worst,
        "automatic_labels": labels,
    }


def test_stage7_config_preserves_oof_and_holdout_boundaries() -> None:
    config = load_stage7_config(CONFIG)
    assert config["data_policy"]["split"] == "train_family_grouped_oof_only"
    assert config["data_policy"]["model_refit"] is False
    assert config["data_policy"]["changes_require_new_method_id"] is True
    assert "forbidden" in config["data_policy"]["new_final_holdout_access"]
    assert config["cohorts"]["absolute_error_threshold_tokens"] == 100.0
    assert config["cohorts"]["worst_fraction"] == 0.05


def test_direction_change_ignores_plateaus() -> None:
    assert _direction_changes([1.0, 2.0, 2.0, 1.0, 3.0]) == 2
    assert _direction_changes([1.0, 1.0, 1.0]) == 0


def test_repetition_metrics_capture_repeated_ngram_and_token_run() -> None:
    import numpy as np

    repeated_fraction, longest = _repetition_metrics(
        np.asarray([1, 2, 3, 4, 1, 2, 3, 4, 9, 9, 9]), ngram_size=4
    )
    assert repeated_fraction > 0
    assert longest == 3


def test_review_queue_is_union_and_keeps_manual_labels_unresolved() -> None:
    queue = build_review_queue([_row(absolute=True, worst=False), _row(absolute=False, worst=True)])
    assert len(queue) == 2
    assert queue[0]["manual_review_required"] == [
        "open_ended_prompt",
        "hallucination",
    ]


def test_summary_does_not_treat_manual_labels_as_negative() -> None:
    report = summarize_audit([_row(absolute=True, worst=True, rebound=True)])
    assert report["union_review_queue"]["automatic_label_counts"]["entropy_rebound"] == 1
    assert report["union_review_queue"]["negative_bias_sequence_rate"] == 1.0
    assert report["semantic_label_status"]["hallucination"].startswith("unresolved")

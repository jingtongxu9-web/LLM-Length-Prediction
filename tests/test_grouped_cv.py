import numpy as np

from llm_length_prediction.evaluation.grouped_cv import (
    DiagnosticRow,
    cross_validate,
    grouped_folds,
    select_families,
)


def make_rows() -> list[DiagnosticRow]:
    rows = []
    for family_index in range(10):
        for length_index, intended_length in enumerate(("short", "medium", "long")):
            prompt_id = f"f{family_index}-{intended_length}"
            for seed in range(3):
                rows.append(
                    DiagnosticRow(
                        prompt_id=prompt_id,
                        prompt_family_id=f"family-{family_index}",
                        task_type=("qa", "code")[family_index % 2],
                        intended_length=intended_length,
                        prompt_tokens=10 + family_index,
                        output_tokens=20 + 50 * length_index + family_index + seed,
                        hidden_state=(
                            float(length_index),
                            float(family_index),
                            float(length_index * 2 + family_index),
                        ),
                    )
                )
    return rows


def test_grouped_folds_never_split_a_family() -> None:
    rows = make_rows()
    groups = [row.prompt_family_id for row in rows]
    for train, validation in grouped_folds(groups, 5):
        train_groups = {groups[index] for index in train}
        validation_groups = {groups[index] for index in validation}
        assert train_groups.isdisjoint(validation_groups)
        assert len(validation) > 0


def test_cross_validation_produces_finite_oof_metrics() -> None:
    result = cross_validate(make_rows(), model_name="alps_hidden", alpha=1.0, n_splits=5)
    assert result["family_count"] == 10
    assert len(result["predictions"]) == 90
    assert np.isfinite(result["rollout_metrics"]["rmse_log1p"])
    assert 0.0 <= result["rollout_metrics"]["interval_95_coverage"] <= 1.0


def test_all_baselines_run() -> None:
    rows = make_rows()
    for model in ("global_mean", "prompt_tokens", "metadata", "metadata_prompt_tokens"):
        result = cross_validate(rows, model_name=model, alpha=1.0, n_splits=5)
        assert result["model"] == model


def test_learning_curve_sampling_keeps_whole_families() -> None:
    rows = make_rows()
    subset = select_families(rows, 0.5, repeat=2)
    selected = {row.prompt_family_id for row in subset}
    assert len(selected) == 5
    for family in selected:
        assert sum(row.prompt_family_id == family for row in subset) == 9

import math

from llm_length_prediction.evaluation.breakdown import (
    build_group_breakdowns,
    build_matched_length_analysis,
    build_prompt_mean_point_analysis,
    build_seed_stability,
    flatten_group_breakdowns,
    flatten_matched_length_analysis,
    flatten_prompt_mean_point_analysis,
    render_markdown_report,
)


def _rows() -> list[dict]:
    rows = []
    targets = {
        "short": (10, 12.0, 0, 20),
        "medium": (100, 90.0, 20, 150),
        "long": (300, 270.0, 150, 500),
    }
    for task_index, task in enumerate(("qa", "summarization", "code")):
        for seed in (42, 43):
            for length, (actual, predicted, minimum, maximum) in targets.items():
                adjusted_actual = actual + task_index * 2 + seed - 42
                adjusted_predicted = predicted + task_index * 2
                rows.append(
                    {
                        "prompt_id": f"{task}_001_{length}",
                        "prompt_family_id": f"{task}_001",
                        "task_type": task,
                        "intended_length": length,
                        "intended_output_min": minimum,
                        "intended_output_max": maximum,
                        "seed": seed,
                        "actual_output_tokens": adjusted_actual,
                        "predicted_log1p_mu": math.log1p(adjusted_predicted),
                        "predicted_mean_output_tokens": adjusted_predicted,
                    }
                )
    return rows


def test_group_breakdowns_cover_length_task_interaction_and_seed() -> None:
    report = build_group_breakdowns(_rows(), residual_variance=0.1)

    assert report["overall"]["count"] == 18
    assert report["overall"]["unique_family_count"] == 3
    assert set(report["by_intended_length"]) == {"short", "medium", "long"}
    assert report["by_intended_length"]["short"]["count"] == 6
    assert set(report["by_task_type"]) == {"qa", "summarization", "code"}
    assert report["by_task_type"]["qa"]["count"] == 6
    assert len(report["by_task_type_and_intended_length"]) == 9
    assert report["by_task_type_and_intended_length"]["code/long"]["count"] == 2
    assert set(report["by_seed"]) == {"42", "43"}
    assert len(flatten_group_breakdowns(report)) == 1 + 3 + 3 + 9 + 2


def test_prompt_mean_analysis_averages_seed_outcomes_before_point_metrics() -> None:
    report = build_prompt_mean_point_analysis(_rows())

    assert report["seed_counts_per_prompt"] == [2]
    assert report["overall"]["count"] == 9
    assert report["overall"]["unique_family_count"] == 3
    assert report["by_intended_length"]["short"]["count"] == 3
    assert report["by_task_type"]["qa"]["count"] == 3
    assert len(report["by_task_type_and_intended_length"]) == 9
    assert len(flatten_prompt_mean_point_analysis(report)) == 1 + 3 + 3 + 9


def test_matched_length_analysis_reports_monotonicity_and_contrasts() -> None:
    report = build_matched_length_analysis(_rows())
    rollout = report["rollout_level"]["overall"]
    family_mean = report["family_mean_level"]["overall"]

    assert rollout["triplet_count"] == 6
    assert rollout["actual_strict_monotonic_rate"] == 1.0
    assert rollout["predicted_strict_monotonic_rate"] == 1.0
    assert rollout["contrasts"]["short_to_long"]["actual_delta_mean_tokens"] == 290.0
    assert family_mean["triplet_count"] == 3
    assert len(flatten_matched_length_analysis(report)) == 2 * 4 * 3


def test_seed_stability_distinguishes_generation_and_prefill_prediction() -> None:
    report = build_seed_stability(_rows())

    assert report["prompt_count"] == 9
    assert report["mean_actual_seed_std_tokens"] == 0.5
    assert report["max_prediction_seed_span_tokens"] == 0.0


def test_markdown_report_contains_all_primary_analysis_sections() -> None:
    rows = _rows()
    report = {
        "experiment_id": "test-experiment",
        "split": "test",
        "source_predictions": "test_evaluation.csv",
        **build_group_breakdowns(rows, residual_variance=0.1),
        "prompt_mean_point_analysis": build_prompt_mean_point_analysis(rows),
        "seed_stability": build_seed_stability(rows),
        "matched_length_analysis": build_matched_length_analysis(rows),
    }

    markdown = render_markdown_report(report)
    assert "## By intended length" in markdown
    assert "## By task type" in markdown
    assert "## Task type × intended length" in markdown
    assert "## Prompt-mean point accuracy: overall" in markdown
    assert "## Matched Short → Medium → Long" in markdown
    assert "| code/long |" in markdown

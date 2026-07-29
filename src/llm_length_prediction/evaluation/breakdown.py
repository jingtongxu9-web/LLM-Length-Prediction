from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from llm_length_prediction.evaluation.metrics import (
    log1p_prior_metrics,
    severe_underestimation_rate,
)

LENGTH_ORDER = ("short", "medium", "long")
TASK_ORDER = ("qa", "summarization", "code")


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.square(actual - actual.mean()).sum())
    return (
        0.0
        if denominator == 0.0
        else 1.0 - float(np.square(actual - predicted).sum()) / denominator
    )


def _point_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    if not rows:
        raise ValueError("at least one point-evaluation row is required")

    actual = np.asarray([float(row["actual_output_tokens"]) for row in rows], dtype=np.float64)
    predicted = np.asarray(
        [float(row["predicted_mean_output_tokens"]) for row in rows], dtype=np.float64
    )
    errors = predicted - actual
    actual_log = np.log1p(actual)
    predicted_log = np.log1p(np.maximum(0.0, predicted))
    if len(rows) < 2 or actual.std() == 0.0 or predicted.std() == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])

    return {
        "count": len(rows),
        "unique_prompt_count": len({str(row["prompt_id"]) for row in rows}),
        "unique_family_count": len({str(row["prompt_family_id"]) for row in rows}),
        "actual_mean_tokens": float(actual.mean()),
        "actual_median_tokens": float(np.median(actual)),
        "actual_std_tokens": float(actual.std()),
        "actual_min_tokens": float(actual.min()),
        "actual_max_tokens": float(actual.max()),
        "predicted_mean_tokens": float(predicted.mean()),
        "predicted_median_tokens": float(np.median(predicted)),
        "bias_tokens": float(errors.mean()),
        "mae_tokens": float(np.abs(errors).mean()),
        "median_absolute_error_tokens": float(np.median(np.abs(errors))),
        "rmse_tokens": float(np.sqrt(np.square(errors).mean())),
        "r_squared_tokens": _r_squared(actual, predicted),
        "r_squared_log1p_point": _r_squared(actual_log, predicted_log),
        "pearson_correlation_tokens": correlation,
        "underprediction_rate": float(np.mean(errors < 0.0)),
        "overprediction_rate": float(np.mean(errors > 0.0)),
    }


def summarize_observations(
    rows: Sequence[Mapping[str, Any]],
    residual_variance: float,
) -> dict[str, int | float]:
    """Summarize rollout-level point accuracy, calibration, and diagnostics."""

    if not rows:
        raise ValueError("at least one evaluation row is required")

    actual = np.asarray([int(row["actual_output_tokens"]) for row in rows], dtype=np.float64)
    predicted = np.asarray(
        [float(row["predicted_mean_output_tokens"]) for row in rows], dtype=np.float64
    )
    mus = np.asarray([float(row["predicted_log1p_mu"]) for row in rows], dtype=np.float64)
    signed_errors = predicted - actual
    absolute_errors = np.abs(signed_errors)

    radius = 1.959963984540054 * math.sqrt(max(residual_variance, 1e-12))
    lower = np.maximum(0.0, np.expm1(mus - radius))
    upper = np.expm1(mus + radius)
    in_intended_range = [
        int(row["intended_output_min"])
        <= int(row["actual_output_tokens"])
        <= int(row["intended_output_max"])
        for row in rows
    ]

    prior_metrics = log1p_prior_metrics(
        [int(value) for value in actual],
        [float(value) for value in predicted],
        [float(value) for value in mus],
        residual_variance,
    )
    return {
        "count": len(rows),
        "unique_prompt_count": len({str(row["prompt_id"]) for row in rows}),
        "unique_family_count": len({str(row["prompt_family_id"]) for row in rows}),
        "actual_mean_tokens": float(actual.mean()),
        "actual_median_tokens": float(np.median(actual)),
        "actual_std_tokens": float(actual.std()),
        "actual_min_tokens": int(actual.min()),
        "actual_max_tokens": int(actual.max()),
        "predicted_mean_tokens": float(predicted.mean()),
        "predicted_median_tokens": float(np.median(predicted)),
        "bias_tokens": float(signed_errors.mean()),
        "median_absolute_error_tokens": float(np.median(absolute_errors)),
        "underprediction_rate": float(np.mean(signed_errors < 0.0)),
        "overprediction_rate": float(np.mean(signed_errors > 0.0)),
        "severe_underestimation_rate_100": severe_underestimation_rate(
            [int(value) for value in actual],
            [float(value) for value in predicted],
            threshold=100.0,
        ),
        "intended_range_compliance": float(np.mean(in_intended_range)),
        "mean_interval_95_width_tokens": float(np.mean(upper - lower)),
        "r_squared_tokens": _r_squared(actual, predicted),
        **prior_metrics,
    }


def _ordered_groups(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    order: Sequence[str],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    ordered = {name: grouped.pop(name) for name in order if name in grouped}
    ordered.update(sorted(grouped.items()))
    return ordered


def _summarize_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    residual_variance: float,
) -> dict[str, dict[str, int | float]]:
    return {
        name: summarize_observations(group_rows, residual_variance)
        for name, group_rows in groups.items()
    }


def build_group_breakdowns(
    rows: Sequence[Mapping[str, Any]],
    residual_variance: float,
) -> dict[str, Any]:
    """Build overall, marginal, interaction, and seed-level evaluation summaries."""

    length_groups = _ordered_groups(rows, "intended_length", LENGTH_ORDER)
    task_groups = _ordered_groups(rows, "task_type", TASK_ORDER)
    seed_order = tuple(str(seed) for seed in sorted({int(row["seed"]) for row in rows}))
    seed_groups = _ordered_groups(rows, "seed", seed_order)

    interaction_groups: dict[str, list[Mapping[str, Any]]] = {}
    for task in TASK_ORDER:
        for length in LENGTH_ORDER:
            selected = [
                row for row in rows if row["task_type"] == task and row["intended_length"] == length
            ]
            if selected:
                interaction_groups[f"{task}/{length}"] = selected

    return {
        "overall": summarize_observations(rows, residual_variance),
        "by_intended_length": _summarize_groups(length_groups, residual_variance),
        "by_task_type": _summarize_groups(task_groups, residual_variance),
        "by_task_type_and_intended_length": _summarize_groups(
            interaction_groups, residual_variance
        ),
        "by_seed": _summarize_groups(seed_groups, residual_variance),
    }


def _average_rows_by_prompt(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)

    averaged = []
    for prompt_id, prompt_rows in grouped.items():
        first = prompt_rows[0]
        for name in ("prompt_family_id", "task_type", "intended_length"):
            if any(row[name] != first[name] for row in prompt_rows):
                raise ValueError(f"inconsistent {name} for prompt_id={prompt_id}")
        averaged.append(
            {
                "prompt_id": prompt_id,
                "prompt_family_id": first["prompt_family_id"],
                "task_type": first["task_type"],
                "intended_length": first["intended_length"],
                "actual_output_tokens": float(
                    np.mean([float(row["actual_output_tokens"]) for row in prompt_rows])
                ),
                "predicted_mean_output_tokens": float(
                    np.mean([float(row["predicted_mean_output_tokens"]) for row in prompt_rows])
                ),
                "seed_count": len(prompt_rows),
            }
        )
    return averaged


def _point_group_summaries(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, int | float]]:
    return {name: _point_summary(group_rows) for name, group_rows in groups.items()}


def build_prompt_mean_point_analysis(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate one ALPS point prediction against the three-seed mean per prompt."""

    averaged = _average_rows_by_prompt(rows)
    length_groups = _ordered_groups(averaged, "intended_length", LENGTH_ORDER)
    task_groups = _ordered_groups(averaged, "task_type", TASK_ORDER)
    interaction_groups: dict[str, list[Mapping[str, Any]]] = {}
    for task in TASK_ORDER:
        for length in LENGTH_ORDER:
            selected = [
                row
                for row in averaged
                if row["task_type"] == task and row["intended_length"] == length
            ]
            if selected:
                interaction_groups[f"{task}/{length}"] = selected

    seed_counts = {int(row["seed_count"]) for row in averaged}
    return {
        "evaluation_unit": "prompt_id_after_averaging_actual_length_across_seeds",
        "seed_counts_per_prompt": sorted(seed_counts),
        "overall": _point_summary(averaged),
        "by_intended_length": _point_group_summaries(length_groups),
        "by_task_type": _point_group_summaries(task_groups),
        "by_task_type_and_intended_length": _point_group_summaries(interaction_groups),
    }


def _triplet_summary(triplets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not triplets:
        return {"triplet_count": 0}

    actual_strict = []
    actual_nondecreasing = []
    predicted_strict = []
    predicted_nondecreasing = []
    contrast_pairs = (("short", "medium"), ("medium", "long"), ("short", "long"))
    contrast_values: dict[str, dict[str, list[float]]] = {
        f"{start}_to_{end}": {
            "actual": [],
            "predicted": [],
            "error": [],
        }
        for start, end in contrast_pairs
    }

    for triplet in triplets:
        actual = [float(triplet[length]["actual"]) for length in LENGTH_ORDER]
        predicted = [float(triplet[length]["predicted"]) for length in LENGTH_ORDER]
        actual_strict.append(actual[0] < actual[1] < actual[2])
        actual_nondecreasing.append(actual[0] <= actual[1] <= actual[2])
        predicted_strict.append(predicted[0] < predicted[1] < predicted[2])
        predicted_nondecreasing.append(predicted[0] <= predicted[1] <= predicted[2])

        for start, end in contrast_pairs:
            actual_delta = float(triplet[end]["actual"]) - float(triplet[start]["actual"])
            predicted_delta = float(triplet[end]["predicted"]) - float(triplet[start]["predicted"])
            values = contrast_values[f"{start}_to_{end}"]
            values["actual"].append(actual_delta)
            values["predicted"].append(predicted_delta)
            values["error"].append(predicted_delta - actual_delta)

    contrasts = {}
    for name, values in contrast_values.items():
        errors = np.asarray(values["error"], dtype=np.float64)
        contrasts[name] = {
            "actual_delta_mean_tokens": float(np.mean(values["actual"])),
            "predicted_delta_mean_tokens": float(np.mean(values["predicted"])),
            "delta_bias_tokens": float(errors.mean()),
            "delta_mae_tokens": float(np.abs(errors).mean()),
            "delta_rmse_tokens": float(np.sqrt(np.square(errors).mean())),
        }

    return {
        "triplet_count": len(triplets),
        "actual_strict_monotonic_rate": float(np.mean(actual_strict)),
        "actual_nondecreasing_rate": float(np.mean(actual_nondecreasing)),
        "predicted_strict_monotonic_rate": float(np.mean(predicted_strict)),
        "predicted_nondecreasing_rate": float(np.mean(predicted_nondecreasing)),
        "contrasts": contrasts,
    }


def _complete_triplets(
    rows: Iterable[Mapping[str, Any]],
    *,
    average_seeds: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        family = str(row["prompt_family_id"])
        task = str(row["task_type"])
        key = (family, task) if average_seeds else (family, task, str(row["seed"]))
        grouped[key][str(row["intended_length"])].append(row)

    triplets = []
    for key, by_length in grouped.items():
        if any(length not in by_length for length in LENGTH_ORDER):
            continue
        triplet: dict[str, Any] = {
            "prompt_family_id": key[0],
            "task_type": key[1],
        }
        if not average_seeds:
            triplet["seed"] = int(key[2])
        for length in LENGTH_ORDER:
            length_rows = by_length[length]
            triplet[length] = {
                "actual": float(np.mean([int(row["actual_output_tokens"]) for row in length_rows])),
                "predicted": float(
                    np.mean([float(row["predicted_mean_output_tokens"]) for row in length_rows])
                ),
            }
        triplets.append(triplet)
    return triplets


def _matched_level_report(triplets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task = {
        task: _triplet_summary([triplet for triplet in triplets if triplet["task_type"] == task])
        for task in TASK_ORDER
    }
    return {
        "overall": _triplet_summary(triplets),
        "by_task_type": by_task,
    }


def build_matched_length_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare matched short/medium/long variants within each prompt family."""

    rollout_triplets = _complete_triplets(rows, average_seeds=False)
    family_mean_triplets = _complete_triplets(rows, average_seeds=True)
    return {
        "rollout_level": _matched_level_report(rollout_triplets),
        "family_mean_level": _matched_level_report(family_mean_triplets),
    }


def build_seed_stability(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    """Describe generation variability and verify seed-invariant prefill predictions."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)
    actual_stds = []
    prediction_spans = []
    for prompt_rows in grouped.values():
        actual = [int(row["actual_output_tokens"]) for row in prompt_rows]
        predicted = [float(row["predicted_mean_output_tokens"]) for row in prompt_rows]
        actual_stds.append(float(np.std(actual)))
        prediction_spans.append(float(max(predicted) - min(predicted)))
    return {
        "prompt_count": len(grouped),
        "mean_actual_seed_std_tokens": float(np.mean(actual_stds)),
        "median_actual_seed_std_tokens": float(np.median(actual_stds)),
        "mean_prediction_seed_span_tokens": float(np.mean(prediction_spans)),
        "max_prediction_seed_span_tokens": float(np.max(prediction_spans)),
    }


def flatten_group_breakdowns(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten grouped summaries for a spreadsheet-friendly CSV."""

    rows = [{"dimension": "overall", "group": "all", **report["overall"]}]
    dimensions = (
        ("intended_length", "by_intended_length"),
        ("task_type", "by_task_type"),
        ("task_type_x_intended_length", "by_task_type_and_intended_length"),
        ("seed", "by_seed"),
    )
    for dimension, report_key in dimensions:
        for name, metrics in report[report_key].items():
            rows.append({"dimension": dimension, "group": name, **metrics})
    return rows


def flatten_prompt_mean_point_analysis(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten prompt-mean point metrics for CSV export."""

    rows = [{"dimension": "overall", "group": "all", **report["overall"]}]
    dimensions = (
        ("intended_length", "by_intended_length"),
        ("task_type", "by_task_type"),
        ("task_type_x_intended_length", "by_task_type_and_intended_length"),
    )
    for dimension, report_key in dimensions:
        for name, metrics in report[report_key].items():
            rows.append({"dimension": dimension, "group": name, **metrics})
    return rows


def flatten_matched_length_analysis(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten matched length contrasts for CSV export."""

    rows = []
    for level in ("rollout_level", "family_mean_level"):
        level_report = report[level]
        groups = {"all": level_report["overall"], **level_report["by_task_type"]}
        for task_type, summary in groups.items():
            if summary["triplet_count"] == 0:
                continue
            common = {
                "analysis_level": level,
                "task_type": task_type,
                "triplet_count": summary["triplet_count"],
                "actual_strict_monotonic_rate": summary["actual_strict_monotonic_rate"],
                "actual_nondecreasing_rate": summary["actual_nondecreasing_rate"],
                "predicted_strict_monotonic_rate": summary["predicted_strict_monotonic_rate"],
                "predicted_nondecreasing_rate": summary["predicted_nondecreasing_rate"],
            }
            for contrast, metrics in summary["contrasts"].items():
                rows.append({**common, "contrast": contrast, **metrics})
    return rows


def _format_metric_table(
    title: str,
    groups: Mapping[str, Mapping[str, int | float]],
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Group | N | Actual mean | Predicted mean | Bias | MAE | RMSE | Raw R² | "
        "Log R² | 95% coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in groups.items():
        lines.append(
            f"| {name} | {metrics['count']} | {metrics['actual_mean_tokens']:.2f} | "
            f"{metrics['predicted_mean_tokens']:.2f} | {metrics['bias_tokens']:.2f} | "
            f"{metrics['mae_tokens']:.2f} | {metrics['rmse_tokens']:.2f} | "
            f"{metrics['r_squared_tokens']:.4f} | "
            f"{metrics['r_squared_log1p']:.4f} | "
            f"{metrics['interval_95_coverage']:.1%} |"
        )
    lines.append("")
    return lines


def _format_prompt_mean_table(
    title: str,
    groups: Mapping[str, Mapping[str, int | float]],
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Group | Prompts | Actual seed-mean | Predicted mean | Bias | MAE | RMSE | "
        "Raw R² | Log point R² | Pearson r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in groups.items():
        lines.append(
            f"| {name} | {metrics['count']} | {metrics['actual_mean_tokens']:.2f} | "
            f"{metrics['predicted_mean_tokens']:.2f} | {metrics['bias_tokens']:.2f} | "
            f"{metrics['mae_tokens']:.2f} | {metrics['rmse_tokens']:.2f} | "
            f"{metrics['r_squared_tokens']:.4f} | "
            f"{metrics['r_squared_log1p_point']:.4f} | "
            f"{metrics['pearson_correlation_tokens']:.4f} |"
        )
    lines.append("")
    return lines


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render the JSON report as compact human-readable Markdown tables."""

    lines = [
        f"# ALPS {str(report['split']).title()} breakdown",
        "",
        f"- Experiment: `{report['experiment_id']}`",
        f"- Source: `{report['source_predictions']}`",
        "- Bias = predicted mean tokens - actual tokens.",
        "- Point accuracy is primary at prompt-mean level; rollout rows remain primary for "
        "NLL and coverage.",
        "",
    ]
    prompt_mean = report["prompt_mean_point_analysis"]
    lines.extend(
        _format_prompt_mean_table(
            "Prompt-mean point accuracy: overall",
            {"all": prompt_mean["overall"]},
        )
    )
    lines.extend(
        _format_prompt_mean_table(
            "Prompt-mean point accuracy: by intended length",
            prompt_mean["by_intended_length"],
        )
    )
    lines.extend(
        _format_prompt_mean_table(
            "Prompt-mean point accuracy: by task type",
            prompt_mean["by_task_type"],
        )
    )
    lines.extend(
        _format_prompt_mean_table(
            "Prompt-mean point accuracy: task type × intended length",
            prompt_mean["by_task_type_and_intended_length"],
        )
    )
    lines.extend(_format_metric_table("Overall", {"all": report["overall"]}))
    lines.extend(_format_metric_table("By intended length", report["by_intended_length"]))
    lines.extend(_format_metric_table("By task type", report["by_task_type"]))
    lines.extend(
        _format_metric_table(
            "Task type × intended length",
            report["by_task_type_and_intended_length"],
        )
    )
    lines.extend(_format_metric_table("By seed", report["by_seed"]))

    seed_stability = report["seed_stability"]
    lines.extend(
        [
            "## Seed stability",
            "",
            f"- Prompt count: {seed_stability['prompt_count']}",
            "- Mean actual rollout standard deviation: "
            f"{seed_stability['mean_actual_seed_std_tokens']:.2f} tokens",
            "- Maximum prediction span across seeds: "
            f"{seed_stability['max_prediction_seed_span_tokens']:.6f} tokens",
            "",
            "## Matched Short → Medium → Long",
            "",
            "| Level | Task | Triplets | Actual strict monotonic | Predicted strict monotonic |",
            "|---|---|---:|---:|---:|",
        ]
    )
    matched = report["matched_length_analysis"]
    for level in ("rollout_level", "family_mean_level"):
        level_report = matched[level]
        groups = {"all": level_report["overall"], **level_report["by_task_type"]}
        for task, summary in groups.items():
            if summary["triplet_count"] == 0:
                continue
            lines.append(
                f"| {level} | {task} | {summary['triplet_count']} | "
                f"{summary['actual_strict_monotonic_rate']:.1%} | "
                f"{summary['predicted_strict_monotonic_rate']:.1%} |"
            )

    lines.extend(
        [
            "",
            "### Family-mean length contrasts",
            "",
            "| Task | Contrast | Actual Δ | Predicted Δ | Δ bias | Δ MAE | Δ RMSE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    family_report = matched["family_mean_level"]
    family_groups = {"all": family_report["overall"], **family_report["by_task_type"]}
    for task, summary in family_groups.items():
        if summary["triplet_count"] == 0:
            continue
        for contrast, metrics in summary["contrasts"].items():
            lines.append(
                f"| {task} | {contrast} | "
                f"{metrics['actual_delta_mean_tokens']:.2f} | "
                f"{metrics['predicted_delta_mean_tokens']:.2f} | "
                f"{metrics['delta_bias_tokens']:.2f} | "
                f"{metrics['delta_mae_tokens']:.2f} | "
                f"{metrics['delta_rmse_tokens']:.2f} |"
            )

    lines.extend(
        [
            "",
            "Subgroup R² can be unstable for narrow cells. Interpret it together with "
            "sample count, MAE, bias, coverage, and family-level contrasts.",
            "",
        ]
    )
    return "\n".join(lines)

"""Add a leakage-safe prompt-token baseline to the frozen Hybrid OOF comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.evaluation.hybrid import (
    absolute_step_breakdown,
    family_bootstrap_interval,
    family_macro_metrics,
    family_metric_rows,
    paired_family_mae_difference,
    sequence_balanced_metrics,
)
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.hybrid_experiment import (
    enforce_censoring_policy,
    hybrid_dataset_digest,
    hybrid_samples,
    load_complete_hybrid_split,
    load_hybrid_config,
    load_hybrid_experiment,
    partition_censored,
    validate_hybrid_config,
)
from llm_length_prediction.models.prompt_token_baseline import (
    METHOD_ID,
    fit_prompt_token_ridge,
    predict_prompt_token_countdown,
)

DEFAULT_PROTOCOL = Path("configs/experiments/alps_plp_main_comparison.json")


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "alps-plp-four-method-main-comparison-2026":
        raise ValueError("unexpected four-method comparison protocol")
    if tuple(protocol.get("methods", {}))[0] != METHOD_ID:
        raise ValueError("prompt-token baseline must be the first comparison method")
    return protocol


def _source_predictions(
    report_path: Path,
    predictions_path: Path,
    *,
    required_methods: Sequence[str],
    samples: Sequence[Any],
    expected_digest: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol_id") != "alps-plp-hybrid-v1-v2-development-2026":
        raise ValueError("source is not the frozen explicit Hybrid v1/v2 OOF")
    if report.get("test_opened") is not False:
        raise ValueError("source Hybrid OOF must not have opened Test")
    if report.get("training_dataset_digest") != expected_digest:
        raise ValueError("source OOF and current Train traces have different digests")
    if any(method not in report.get("methods", {}) for method in required_methods):
        raise ValueError("source OOF is missing a required main method")
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        key = row["prompt_id"], int(row["seed"]), int(row["step"])
        if key in indexed:
            raise ValueError(f"duplicate source OOF key: {key}")
        indexed[key] = row
    if len(indexed) != len(samples):
        raise ValueError("source OOF row count differs from current samples")
    values = {
        method: np.full(len(samples), np.nan, dtype=np.float64)
        for method in required_methods
    }
    for index, sample in enumerate(samples):
        key = sample.prompt_id, sample.seed, sample.step
        row = indexed.get(key)
        if row is None or row["prompt_family_id"] != sample.prompt_family_id:
            raise ValueError(f"source OOF identity mismatch: {key}")
        for method in required_methods:
            value = float(row[method])
            if not math.isfinite(value):
                raise ValueError(f"non-finite source prediction for {method}")
            values[method][index] = value
    return report, values


def _group_rows(
    samples: Sequence[Any],
    predictions: dict[str, np.ndarray],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for label in sorted(set(labels)):
        indices = [index for index, value in enumerate(labels) if value == label]
        subset = [samples[index] for index in indices]
        rows.append(
            {
                "group": label,
                "point_count": len(indices),
                "trace_count": len({sample.trace_key for sample in subset}),
                "methods": {
                    method: sequence_balanced_metrics(subset, values[indices])
                    for method, values in predictions.items()
                },
            }
        )
    return rows


def _progress_rows(
    samples: Sequence[Any],
    predictions: dict[str, np.ndarray],
    boundaries: Sequence[float],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    if len(boundaries) != len(labels) + 1:
        raise ValueError("decode progress boundaries and labels do not align")
    progress = np.asarray(
        [sample.step / sample.output_tokens for sample in samples], dtype=np.float64
    )
    rows = []
    for index, label in enumerate(labels):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (progress >= lower) & (
            (progress <= upper) if index == len(labels) - 1 else (progress < upper)
        )
        indices = np.flatnonzero(mask).tolist()
        if not indices:
            continue
        subset = [samples[value] for value in indices]
        rows.append(
            {
                "group": label,
                "lower": lower,
                "upper": upper,
                "point_count": len(indices),
                "trace_count": len({sample.trace_key for sample in subset}),
                "methods": {
                    method: sequence_balanced_metrics(subset, values[indices])
                    for method, values in predictions.items()
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--source-predictions", type=Path)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    protocol = _load_protocol(args.protocol)
    config_path = Path(protocol["method_config"])
    config = load_hybrid_config(config_path)
    experiment, records = load_hybrid_experiment(config)
    validate_hybrid_config(config, experiment)
    loaded = load_complete_hybrid_split(
        config, experiment, records, split="train", trace_root=args.trace_root
    )
    effective, censored = partition_censored(loaded)
    censoring = enforce_censoring_policy(
        loaded_count=len(loaded),
        censored_count=censored,
        warning_rate=float(config["censoring"]["warning_rate"]),
        abort_rate=float(config["censoring"]["abort_rate"]),
    )
    samples = hybrid_samples(effective)
    digest = hybrid_dataset_digest(effective)
    source = protocol["source_oof"]
    report_path = args.source_report or Path(source["report"])
    predictions_path = args.source_predictions or Path(source["predictions"])
    required = [str(value) for value in source["required_methods"]]
    source_report, source_predictions = _source_predictions(
        report_path,
        predictions_path,
        required_methods=required,
        samples=samples,
        expected_digest=digest,
    )
    family_folds = {
        str(family): int(fold) for family, fold in source_report["family_folds"].items()
    }
    trace_by_key = {(trace.prompt_id, seed): trace for _, seed, _, trace in effective}
    baseline = np.full(len(samples), np.nan, dtype=np.float64)
    fold_reports = []
    alpha = float(protocol["methods"][METHOD_ID]["ridge_alpha"])
    for fold in sorted(set(family_folds.values())):
        train_traces = [
            (record, trace)
            for record, _, _, trace in effective
            if family_folds[record["prompt_family_id"]] != fold
        ]
        validation_indices = [
            index
            for index, sample in enumerate(samples)
            if family_folds[sample.prompt_family_id] == fold
        ]
        fitted = fit_prompt_token_ridge(
            [trace.prompt_tokens for _, trace in train_traces],
            [trace.output_tokens for _, trace in train_traces],
            alpha=alpha,
        )
        baseline[validation_indices] = predict_prompt_token_countdown(
            fitted,
            [
                trace_by_key[(samples[index].prompt_id, samples[index].seed)].prompt_tokens
                for index in validation_indices
            ],
            [samples[index].step for index in validation_indices],
        )
        fold_reports.append(
            {
                "fold": fold,
                "train_trace_count": len(train_traces),
                "validation_trace_count": len(
                    {samples[index].trace_key for index in validation_indices}
                ),
                "validation_point_count": len(validation_indices),
            }
        )
    if np.any(~np.isfinite(baseline)):
        raise RuntimeError("prompt-token baseline OOF predictions are incomplete")
    predictions = {METHOD_ID: baseline, **source_predictions}

    evaluation = protocol["evaluation"]
    bootstrap = evaluation["bootstrap"]
    replicates = int(bootstrap["replicates"])
    confidence = float(bootstrap["confidence_level"])
    familywise = float(evaluation["primary_familywise_confidence_level"])
    seed = int(bootstrap["seed"])
    method_reports = {}
    for method_index, (method, values) in enumerate(predictions.items()):
        family_rows = family_metric_rows(samples, values)
        family_mae = {
            str(row["prompt_family_id"]): float(row["sequence_balanced_mae_tokens"])
            for row in family_rows
        }
        method_reports[method] = {
            "metrics": family_macro_metrics(samples, values),
            "family_macro_mae_ci_95": family_bootstrap_interval(
                family_mae,
                replicates=replicates,
                confidence=confidence,
                seed=seed + method_index,
            ),
            "absolute_step_breakdown": absolute_step_breakdown(
                samples, values, boundaries=evaluation["absolute_step_bins"]
            ),
        }
    candidate = "alps_plp_concat_v1"
    primary_paired = {}
    for index, comparator in enumerate(
        (METHOD_ID, "alps_countdown", "plp_terminal_zero_v3")
    ):
        key = f"{candidate}_minus_{comparator}"
        primary_paired[key] = {
            "ci_95": paired_family_mae_difference(
                samples,
                predictions[candidate],
                predictions[comparator],
                replicates=replicates,
                confidence=confidence,
                seed=seed + 100 + index,
            ),
            "familywise_ci": paired_family_mae_difference(
                samples,
                predictions[candidate],
                predictions[comparator],
                replicates=replicates,
                confidence=familywise,
                seed=seed + 200 + index,
            ),
        }
    explanatory_paired = {}
    for index, method in enumerate(("alps_countdown", "plp_terminal_zero_v3")):
        explanatory_paired[f"{method}_minus_{METHOD_ID}"] = paired_family_mae_difference(
            samples,
            predictions[method],
            predictions[METHOD_ID],
            replicates=replicates,
            confidence=confidence,
            seed=seed + 300 + index,
        )

    task_labels = [sample.task for sample in samples]
    length_labels = [sample.intended_length for sample in samples]
    task_length_labels = [f"{sample.task}__{sample.intended_length}" for sample in samples]
    fold_labels = [str(family_folds[sample.prompt_family_id]) for sample in samples]
    breakdowns = {
        "by_decode_progress": _progress_rows(
            samples,
            predictions,
            evaluation["decode_progress_boundaries"],
            evaluation["decode_progress_labels"],
        ),
        "by_task": _group_rows(samples, predictions, task_labels),
        "by_intended_length": _group_rows(samples, predictions, length_labels),
        "by_task_and_intended_length": _group_rows(
            samples, predictions, task_length_labels
        ),
        "by_outer_fold": _group_rows(samples, predictions, fold_labels),
    }

    rows = []
    for index, sample in enumerate(samples):
        trace = trace_by_key[(sample.prompt_id, sample.seed)]
        rows.append(
            {
                "prompt_id": sample.prompt_id,
                "prompt_family_id": sample.prompt_family_id,
                "task": sample.task,
                "intended_length": sample.intended_length,
                "seed": sample.seed,
                "prompt_tokens": trace.prompt_tokens,
                "step": sample.step,
                "output_tokens": sample.output_tokens,
                "remaining_tokens": sample.remaining_tokens,
                "outer_fold": family_folds[sample.prompt_family_id],
                **{method: values[index] for method, values in predictions.items()},
            }
        )
    output = args.output_dir or Path(protocol["outputs"]["run_root"]) / "oof"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "split": "train_family_grouped_supplemental_oof",
        "test_opened": False,
        "config_sha256": file_sha256(config_path),
        "protocol_sha256": file_sha256(args.protocol),
        "source_oof_report_sha256": file_sha256(report_path),
        "source_oof_predictions_sha256": file_sha256(predictions_path),
        "training_dataset_digest": digest,
        "censoring": censoring,
        "family_folds": family_folds,
        "baseline_fold_reports": fold_reports,
        "methods": method_reports,
        "primary_paired_differences": primary_paired,
        "explanatory_paired_differences": explanatory_paired,
        "breakdowns": breakdowns,
        "selection_status": evaluation["selection_status"],
        "interpretation": (
            "same-family supplemental baseline; concat v1 was selected before this analysis; "
            "old Test remains unopened for Hybrid"
        ),
    }
    (output / "oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote four-method main comparison for {len(samples)} points to {output}")


if __name__ == "__main__":
    main()

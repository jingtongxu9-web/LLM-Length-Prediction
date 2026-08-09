"""Train only gated residual v2.1 and reuse verified Hybrid v1/v2 OOF controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from llm_length_prediction.evaluation.gated_residual import gated_residual_diagnostics
from llm_length_prediction.evaluation.hybrid import (
    absolute_step_breakdown,
    family_bootstrap_interval,
    family_macro_metrics,
    family_metric_rows,
    paired_family_mae_difference,
    task_stratified_family_folds,
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
from llm_length_prediction.models.gated_residual import (
    METHOD_ID,
    SCALAR_RIDGE_ID,
    correction_bounds,
    fit_gated_residual,
    fit_scalar_residual_ridge,
    predict_gated_residual,
    predict_scalar_residual_ridge,
    progress_values,
)
from llm_length_prediction.models.hybrid import (
    alps_prior_summaries,
    cross_fitted_prior_summaries,
)

DEFAULT_METHOD_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/alps_plp_gated_residual_v2_1.json")


def load_protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "alps-plp-gated-residual-v2-1-development-2026":
        raise ValueError("unexpected gated residual v2.1 protocol")
    if payload.get("method_id") != METHOD_ID:
        raise ValueError("gated residual protocol method ID does not match implementation")
    return payload


def _sample_key(prompt_id: str, seed: int, step: int) -> tuple[str, int, int]:
    return prompt_id, seed, step


def load_source_predictions(
    report_path: Path,
    predictions_path: Path,
    *,
    required_methods: list[str],
    samples: list[object],
    expected_digest: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol_id") != "alps-plp-hybrid-v1-v2-development-2026":
        raise ValueError("source OOF protocol is not the frozen Hybrid v1/v2 comparison")
    if report.get("test_opened") is not False:
        raise ValueError("source OOF must not have opened Test")
    if report.get("training_dataset_digest") != expected_digest:
        raise ValueError("source OOF and current Train traces have different digests")
    if list(report.get("methods", {})) != required_methods:
        raise ValueError("source OOF method order changed")
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        key = _sample_key(row["prompt_id"], int(row["seed"]), int(row["step"]))
        if key in indexed:
            raise ValueError(f"source OOF has duplicate prediction key: {key}")
        indexed[key] = row
    if len(indexed) != len(samples):
        raise ValueError("source OOF row count differs from current Hybrid samples")
    predictions = {
        method: np.full(len(samples), np.nan, dtype=np.float64)
        for method in required_methods
    }
    for index, sample in enumerate(samples):
        key = _sample_key(sample.prompt_id, sample.seed, sample.step)  # type: ignore[attr-defined]
        row = indexed.get(key)
        if row is None:
            raise ValueError(f"source OOF is missing prediction key: {key}")
        if row["prompt_family_id"] != sample.prompt_family_id:  # type: ignore[attr-defined]
            raise ValueError(f"source OOF family mismatch for {key}")
        expected_fold = int(report["family_folds"][sample.prompt_family_id])  # type: ignore[attr-defined,index]
        if int(row["outer_fold"]) != expected_fold:
            raise ValueError(f"source OOF fold mismatch for {key}")
        for method in required_methods:
            value = float(row[method])
            if not math.isfinite(value):
                raise ValueError(f"source OOF contains non-finite {method} prediction")
            predictions[method][index] = value
    return report, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-config", type=Path, default=DEFAULT_METHOD_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--source-predictions", type=Path)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_hybrid_config(args.method_config)
    experiment, records = load_hybrid_experiment(config)
    validate_hybrid_config(config, experiment)
    protocol = load_protocol(args.protocol)
    source = protocol["source_oof"]  # type: ignore[index]
    report_path = args.source_report or Path(source["report"])
    predictions_path = args.source_predictions or Path(source["predictions"])
    required_methods = [str(method) for method in source["required_methods"]]
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
    source_report, method_predictions = load_source_predictions(
        report_path,
        predictions_path,
        required_methods=required_methods,
        samples=samples,  # type: ignore[arg-type]
        expected_digest=digest,
    )
    policy = protocol["data_policy"]  # type: ignore[index]
    folds = int(policy["train_oof_folds"])
    fold_seed = int(policy["fold_seed"])
    expected_folds = task_stratified_family_folds(samples, folds=folds, seed=fold_seed)
    source_folds = {str(key): int(value) for key, value in source_report["family_folds"].items()}
    if source_folds != expected_folds:
        raise ValueError("source OOF family folds differ from the frozen v2.1 folds")

    output = args.output_dir or Path(protocol["outputs"]["run_root"]) / "oof"  # type: ignore[index]
    output.mkdir(parents=True, exist_ok=True)
    candidate = np.full(len(samples), np.nan, dtype=np.float64)
    scalar_ridge_prediction = np.full(len(samples), np.nan, dtype=np.float64)
    correction = np.full(len(samples), np.nan, dtype=np.float64)
    gate = np.full(len(samples), np.nan, dtype=np.float64)
    gate_confidence = np.full(len(samples), np.nan, dtype=np.float64)
    bounded = np.full(len(samples), np.nan, dtype=np.float64)
    terminal_probability = np.full(len(samples), np.nan, dtype=np.float64)
    fold_reports = []
    for fold in range(folds):
        train_indices = [
            index
            for index, sample in enumerate(samples)
            if source_folds[sample.prompt_family_id] != fold
        ]
        validation_indices = [
            index
            for index, sample in enumerate(samples)
            if source_folds[sample.prompt_family_id] == fold
        ]
        train = [samples[index] for index in train_indices]
        validation = [samples[index] for index in validation_indices]
        print(
            f"starting gated residual outer fold {fold + 1}/{folds}: "
            f"train={len(train)} points, validation={len(validation)} points",
            flush=True,
        )
        inner_folds = task_stratified_family_folds(
            train,
            folds=int(policy["inner_prior_crossfit_folds"]),
            seed=fold_seed + fold + 1,
        )
        cross_fitted_prior = cross_fitted_prior_summaries(train, inner_folds)
        scalar_ridge = fit_scalar_residual_ridge(
            train,
            cross_fitted_prior,
            alpha=float(protocol["scalar_residual_ridge"]["alpha"]),  # type: ignore[index]
        )
        fitted = fit_gated_residual(
            train,
            cross_fitted_prior,
            method_config=config,
            protocol=protocol,  # type: ignore[arg-type]
            device=args.device,
        )
        predicted, diagnostics = predict_gated_residual(
            fitted,
            validation,
            batch_size=int(protocol["training"]["batch_size"]),  # type: ignore[index]
        )
        validation_prior = alps_prior_summaries(fitted.alps_prior, validation)
        scalar_predicted = predict_scalar_residual_ridge(
            scalar_ridge, validation, validation_prior
        )
        scalar_ridge_prediction[validation_indices] = scalar_predicted
        candidate[validation_indices] = predicted
        correction[validation_indices] = diagnostics["applied_correction"]
        gate[validation_indices] = diagnostics["gate"]
        gate_confidence[validation_indices] = diagnostics["gate_confidence"]
        bounded[validation_indices] = diagnostics["bounded_correction"]
        terminal_probability[validation_indices] = diagnostics["terminal_probability"]
        fold_report = {
            "fold": fold,
            "train_family_count": len({sample.prompt_family_id for sample in train}),
            "validation_family_count": len(
                {sample.prompt_family_id for sample in validation}
            ),
            "train_point_count": len(train),
            "validation_point_count": len(validation),
            "training_report": fitted.metadata,
        }
        fold_reports.append(fold_report)
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "completed_outer_folds": len(fold_reports),
                    "total_outer_folds": folds,
                    "last_completed_fold": fold,
                    "fold_reports": fold_reports,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"completed gated residual outer fold {fold + 1}/{folds}", flush=True)
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(scalar_ridge_prediction)):
        raise RuntimeError("supplemental residual OOF predictions are incomplete")
    method_predictions[SCALAR_RIDGE_ID] = scalar_ridge_prediction
    method_predictions[METHOD_ID] = candidate

    evaluation = protocol["evaluation"]  # type: ignore[index]
    bootstrap = evaluation["bootstrap"]
    replicates = int(bootstrap["replicates"])
    confidence = float(bootstrap["confidence_level"])
    familywise_confidence = float(evaluation["familywise_confidence_level"])
    seed = int(bootstrap["seed"])
    method_reports = {}
    for method_index, (method, predictions) in enumerate(method_predictions.items()):
        family_rows = family_metric_rows(samples, predictions)
        family_mae = {
            str(row["prompt_family_id"]): float(row["sequence_balanced_mae_tokens"])
            for row in family_rows
        }
        method_reports[method] = {
            "metrics": family_macro_metrics(samples, predictions),
            "family_macro_mae_ci_95": family_bootstrap_interval(
                family_mae,
                replicates=replicates,
                confidence=confidence,
                seed=seed + method_index,
            ),
            "absolute_step_breakdown": absolute_step_breakdown(
                samples,
                predictions,
                boundaries=evaluation["absolute_step_bins"],
            ),
        }
    paired = {}
    comparators = [*required_methods, SCALAR_RIDGE_ID]
    for index, comparator in enumerate(comparators):
        paired[f"{METHOD_ID}_minus_{comparator}"] = {
            "ci_95": paired_family_mae_difference(
                samples,
                candidate,
                method_predictions[comparator],
                replicates=replicates,
                confidence=confidence,
                seed=seed + 100 + index,
            ),
            "familywise_ci": paired_family_mae_difference(
                samples,
                candidate,
                method_predictions[comparator],
                replicates=replicates,
                confidence=familywise_confidence,
                seed=seed + 200 + index,
            ),
        }
    alps = method_predictions["alps_countdown"]
    progress = progress_values(samples, alps)
    method_settings = protocol["method"]  # type: ignore[index]
    bounds = correction_bounds(
        alps,
        fraction=float(method_settings["correction_bound_fraction"]),
        minimum_tokens=float(method_settings["minimum_correction_bound_tokens"]),
    )
    diagnostics_report = gated_residual_diagnostics(
        samples,
        method_predictions,
        candidate_id=METHOD_ID,
        alps_id="alps_countdown",
        concat_id="alps_plp_concat_v1",
        scalar_id=SCALAR_RIDGE_ID,
        applied=correction,
        gate=gate,
        gate_confidence=gate_confidence,
        bounded=bounded,
        terminal_probability=terminal_probability,
        progress=progress,
        bounds=bounds,
        family_folds=source_folds,
        settings=evaluation["gate_diagnostics"],
        terminal_threshold=float(method_settings["terminal_threshold"]),
    )

    actual = np.asarray([sample.remaining_tokens for sample in samples], dtype=np.float64)
    improvement_over_alps = np.abs(alps - actual) - np.abs(candidate - actual)
    concat_v1 = method_predictions["alps_plp_concat_v1"]
    improvement_over_concat = np.abs(concat_v1 - actual) - np.abs(candidate - actual)
    correction_success = np.abs(candidate - actual) < np.abs(alps - actual)
    saturation_ratio = float(evaluation["gate_diagnostics"]["bound_saturation_ratio"])
    bound_saturated = np.abs(bounded) >= saturation_ratio * bounds

    rows = []
    for index, sample in enumerate(samples):
        rows.append(
            {
                "prompt_id": sample.prompt_id,
                "prompt_family_id": sample.prompt_family_id,
                "task": sample.task,
                "intended_length": sample.intended_length,
                "seed": sample.seed,
                "step": sample.step,
                "remaining_tokens": sample.remaining_tokens,
                "outer_fold": source_folds[sample.prompt_family_id],
                **{
                    method: predictions[index]
                    for method, predictions in method_predictions.items()
                },
                "v2_1_applied_correction": correction[index],
                "v2_1_gate": gate[index],
                "v2_1_gate_confidence": gate_confidence[index],
                "v2_1_bounded_correction": bounded[index],
                "v2_1_terminal_probability": terminal_probability[index],
                "v2_1_progress": progress[index],
                "v2_1_correction_bound": bounds[index],
                "v2_1_improvement_over_alps": improvement_over_alps[index],
                "v2_1_improvement_over_concat_v1": improvement_over_concat[index],
                "v2_1_correction_success": int(correction_success[index]),
                "v2_1_bound_saturated": int(bound_saturated[index]),
            }
        )
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 2,
        "protocol_id": protocol["protocol_id"],
        "split": "train_family_grouped_supplemental_oof",
        "test_opened": False,
        "method_config_sha256": file_sha256(args.method_config),
        "protocol_sha256": file_sha256(args.protocol),
        "source_oof_report_sha256": file_sha256(report_path),
        "source_oof_predictions_sha256": file_sha256(predictions_path),
        "training_dataset_digest": digest,
        "censoring": censoring,
        "family_folds": source_folds,
        "fold_reports": fold_reports,
        "methods": method_reports,
        "paired_differences": paired,
        "gated_correction_diagnostics": diagnostics_report,
        "selection_rule": evaluation["selection_rule"],
        "interpretation": "supplemental OOF reusing frozen controls; old Test remains unopened",
    }
    (output / "oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote gated residual v2.1 OOF report for {len(samples)} points to {output}")


if __name__ == "__main__":
    main()

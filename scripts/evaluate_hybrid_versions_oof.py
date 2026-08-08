"""Compare explicit Hybrid v1/v2 with leakage-safe family-grouped OOF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

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
from llm_length_prediction.models.hybrid import cross_fitted_prior_summaries
from llm_length_prediction.models.hybrid_versions import (
    METHOD_IDS,
    fit_hybrid_versions,
    predict_hybrid_versions,
)

DEFAULT_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/alps_plp_hybrid_versions.json")


def load_protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "alps-plp-hybrid-v1-v2-development-2026":
        raise ValueError("unexpected Hybrid v1/v2 protocol")
    if tuple(payload["methods"]) != METHOD_IDS:  # type: ignore[arg-type]
        raise ValueError("Hybrid v1/v2 method order does not match implementation")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_hybrid_config(args.config)
    experiment, records = load_hybrid_experiment(config)
    validate_hybrid_config(config, experiment)
    protocol = load_protocol(args.protocol)
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
    policy = protocol["data_policy"]  # type: ignore[index]
    folds = int(policy["train_oof_folds"])
    fold_seed = int(policy["fold_seed"])
    outer_folds = task_stratified_family_folds(samples, folds=folds, seed=fold_seed)
    output = args.output_dir or Path(protocol["outputs"]["run_root"]) / "oof"  # type: ignore[index]
    output.mkdir(parents=True, exist_ok=True)
    predictions = {
        method: np.full(len(samples), np.nan, dtype=np.float64) for method in METHOD_IDS
    }
    correction = np.full(len(samples), np.nan, dtype=np.float64)
    terminal_probability = np.full(len(samples), np.nan, dtype=np.float64)
    fold_reports = []
    for fold in range(folds):
        train_indices = [
            index
            for index, sample in enumerate(samples)
            if outer_folds[sample.prompt_family_id] != fold
        ]
        validation_indices = [
            index
            for index, sample in enumerate(samples)
            if outer_folds[sample.prompt_family_id] == fold
        ]
        train = [samples[index] for index in train_indices]
        validation = [samples[index] for index in validation_indices]
        print(
            f"starting outer fold {fold + 1}/{folds}: "
            f"train={len(train)} points, validation={len(validation)} points",
            flush=True,
        )
        inner_folds = task_stratified_family_folds(
            train,
            folds=int(policy["inner_prior_crossfit_folds"]),
            seed=fold_seed + fold + 1,
        )
        cross_fitted_prior = cross_fitted_prior_summaries(train, inner_folds)
        fitted = fit_hybrid_versions(
            train,
            cross_fitted_prior,
            config=config,
            protocol=protocol,  # type: ignore[arg-type]
            device=args.device,
        )
        predicted, diagnostics = predict_hybrid_versions(
            fitted,
            validation,
            batch_size=int(config["training"]["batch_size"]),
        )
        for method in METHOD_IDS:
            predictions[method][validation_indices] = predicted[method]
        correction[validation_indices] = diagnostics["residual_correction"]
        terminal_probability[validation_indices] = diagnostics["terminal_probability"]
        fold_report = {
            "fold": fold,
            "train_family_count": len({sample.prompt_family_id for sample in train}),
            "validation_family_count": len(
                {sample.prompt_family_id for sample in validation}
            ),
            "train_point_count": len(train),
            "validation_point_count": len(validation),
            "training_reports": fitted.reports,
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
        print(f"completed outer fold {fold + 1}/{folds}", flush=True)
    if any(np.any(~np.isfinite(values)) for values in predictions.values()):
        raise RuntimeError("OOF method predictions are incomplete")
    if np.any(~np.isfinite(correction)) or np.any(~np.isfinite(terminal_probability)):
        raise RuntimeError("OOF residual diagnostics are incomplete")

    evaluation = protocol["evaluation"]  # type: ignore[index]
    bootstrap = evaluation["bootstrap"]
    replicates = int(bootstrap["replicates"])
    confidence = float(bootstrap["confidence_level"])
    familywise_confidence = float(evaluation["familywise_confidence_level"])
    seed = int(bootstrap["seed"])
    method_reports = {}
    for method_index, method in enumerate(METHOD_IDS):
        family_rows = family_metric_rows(samples, predictions[method])
        family_mae = {
            str(row["prompt_family_id"]): float(row["sequence_balanced_mae_tokens"])
            for row in family_rows
        }
        method_reports[method] = {
            "metrics": family_macro_metrics(samples, predictions[method]),
            "family_macro_mae_ci_95": family_bootstrap_interval(
                family_mae,
                replicates=replicates,
                confidence=confidence,
                seed=seed + method_index,
            ),
            "absolute_step_breakdown": absolute_step_breakdown(
                samples,
                predictions[method],
                boundaries=evaluation["absolute_step_bins"],
            ),
        }
    pairs = (
        ("alps_plp_concat_v1", "alps_countdown"),
        ("alps_plp_concat_v1", "plp_terminal_zero_v3"),
        ("alps_plp_residual_v2", "alps_countdown"),
        ("alps_plp_residual_v2", "plp_terminal_zero_v3"),
        ("alps_plp_residual_v2", "alps_plp_concat_v1"),
    )
    paired = {}
    for index, (candidate, comparator) in enumerate(pairs):
        key = f"{candidate}_minus_{comparator}"
        paired[key] = {
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
                confidence=familywise_confidence,
                seed=seed + 200 + index,
            ),
        }
    actual = np.asarray([sample.remaining_tokens for sample in samples], dtype=np.float64)
    weights = np.asarray([sample.sequence_weight for sample in samples], dtype=np.float64)
    residual_better = np.abs(predictions["alps_plp_residual_v2"] - actual) < np.abs(
        predictions["alps_countdown"] - actual
    )
    correction_report = {
        "sequence_balanced_correction_success_rate": float(
            np.sum(weights * residual_better) / weights.sum()
        ),
        "sequence_balanced_mean_predicted_correction_tokens": float(
            np.sum(weights * correction) / weights.sum()
        ),
        "sequence_balanced_mean_needed_correction_tokens": float(
            np.sum(weights * (actual - predictions["alps_countdown"])) / weights.sum()
        ),
        "sequence_balanced_terminal_decision_rate": float(
            np.sum(weights * (terminal_probability >= 0.5)) / weights.sum()
        ),
    }

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
                "outer_fold": outer_folds[sample.prompt_family_id],
                **{method: predictions[method][index] for method in METHOD_IDS},
                "v2_residual_correction": correction[index],
                "v2_terminal_probability": terminal_probability[index],
            }
        )
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "split": "train_family_grouped_oof",
        "test_opened": False,
        "config_sha256": file_sha256(args.config),
        "protocol_sha256": file_sha256(args.protocol),
        "training_dataset_digest": hybrid_dataset_digest(effective),
        "censoring": censoring,
        "family_folds": outer_folds,
        "fold_reports": fold_reports,
        "methods": method_reports,
        "paired_differences": paired,
        "residual_correction_diagnostics": correction_report,
        "interpretation": "development-only OOF; the old PLP holdout is not reused",
    }
    (output / "oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote Hybrid v1/v2 OOF report for {len(samples)} points to {output}")


if __name__ == "__main__":
    main()

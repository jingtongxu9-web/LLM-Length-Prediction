"""Run the frozen family-grouped OOF comparison before opening final Test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from llm_length_prediction.evaluation.hybrid import (
    absolute_step_breakdown,
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
from llm_length_prediction.models.hybrid_suite import METHOD_IDS, fit_suite, predict_suite

DEFAULT_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/alps_plp_hybrid_v3_protocol.json")


def _protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "alps-plp-hybrid-v3-confirmatory-2026":
        raise ValueError("unexpected Hybrid v3 protocol")
    if tuple(payload["methods"]) != METHOD_IDS:  # type: ignore[arg-type]
        raise ValueError("protocol method order does not match the implementation")
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
    protocol = _protocol(args.protocol)
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
    outer_folds = task_stratified_family_folds(
        samples, folds=int(policy["train_oof_folds"]), seed=int(policy["fold_seed"])
    )
    method_predictions = {
        method: np.full(len(samples), np.nan, dtype=np.float64) for method in METHOD_IDS
    }
    fold_reports = []
    for fold in range(int(policy["train_oof_folds"])):
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
        inner_folds = task_stratified_family_folds(
            train,
            folds=int(policy["inner_prior_crossfit_folds"]),
            seed=int(policy["fold_seed"]) + fold + 1,
        )
        cross_fitted_prior = cross_fitted_prior_summaries(train, inner_folds)
        fitted = fit_suite(
            train,
            cross_fitted_prior,
            config=config,
            protocol=protocol,
            device=args.device,
        )
        predicted = predict_suite(
            fitted,
            validation,
            device=args.device,
            batch_size=int(config["training"]["batch_size"]),
        )
        for method in METHOD_IDS:
            method_predictions[method][validation_indices] = predicted[method]
        fold_reports.append(
            {
                "fold": fold,
                "train_family_count": len({sample.prompt_family_id for sample in train}),
                "validation_family_count": len({sample.prompt_family_id for sample in validation}),
                "train_point_count": len(train),
                "validation_point_count": len(validation),
                "training_reports": fitted.reports,
            }
        )
    if any(np.any(~np.isfinite(values)) for values in method_predictions.values()):
        raise RuntimeError("OOF predictions are incomplete")

    bootstrap = protocol["evaluation"]["bootstrap"]  # type: ignore[index]
    familywise = protocol["evaluation"]["primary_familywise_rule"]  # type: ignore[index]
    replicates = int(bootstrap["replicates"])
    standard_confidence = float(bootstrap["confidence_level"])
    primary_confidence = float(familywise["per_comparison_confidence_level"])
    seed = int(bootstrap["seed"])
    methods = {}
    for method_index, method in enumerate(METHOD_IDS):
        predictions = method_predictions[method]
        family_rows = family_metric_rows(samples, predictions)
        family_mae = {
            str(row["prompt_family_id"]): float(row["sequence_balanced_mae_tokens"])
            for row in family_rows
        }
        from llm_length_prediction.evaluation.hybrid import family_bootstrap_interval

        methods[method] = {
            "metrics": family_macro_metrics(samples, predictions),
            "family_macro_mae_ci_95": family_bootstrap_interval(
                family_mae,
                replicates=replicates,
                confidence=standard_confidence,
                seed=seed + method_index,
            ),
            "absolute_step_breakdown": absolute_step_breakdown(
                samples,
                predictions,
                boundaries=protocol["evaluation"]["absolute_step_bins"],  # type: ignore[index]
            ),
        }
    hybrid = method_predictions["alps_plp_hybrid_v3"]
    differences = {}
    for index, comparator in enumerate(METHOD_IDS[:-1]):
        differences[comparator] = {
            "ci_95": paired_family_mae_difference(
                samples,
                hybrid,
                method_predictions[comparator],
                replicates=replicates,
                confidence=standard_confidence,
                seed=seed + 100 + index,
            ),
            "primary_familywise_ci": paired_family_mae_difference(
                samples,
                hybrid,
                method_predictions[comparator],
                replicates=replicates,
                confidence=primary_confidence,
                seed=seed + 200 + index,
            ),
        }
    output = args.output_dir or Path(config["outputs"]["run_root"]) / "oof"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample in enumerate(samples):
        rows.append(
            {
                "prompt_id": sample.prompt_id,
                "prompt_family_id": sample.prompt_family_id,
                "task": sample.task,
                "seed": sample.seed,
                "step": sample.step,
                "remaining_tokens": sample.remaining_tokens,
                "outer_fold": outer_folds[sample.prompt_family_id],
                **{method: method_predictions[method][index] for method in METHOD_IDS},
            }
        )
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "split": "train_grouped_oof",
        "test_opened": False,
        "config_sha256": file_sha256(args.config),
        "protocol_sha256": file_sha256(args.protocol),
        "training_dataset_digest": hybrid_dataset_digest(effective),
        "censoring": censoring,
        "family_folds": outer_folds,
        "fold_reports": fold_reports,
        "methods": methods,
        "hybrid_paired_differences": differences,
        "interpretation": "development-only OOF evidence; not the final holdout claim",
    }
    (output / "oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote leakage-safe OOF report for {len(samples)} points to {output}")


if __name__ == "__main__":
    main()

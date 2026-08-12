"""Leakage-safe Stage-5 fold preparation, fitting, and evaluation."""

from __future__ import annotations

import csv
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.bayesian_stage5 import Stage5Catalog, Stage5TraceRef
from llm_length_prediction.data.bayesian_trace import BayesianTraceV1
from llm_length_prediction.data.stage5 import (
    stage5_bayesian_sequence,
    stage5_hybrid_samples,
    stage5_prior_summary_matrix,
)
from llm_length_prediction.evaluation.sequential import run_bayesian_sequence
from llm_length_prediction.evaluation.stage5 import (
    compact_posterior_rows,
    run_prior_countdown_sequence,
)
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.models.bayesian_scorer import (
    BAYESIAN_METHOD_IDS,
    fit_bayesian_scorer,
    make_bayesian_checkpoint,
    save_bayesian_checkpoint,
)
from llm_length_prediction.models.dynamic import fit_plp_mlp
from llm_length_prediction.models.hybrid import (
    fit_progressive_head,
    fit_summary_scaler,
    hybrid_feature_matrix,
    predict_progressive_head,
)
from llm_length_prediction.models.prior import fit_grouped_oof_log1p_prior
from llm_length_prediction.models.prompt_token_baseline import (
    fit_prompt_token_ridge,
    predict_prompt_token_countdown,
)


def _references_for_fitting(catalog: Stage5Catalog, fold: int) -> list[Stage5TraceRef]:
    temperature = float(catalog.config["data_policy"]["training_temperature"])
    return [
        reference
        for reference in catalog.references
        if catalog.family_folds[reference.prompt_family_id] != fold
        and reference.temperature == temperature
    ]


def _references_for_validation(catalog: Stage5Catalog, fold: int) -> list[Stage5TraceRef]:
    return [
        reference
        for reference in catalog.references
        if catalog.family_folds[reference.prompt_family_id] == fold
    ]


def _load_traces(
    catalog: Stage5Catalog, references: list[Stage5TraceRef]
) -> list[tuple[Stage5TraceRef, BayesianTraceV1]]:
    return [(reference, catalog.load_trace(reference)) for reference in references]


def _inner_family_folds(
    traces: list[tuple[Stage5TraceRef, BayesianTraceV1]],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    family_tasks = {
        reference.prompt_family_id: reference.task for reference, _ in traces
    }
    from collections import defaultdict

    by_task: dict[str, list[str]] = defaultdict(list)
    for family, task in family_tasks.items():
        by_task[task].append(family)
    generator = np.random.default_rng(seed)
    assignments = {}
    for task in sorted(by_task):
        families = sorted(by_task[task])
        generator.shuffle(families)
        for index, family in enumerate(families):
            assignments[family] = index % folds
    if set(assignments.values()) != set(range(folds)):
        raise ValueError("inner prior cross-fit failed to cover every fold")
    return assignments


def _fit_priors(
    train_traces: list[tuple[Stage5TraceRef, BayesianTraceV1]],
    validation_traces: list[tuple[Stage5TraceRef, BayesianTraceV1]],
    *,
    inner_folds: int,
    seed: int,
    alpha: float,
) -> tuple[Any, dict[tuple[str, float, int], float], dict[tuple[str, float, int], float]]:
    features = np.stack([trace.prior_feature for _, trace in train_traces])
    lengths = np.asarray([trace.observed_tokens for _, trace in train_traces], dtype=np.int64)
    groups = [trace.prompt_family_id for _, trace in train_traces]
    family_folds = _inner_family_folds(train_traces, folds=inner_folds, seed=seed)
    row_folds = np.asarray([family_folds[group] for group in groups], dtype=np.int32)
    prior, oof_mu, _ = fit_grouped_oof_log1p_prior(
        features,
        lengths,
        groups,
        folds=inner_folds,
        alpha=alpha,
        seed=seed,
        fold_ids=row_folds,
    )
    train_mu = {
        reference.identity: float(value)
        for (reference, _), value in zip(train_traces, oof_mu, strict=True)
    }
    validation_mu = {
        reference.identity: prior.predict_mu(trace.prior_feature)
        for reference, trace in validation_traces
    }
    return prior, train_mu, validation_mu


def _progressive_common(config: dict[str, Any], device: str) -> dict[str, Any]:
    settings = config["baseline_training"]["progressive_heads"]
    return {
        "num_bins": int(settings["num_bins"]),
        "percentiles": tuple(float(value) for value in settings["target_range_percentiles"]),
        "lambda_ce": float(settings["lambda_ce"]),
        "dropout": float(settings["dropout"]),
        "epochs": int(settings["epochs"]),
        "batch_size": int(settings["batch_size"]),
        "learning_rate": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
        "seed": int(config["bayesian_training"]["seed"]),
        "device": device,
    }


def _fit_discriminative_baselines(
    train_traces: list[tuple[Stage5TraceRef, BayesianTraceV1]],
    validation_traces: list[tuple[Stage5TraceRef, BayesianTraceV1]],
    *,
    prior: Any,
    train_mu: dict[tuple[str, float, int], float],
    validation_mu: dict[tuple[str, float, int], float],
    config: dict[str, Any],
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_samples = [
        sample
        for _, trace in train_traces
        for sample in stage5_hybrid_samples(trace)
    ]
    validation_samples = [
        sample
        for _, trace in validation_traces
        for sample in stage5_hybrid_samples(trace)
    ]
    train_prior = stage5_prior_summary_matrix(
        train_samples, trace_mu=train_mu, variance=prior.residual_variance
    )
    validation_prior = stage5_prior_summary_matrix(
        validation_samples, trace_mu=validation_mu, variance=prior.residual_variance
    )
    prompt_ridge = fit_prompt_token_ridge(
        [trace.prompt_tokens for _, trace in train_traces],
        [trace.observed_tokens for _, trace in train_traces],
        alpha=float(config["prior"]["ridge_alpha"]),
    )
    validation_trace_by_key = {
        reference.identity: trace for reference, trace in validation_traces
    }
    prompt_countdown = predict_prompt_token_countdown(
        prompt_ridge,
        [
            validation_trace_by_key[sample.trace_key].prompt_tokens
            for sample in validation_samples
        ],
        [sample.step for sample in validation_samples],
    )
    dynamic_train = [sample for sample in train_samples if sample.remaining_tokens > 0]
    dynamic_counts = Counter(sample.trace_key for sample in dynamic_train)
    dynamic_config = config["baseline_training"]["dynamic_signal_mlp_v1"]
    dynamic_model, dynamic_report = fit_plp_mlp(
        [sample.dynamic_features for sample in dynamic_train],
        [sample.remaining_tokens for sample in dynamic_train],
        [1.0 / dynamic_counts[sample.trace_key] for sample in dynamic_train],
        hidden_sizes=tuple(int(value) for value in dynamic_config["hidden_sizes"]),
        dropout=float(dynamic_config["dropout"]),
        epochs=int(dynamic_config["epochs"]),
        batch_size=int(dynamic_config["batch_size"]),
        learning_rate=float(dynamic_config["learning_rate"]),
        weight_decay=float(dynamic_config["weight_decay"]),
        seed=int(config["bayesian_training"]["seed"]),
        device=device,
    )
    dynamic_predictions = dynamic_model.predict_remaining_many(
        [sample.dynamic_features for sample in validation_samples]
    )
    common = _progressive_common(config, device)
    settings = config["baseline_training"]["progressive_heads"]
    plp_features = np.stack([sample.plp_features for sample in train_samples])
    plp_head, plp_report = fit_progressive_head(
        train_samples,
        plp_features,
        hidden_dim=int(settings["plp_hidden_dim"]),
        terminal_zero=True,
        weighted_range=False,
        **common,
    )
    prior_weights = np.asarray(
        [sample.sequence_weight for sample in train_samples], dtype=np.float32
    )
    concat_scaler = fit_summary_scaler(train_prior, prior_weights)
    concat_head, concat_report = fit_progressive_head(
        train_samples,
        hybrid_feature_matrix(train_samples, train_prior, scaler=concat_scaler),
        hidden_dim=int(settings["concat_hidden_dim"]),
        terminal_zero=True,
        weighted_range=True,
        **common,
    )
    batch_size = int(settings["batch_size"])
    plp_predictions = predict_progressive_head(
        plp_head,
        np.stack([sample.plp_features for sample in validation_samples]),
        batch_size=batch_size,
        device=str(plp_report["device"]),
    )
    concat_predictions = predict_progressive_head(
        concat_head,
        hybrid_feature_matrix(validation_samples, validation_prior, scaler=concat_scaler),
        batch_size=batch_size,
        device=str(concat_report["device"]),
    )
    rows = []
    for index, sample in enumerate(validation_samples):
        rows.append(
            {
                "prompt_id": sample.prompt_id,
                "prompt_family_id": sample.prompt_family_id,
                "task": sample.task,
                "intended_length": sample.intended_length,
                "temperature": sample.temperature,
                "seed": sample.seed,
                "step": sample.step,
                "true_remaining": sample.remaining_tokens,
                "prompt_token_ridge_countdown": float(prompt_countdown[index]),
                "alps_countdown": float(validation_prior[index, 3]),
                "dynamic_signal_mlp_v1": float(dynamic_predictions[index]),
                "plp_terminal_zero_v3": float(plp_predictions[index]),
                "alps_plp_concat_v1": float(concat_predictions[index]),
            }
        )
    return rows, {
        "prompt_token_ridge_countdown": prompt_ridge.to_dict(),
        "dynamic_signal_mlp_v1": dynamic_report,
        "plp_terminal_zero_v3": plp_report,
        "alps_plp_concat_v1": concat_report,
    }


def run_stage5_fold(
    catalog: Stage5Catalog,
    *,
    fold: int,
    output_dir: Path,
    device: str,
    skip_discriminative_baselines: bool = False,
) -> dict[str, Any]:
    config = catalog.config
    fold_count = int(config["data_policy"]["outer_oof_folds"])
    if not 0 <= fold < fold_count:
        raise ValueError(f"fold must lie in 0..{fold_count - 1}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_refs = _references_for_fitting(catalog, fold)
    validation_refs = _references_for_validation(catalog, fold)
    train_traces = _load_traces(catalog, train_refs)
    validation_traces = _load_traces(catalog, validation_refs)
    prior_config = config["prior"]
    prior, train_mu, validation_mu = _fit_priors(
        train_traces,
        validation_traces,
        inner_folds=int(config["data_policy"]["inner_prior_crossfit_folds"]),
        seed=int(config["data_policy"]["fold_seed"]) + fold + 1,
        alpha=float(prior_config["ridge_alpha"]),
    )
    (output_dir / "prior.json").write_text(
        json.dumps(prior.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    baseline_rows: list[dict[str, Any]] = []
    baseline_reports: dict[str, Any] = {}
    if not skip_discriminative_baselines:
        baseline_rows, baseline_reports = _fit_discriminative_baselines(
            train_traces,
            validation_traces,
            prior=prior,
            train_mu=train_mu,
            validation_mu=validation_mu,
            config=config,
            device=device,
        )
        with (output_dir / "baseline_predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]))
            writer.writeheader()
            writer.writerows(baseline_rows)

    train_sequences = [
        stage5_bayesian_sequence(
            trace,
            prior_mu=train_mu[reference.identity],
            prior_log_variance=prior.residual_variance,
        )
        for reference, trace in train_traces
    ]
    validation_sequences = [
        stage5_bayesian_sequence(
            trace,
            prior_mu=validation_mu[reference.identity],
            prior_log_variance=prior.residual_variance,
        )
        for reference, trace in validation_traces
    ]
    training = config["bayesian_training"]
    method_reports = {}
    method_files = {}
    trace_sha256 = {
        f"{reference.prompt_id}|{reference.temperature:.3f}|{reference.seed}": (
            reference.trace_sha256
        )
        for reference in validation_refs
    }
    prior_prediction_path = output_dir / "alps_countdown_posterior_predictions.jsonl"
    with prior_prediction_path.open("w", encoding="utf-8") as handle:
        for sequence in validation_sequences:
            for row in compact_posterior_rows(
                "alps_countdown",
                run_prior_countdown_sequence(sequence),
                outer_fold=fold,
            ):
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    method_files["alps_countdown"] = {
        "predictions": prior_prediction_path.name,
        "predictions_sha256": file_sha256(prior_prediction_path),
    }
    for method_id in BAYESIAN_METHOD_IDS:
        scorer, training_report = fit_bayesian_scorer(
            train_sequences,
            method_id=method_id,
            hidden_projection_dim=int(training["hidden_projection_dim"]),
            hidden_dim=int(training["hidden_dim"]),
            dropout=float(training["dropout"]),
            epochs=int(training["epochs"]),
            sequence_batch_size=int(training["sequence_batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            terminal_bce_weight=float(training["terminal_bce_weight"]),
            stability_weight=float(
                training["posterior_total_variation_stability_weight"]
            ),
            seed=int(training["seed"]),
            device=device,
        )
        checkpoint = make_bayesian_checkpoint(
            scorer,
            contract_sha256=config["scientific_contract"]["sha256"],
            training_report=training_report,
        )
        checkpoint_path = save_bayesian_checkpoint(
            checkpoint, output_dir / f"{method_id}.pt"
        )
        prediction_path = output_dir / f"{method_id}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for sequence in validation_sequences:
                observations = run_bayesian_sequence(
                    sequence, scorer, device=str(training_report["device"])
                )
                for row in compact_posterior_rows(
                    method_id, observations, outer_fold=fold
                ):
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        method_reports[method_id] = training_report
        method_files[method_id] = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "predictions": prediction_path.name,
            "predictions_sha256": file_sha256(prediction_path),
        }
        del scorer, checkpoint
        gc.collect()
    report = {
        "schema_version": 1,
        "stage5_id": config["stage5_id"],
        "fold": fold,
        "status": "pass",
        "dataset_digest": catalog.dataset_digest,
        "training_temperature": config["data_policy"]["training_temperature"],
        "training_trace_count": len(train_traces),
        "validation_trace_count": len(validation_traces),
        "training_family_count": len({row.prompt_family_id for row in train_refs}),
        "validation_family_count": len(
            {row.prompt_family_id for row in validation_refs}
        ),
        "evaluated_temperatures": sorted({row.temperature for row in validation_refs}),
        "robustness_refit_performed": False,
        "prior_residual_variance": prior.residual_variance,
        "validation_trace_sha256": trace_sha256,
        "baseline_training_reports": baseline_reports,
        "bayesian_training_reports": method_reports,
        "files": method_files,
        "final_holdout_accessed": False,
    }
    (output_dir / "fold_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report

"""Frozen Stage-6 diagnostics over Stage-5 OOF predictions and checkpoints."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.bayesian_stage5 import load_stage5_catalog
from llm_length_prediction.data.stage5 import stage5_bayesian_sequence
from llm_length_prediction.evaluation.sequential import run_bayesian_sequence
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.models.bayesian_scorer import (
    load_bayesian_checkpoint,
    restore_bayesian_scorer,
)
from llm_length_prediction.models.prior import StandardizedRidgeLogNormalPrior

STAGE6_ID = "bayesian-sequential-v1-stage6-uncertainty-serving"
STAGE5_PREFIX = Path("artifacts/runs/bayesian_sequential_v1/stage5_oof")
STAGE4_REPORT = Path(
    "artifacts/runs/bayesian_sequential_v1/full_train/collection_report.json"
)
STAGE4_INDEX = Path(
    "artifacts/runs/bayesian_sequential_v1/full_train/collection_index.jsonl"
)


@dataclass(frozen=True)
class Stage6Sources:
    config: dict[str, Any]
    stage4_root: Path
    stage5_root: Path
    stage5_report: dict[str, Any]
    selection: dict[str, Any]
    stage4_rows: tuple[dict[str, Any], ...]

    @property
    def duration_by_sequence(self) -> dict[tuple[str, float, int], float]:
        return {
            (str(row["prompt_id"]), float(row["temperature"]), int(row["seed"])): float(
                row["duration_ms"]
            )
            for row in self.stage4_rows
        }

    @property
    def total_tokens_by_sequence(self) -> dict[tuple[str, float, int], int]:
        return {
            (str(row["prompt_id"]), float(row["temperature"]), int(row["seed"])): int(
                row["observed_tokens"]
            )
            for row in self.stage4_rows
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_stage6_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = _read_json(config_path)
    if payload.get("schema_version") != 1 or payload.get("stage6_id") != STAGE6_ID:
        raise ValueError("unsupported Stage-6 configuration")
    if payload.get("claim_scope") != "train_family_oof_diagnostic_not_final_holdout":
        raise ValueError("Stage-6 claim scope changed")
    policy = payload["data_policy"]
    forbidden = str(policy.get("new_final_holdout_access", ""))
    if "forbidden" not in forbidden:
        raise ValueError("Stage-6 cannot access a final holdout")
    if any(
        policy.get(name) is not False
        for name in ("model_refit", "threshold_tuning", "robustness_refit")
    ):
        raise ValueError("Stage-6 cannot fit models or tune thresholds")
    contract = payload["scientific_contract"]
    if file_sha256(contract["path"]) != contract["sha256"]:
        raise ValueError("scientific contract changed after Stage-6 freeze")
    stage5 = payload["stage5"]
    if file_sha256(stage5["config"]) != stage5["config_sha256"]:
        raise ValueError("Stage-5 configuration changed after Stage-6 freeze")
    schema = payload["report_schema"]
    if file_sha256(schema["path"]) != schema["sha256"]:
        raise ValueError("Stage-6 report schema changed after the analysis freeze")
    serving = payload["serving_replay"]
    kv = serving["kv_model_contract"]
    expected_bytes = (
        int(kv["num_hidden_layers"])
        * int(kv["num_key_value_heads"])
        * int(kv["head_dim"])
        * int(kv["key_value_tensors"])
        * int(kv["dtype_bytes"])
    )
    if expected_bytes != int(kv["bytes_per_output_token"]):
        raise ValueError("KV bytes-per-token contract is internally inconsistent")
    return payload


def _validate_stage5_manifest(stage5_root: Path, *, expected_sha256: str) -> int:
    manifest = stage5_root / "file_sha256.txt"
    if file_sha256(manifest) != expected_sha256:
        raise ValueError("Stage-5 file manifest digest changed")
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative_text = line.split(maxsplit=1)
        relative = Path(relative_text)
        try:
            within = relative.relative_to(STAGE5_PREFIX)
        except ValueError as error:
            raise ValueError("Stage-5 manifest path escaped its frozen root") from error
        path = stage5_root / within
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Stage-5 file failed digest validation: {within}")
        count += 1
    return count


def load_stage6_sources(
    config_path: str | Path,
    *,
    stage4_root: str | Path,
    stage5_root: str | Path,
    verify_stage5_files: bool = False,
) -> Stage6Sources:
    config = load_stage6_config(config_path)
    resolved_stage4 = Path(stage4_root).expanduser().resolve()
    resolved_stage5 = Path(stage5_root).expanduser().resolve()
    report_path = resolved_stage5 / "oof_report.json"
    selection_path = resolved_stage5 / "selection.json"
    if not report_path.is_file() or not selection_path.is_file():
        raise ValueError("Stage-5 root is missing OOF report or selection")
    frozen_stage5 = config["stage5"]
    if file_sha256(report_path) != frozen_stage5["oof_report_sha256"]:
        raise ValueError("Stage-5 OOF report digest changed")
    if file_sha256(selection_path) != frozen_stage5["selection_sha256"]:
        raise ValueError("Stage-5 selection digest changed")
    manifest_count = None
    if verify_stage5_files:
        manifest_count = _validate_stage5_manifest(
            resolved_stage5,
            expected_sha256=frozen_stage5["file_manifest_sha256"],
        )
    report = _read_json(report_path)
    selection = _read_json(selection_path)
    if (
        report.get("status") != "pass"
        or report.get("dataset_digest") != frozen_stage5["dataset_digest"]
        or report.get("final_holdout_accessed") is not False
        or report.get("robustness_refit_performed") is not False
        or len(report.get("fold_reports", [])) != frozen_stage5["required_fold_count"]
    ):
        raise ValueError("Stage-5 OOF report violates the frozen Stage-6 source contract")
    if (
        selection.get("selected_method") != frozen_stage5["selected_method"]
        or selection.get("final_holdout_selects_nothing") is not True
    ):
        raise ValueError("Stage-5 selected method changed")
    selected_metrics = report["methods"][frozen_stage5["selected_method"]][
        "all_temperatures"
    ]
    if (
        selected_metrics["sequence_count"] != frozen_stage5["required_sequence_count"]
        or selected_metrics["observation_count"]
        != frozen_stage5["required_observation_count"]
    ):
        raise ValueError("Stage-5 selected predictions lost frozen coverage")

    stage4 = config["stage4"]
    collection_report = resolved_stage4 / STAGE4_REPORT
    collection_index = resolved_stage4 / STAGE4_INDEX
    if file_sha256(collection_report) != stage4["collection_report_sha256"]:
        raise ValueError("Stage-4 collection report digest changed")
    if file_sha256(collection_index) != stage4["collection_index_sha256"]:
        raise ValueError("Stage-4 collection index digest changed")
    collection = _read_json(collection_report)
    rows = tuple(_read_jsonl(collection_index))
    if (
        collection.get("status") != "pass"
        or collection.get("valid_trace_count") != stage4["required_trace_count"]
        or collection.get("final_holdout_accessed") is not False
        or len(rows) != stage4["required_trace_count"]
    ):
        raise ValueError("Stage-4 source is incomplete or accessed a holdout")
    identities = {
        (str(row["prompt_id"]), float(row["temperature"]), int(row["seed"]))
        for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("Stage-4 collection index contains duplicate request identities")
    sources = Stage6Sources(
        config=config,
        stage4_root=resolved_stage4,
        stage5_root=resolved_stage5,
        stage5_report=report,
        selection=selection,
        stage4_rows=rows,
    )
    # Kept in the report by callers without changing the immutable dataclass contract.
    sources.config.setdefault("_runtime_validation", {})[
        "stage5_manifest_file_count"
    ] = manifest_count
    return sources


def load_posterior_rows(sources: Stage6Sources, method: str) -> list[dict[str, Any]]:
    rows = []
    for fold_report in sources.stage5_report["fold_reports"]:
        fold = int(fold_report["fold"])
        file_info = fold_report["files"][method]
        path = sources.stage5_root / "folds" / f"fold_{fold}" / file_info["predictions"]
        if file_sha256(path) != file_info["predictions_sha256"]:
            raise ValueError(f"Stage-5 prediction digest changed for {method} fold {fold}")
        rows.extend(_read_jsonl(path))
    return rows


def load_baseline_rows(sources: Stage6Sources) -> list[dict[str, Any]]:
    rows = []
    numeric = (
        "temperature",
        "prompt_token_ridge_countdown",
        "alps_countdown",
        "dynamic_signal_mlp_v1",
        "plp_terminal_zero_v3",
        "alps_plp_concat_v1",
    )
    integers = ("seed", "step", "true_remaining")
    for fold in range(int(sources.config["stage5"]["required_fold_count"])):
        path = sources.stage5_root / "folds" / f"fold_{fold}" / "baseline_predictions.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                converted: dict[str, Any] = dict(row)
                for name in numeric:
                    converted[name] = float(row[name])
                for name in integers:
                    converted[name] = int(row[name])
                converted["outer_fold"] = fold
                rows.append(converted)
    return rows


def sequence_id(row: dict[str, Any]) -> tuple[str, float, int]:
    return str(row["prompt_id"]), float(row["temperature"]), int(row["seed"])


def observation_id(row: dict[str, Any]) -> tuple[str, float, int, int]:
    return (*sequence_id(row), int(row["step"]))


def _progress(row: dict[str, Any]) -> float:
    total = int(row["step"]) + int(row["true_remaining"])
    return int(row["step"]) / max(total, 1)


def _progress_label(progress: float, config: dict[str, Any]) -> str:
    edges = config["analysis"]["progress_bin_edges"]
    labels = config["analysis"]["progress_bin_labels"]
    for lower, upper, label in zip(edges[:-1], edges[1:], labels, strict=True):
        if float(lower) <= progress < float(upper):
            return str(label)
    raise ValueError(f"progress lies outside frozen bins: {progress}")


def _finite_mean(values: Iterable[float | None]) -> float | None:
    array = np.asarray([math.nan if value is None else float(value) for value in values])
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else None


CURVE_METRICS = (
    "posterior_nll",
    "crps",
    "error_tokens",
    "posterior_variance_lower_bound",
    "posterior_entropy",
    "interval_50_coverage",
    "interval_50_width",
    "interval_90_coverage",
    "interval_90_width",
    "interval_95_coverage",
    "interval_95_width",
)


def uncertainty_curve_rows(
    methods: dict[str, Sequence[dict[str, Any]]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for method, rows in methods.items():
        sequence_bins: dict[tuple[tuple[str, float, int], str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for row in rows:
            sequence_bins[(sequence_id(row), _progress_label(_progress(row), config))].append(
                row
            )
        per_sequence = []
        for (identity, label), values in sequence_bins.items():
            item = {
                "method_id": method,
                "prompt_id": values[0]["prompt_id"],
                "prompt_family_id": values[0]["prompt_family_id"],
                "temperature": identity[1],
                "progress_bin": label,
                "observation_count": len(values),
            }
            for metric in CURVE_METRICS:
                item[metric] = _finite_mean(row.get(metric) for row in values)
            item["absolute_error_tokens"] = _finite_mean(
                abs(float(row["error_tokens"])) for row in values
            )
            per_sequence.append(item)
        groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
        for row in per_sequence:
            groups[(float(row["temperature"]), str(row["progress_bin"]))].append(row)
        label_order = {
            label: index
            for index, label in enumerate(config["analysis"]["progress_bin_labels"])
        }
        for (temperature, label), values in sorted(
            groups.items(), key=lambda item: (item[0][0], label_order[item[0][1]])
        ):
            item = {
                "method_id": method,
                "temperature": temperature,
                "progress_bin": label,
                "sequence_count": len(values),
                "observation_count": sum(int(row["observation_count"]) for row in values),
            }
            for metric in (*CURVE_METRICS, "absolute_error_tokens"):
                item[f"sequence_balanced_{metric}"] = _finite_mean(
                    row.get(metric) for row in values
                )
            output.append(item)
    return output


def uncertainty_cone_curve_rows(
    rows: Sequence[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    metrics = (
        "true_remaining",
        "posterior_mean_remaining_lower_bound",
        "posterior_q025_remaining",
        "posterior_median_remaining",
        "posterior_q975_remaining",
    )
    sequence_bins: dict[tuple[tuple[str, float, int], str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        label = _progress_label(float(row["decode_progress"]), config)
        sequence_bins[(sequence_id(row), label)].append(row)
    per_sequence = []
    for (identity, label), values in sequence_bins.items():
        item = {
            "temperature": identity[1],
            "progress_bin": label,
            "observation_count": len(values),
        }
        for metric in metrics:
            item[metric] = _finite_mean(float(row[metric]) for row in values)
        per_sequence.append(item)
    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_sequence:
        groups[(float(row["temperature"]), str(row["progress_bin"]))].append(row)
    label_order = {
        label: index
        for index, label in enumerate(config["analysis"]["progress_bin_labels"])
    }
    output = []
    for (temperature, label), values in sorted(
        groups.items(), key=lambda item: (item[0][0], label_order[item[0][1]])
    ):
        item = {
            "temperature": temperature,
            "progress_bin": label,
            "sequence_count": len(values),
            "observation_count": sum(int(row["observation_count"]) for row in values),
        }
        for metric in metrics:
            item[f"sequence_balanced_{metric}"] = _finite_mean(
                row[metric] for row in values
            )
        output.append(item)
    return output


def _posterior_quantile_with_boundary(
    probabilities: np.ndarray, quantile: float, *, has_overflow: bool
) -> tuple[float, bool]:
    values = np.asarray(probabilities, dtype=np.float64)
    index = int(np.searchsorted(np.cumsum(values), quantile, side="left"))
    overflow = bool(has_overflow and index == values.size - 1)
    return float(values.size - 1 if overflow else min(index, values.size - 1)), overflow


def replay_selected_uncertainty_cone(
    sources: Stage6Sources,
    selected_rows: Sequence[dict[str, Any]],
    *,
    verify_trace_hashes: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = sources.config
    stage5_config = config["stage5"]["config"]
    catalog = load_stage5_catalog(
        stage5_config,
        dataset_root=sources.stage4_root,
        verify_trace_hashes=verify_trace_hashes,
    )
    expected = {observation_id(row): float(row["predicted_remaining"]) for row in selected_rows}
    if len(expected) != len(selected_rows):
        raise ValueError("selected Stage-5 predictions contain duplicate observations")
    cone = []
    maximum_mean_difference = 0.0
    overflow_quantile_count = 0
    selected_method = config["stage5"]["selected_method"]
    for fold in range(int(config["stage5"]["required_fold_count"])):
        directory = sources.stage5_root / "folds" / f"fold_{fold}"
        prior = StandardizedRidgeLogNormalPrior.from_dict(_read_json(directory / "prior.json"))
        checkpoint = load_bayesian_checkpoint(directory / f"{selected_method}.pt")
        scorer = restore_bayesian_scorer(checkpoint, device="cpu")
        references = [
            reference
            for reference in catalog.references
            if catalog.family_folds[reference.prompt_family_id] == fold
        ]
        for reference in references:
            trace = catalog.load_trace(reference)
            sequence = stage5_bayesian_sequence(
                trace,
                prior_mu=prior.predict_mu(trace.prior_feature),
                prior_log_variance=prior.residual_variance,
            )
            observations = run_bayesian_sequence(sequence, scorer, device="cpu")
            for observation in observations:
                key = (
                    observation.prompt_id,
                    float(observation.temperature),
                    int(observation.seed),
                    int(observation.step),
                )
                if key not in expected:
                    raise ValueError(f"checkpoint replay produced an unknown observation: {key}")
                difference = abs(
                    observation.summary.mean_remaining_lower_bound - expected[key]
                )
                maximum_mean_difference = max(maximum_mean_difference, difference)
                lower, lower_overflow = _posterior_quantile_with_boundary(
                    observation.probabilities, 0.025, has_overflow=observation.has_overflow
                )
                median, median_overflow = _posterior_quantile_with_boundary(
                    observation.probabilities, 0.5, has_overflow=observation.has_overflow
                )
                upper, upper_overflow = _posterior_quantile_with_boundary(
                    observation.probabilities, 0.975, has_overflow=observation.has_overflow
                )
                overflow_quantile_count += int(
                    lower_overflow or median_overflow or upper_overflow
                )
                true_remaining = int(observation.true_remaining)  # Stage 4 is all EOS.
                cone.append(
                    {
                        "method_id": selected_method,
                        "prompt_id": observation.prompt_id,
                        "prompt_family_id": observation.prompt_family_id,
                        "task": observation.task,
                        "intended_length": observation.intended_length,
                        "temperature": observation.temperature,
                        "seed": observation.seed,
                        "outer_fold": fold,
                        "step": observation.step,
                        "decode_progress": observation.step
                        / max(observation.step + true_remaining, 1),
                        "true_remaining": true_remaining,
                        "posterior_mean_remaining_lower_bound": (
                            observation.summary.mean_remaining_lower_bound
                        ),
                        "posterior_q025_remaining": lower,
                        "posterior_median_remaining": median,
                        "posterior_q975_remaining": upper,
                        "q975_used_overflow_boundary": upper_overflow,
                    }
                )
        del scorer
    if len(cone) != len(expected):
        raise ValueError("checkpoint replay did not reproduce every selected prediction")
    if maximum_mean_difference > 1e-3:
        raise ValueError(
            "checkpoint replay differs from frozen Stage-5 means: "
            f"maximum difference {maximum_mean_difference}"
        )
    return cone, {
        "observation_count": len(cone),
        "maximum_stage5_mean_replay_difference": maximum_mean_difference,
        "overflow_boundary_quantile_observation_count": overflow_quantile_count,
        "trace_hashes_verified_during_replay": verify_trace_hashes,
    }


def _quantiles(values: Sequence[float], quantiles: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {f"q{int(round(value * 100)):02d}": None for value in quantiles}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"q{int(round(value * 100)):02d}": float(np.quantile(array, value))
        for value in quantiles
    }


def convergence_metrics(
    rows: Sequence[dict[str, Any]], *, threshold: float, group: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    sequences: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequences[sequence_id(row)].append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for values in sequences.values():
        ordered = sorted(values, key=lambda row: int(row["step"]))
        truth = int(ordered[0]["step"]) + int(ordered[0]["true_remaining"])
        within = [
            abs(int(row["step"]) + float(row["predicted_remaining"]) - truth)
            / max(truth, 1)
            <= threshold
            for row in ordered
        ]
        stable_index = None
        suffix = True
        for index in range(len(ordered) - 1, -1, -1):
            suffix = suffix and within[index]
            if suffix:
                stable_index = index
        grouped[group(ordered[0])].append(
            {
                "success": stable_index is not None,
                "step": None if stable_index is None else int(ordered[stable_index]["step"]),
                "progress": (
                    None
                    if stable_index is None
                    else int(ordered[stable_index]["step"]) / max(truth, 1)
                ),
            }
        )
    output = {}
    for label, values in sorted(grouped.items()):
        successes = [value for value in values if value["success"]]
        steps = [float(value["step"]) for value in successes]
        progress = [float(value["progress"]) for value in successes]
        output[label] = {
            "sequence_count": len(values),
            "success_count": len(successes),
            "failure_count": len(values) - len(successes),
            "success_rate": len(successes) / len(values),
            "stable_step_mean_on_success": _finite_mean(steps),
            "stable_step_quantiles_on_success": _quantiles(steps, (0.5, 0.9)),
            "stable_progress_mean_on_success": _finite_mean(progress),
            "stable_progress_quantiles_on_success": _quantiles(progress, (0.5, 0.9)),
        }
    return output


def _sequence_balanced_error_metrics(
    rows: Sequence[dict[str, Any]], *, prediction_field: str, severe_threshold: float = 100.0
) -> dict[str, Any]:
    sequences: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequences[sequence_id(row)].append(row)
    sequence_values = []
    for values in sequences.values():
        errors = np.asarray(
            [float(row[prediction_field]) - int(row["true_remaining"]) for row in values]
        )
        under = np.maximum(-errors, 0.0)
        sequence_values.append(
            {
                "mae": float(np.abs(errors).mean()),
                "bias": float(errors.mean()),
                "under": float(under.mean()),
                "severe": float((under > severe_threshold).mean()),
            }
        )
    return {
        "sequence_count": len(sequence_values),
        "sequence_balanced_mae_tokens": _finite_mean(row["mae"] for row in sequence_values),
        "sequence_balanced_bias_tokens": _finite_mean(row["bias"] for row in sequence_values),
        "sequence_balanced_positive_underestimation_tokens": _finite_mean(
            row["under"] for row in sequence_values
        ),
        "sequence_balanced_severe_underestimation_rate_100_tokens": _finite_mean(
            row["severe"] for row in sequence_values
        ),
    }


def long_tail_metrics(
    posterior_methods: dict[str, Sequence[dict[str, Any]]],
    baseline_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    primary_temperature = float(config["analysis"]["primary_temperature"])
    selected = posterior_methods[config["stage5"]["selected_method"]]
    totals_by_sequence = {
        sequence_id(row): int(row["step"]) + int(row["true_remaining"])
        for row in selected
        if float(row["temperature"]) == primary_temperature
    }
    threshold = int(
        np.quantile(
            np.asarray(list(totals_by_sequence.values())),
            float(config["analysis"]["long_tail"]["quantile"]),
            method=str(config["analysis"]["long_tail"]["quantile_method"]),
        )
    )
    early_max = float(config["analysis"]["long_tail"]["early_progress_maximum"])
    severe_threshold = float(
        config["analysis"]["long_tail"]["severe_underestimation_tokens"]
    )
    method_sources: dict[str, tuple[Sequence[dict[str, Any]], str]] = {
        name: (rows, "predicted_remaining") for name, rows in posterior_methods.items()
    }
    for name in ("plp_terminal_zero_v3", "alps_plp_concat_v1"):
        method_sources[name] = (baseline_rows, name)
    output = {}
    for method, (rows, field) in method_sources.items():
        primary_tail = [
            row
            for row in rows
            if float(row["temperature"]) == primary_temperature
            and int(row["step"]) + int(row["true_remaining"]) >= threshold
            and _progress(row) <= early_max
        ]
        intended_long = [
            row
            for row in rows
            if float(row["temperature"]) == primary_temperature
            and str(row["intended_length"]) == "long"
            and _progress(row) <= early_max
        ]
        by_temperature = {}
        for temperature in config["analysis"]["evaluation_temperatures"]:
            values = [
                row
                for row in rows
                if float(row["temperature"]) == float(temperature)
                and int(row["step"]) + int(row["true_remaining"]) >= threshold
                and _progress(row) <= early_max
            ]
            by_temperature[f"{float(temperature):.1f}"] = _sequence_balanced_error_metrics(
                values,
                prediction_field=field,
                severe_threshold=severe_threshold,
            )
        output[method] = {
            "primary_temperature_empirical_top_decile_early": _sequence_balanced_error_metrics(
                primary_tail,
                prediction_field=field,
                severe_threshold=severe_threshold,
            ),
            "primary_temperature_intended_long_early": _sequence_balanced_error_metrics(
                intended_long,
                prediction_field=field,
                severe_threshold=severe_threshold,
            ),
            "empirical_tail_early_by_temperature": by_temperature,
        }
    return {
        "empirical_top_decile_threshold_tokens_from_primary_temperature": threshold,
        "early_progress_maximum": early_max,
        "severe_underestimation_threshold_tokens": severe_threshold,
        "methods": output,
    }


def uncertainty_findings(
    curve_rows: Sequence[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    primary = float(config["analysis"]["primary_temperature"])
    tolerance = float(config["analysis"]["coverage_tolerance_absolute"])
    selected = str(config["stage5"]["selected_method"])
    labels = list(config["analysis"]["progress_bin_labels"])
    rows = {
        str(row["progress_bin"]): row
        for row in curve_rows
        if row["method_id"] == selected and float(row["temperature"]) == primary
    }
    if set(rows) != set(labels):
        raise ValueError("selected method lacks a primary-temperature uncertainty bin")
    first = rows[labels[0]]
    last = rows[labels[-1]]
    variance_first = float(
        first["sequence_balanced_posterior_variance_lower_bound"]
    )
    variance_last = float(last["sequence_balanced_posterior_variance_lower_bound"])
    entropy_first = float(first["sequence_balanced_posterior_entropy"])
    entropy_last = float(last["sequence_balanced_posterior_entropy"])
    coverage = {}
    for level in (50, 90, 95):
        nominal = level / 100.0
        values = {
            label: float(rows[label][f"sequence_balanced_interval_{level}_coverage"])
            for label in labels
        }
        coverage[str(level)] = {
            "nominal": nominal,
            "tolerance_absolute": tolerance,
            "by_progress": values,
            "all_progress_bins_within_tolerance": all(
                abs(value - nominal) <= tolerance for value in values.values()
            ),
            "maximum_absolute_gap": max(abs(value - nominal) for value in values.values()),
        }
    return {
        "selected_method": selected,
        "primary_temperature": primary,
        "variance_first_progress_bin": variance_first,
        "variance_last_progress_bin": variance_last,
        "variance_last_to_first_ratio": variance_last / variance_first,
        "entropy_first_progress_bin": entropy_first,
        "entropy_last_progress_bin": entropy_last,
        "entropy_last_to_first_ratio": entropy_last / entropy_first,
        "variance_decreased_first_to_last": variance_last < variance_first,
        "entropy_decreased_first_to_last": entropy_last < entropy_first,
        "coverage": coverage,
        "uncertainty_success_requires_joint_width_and_coverage_interpretation": True,
    }


def runtime_metrics(
    selected_rows: Sequence[dict[str, Any]], sources: Stage6Sources
) -> dict[str, Any]:
    groups: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        groups[sequence_id(row)].append(row)
    durations = sources.duration_by_sequence
    cumulative = []
    ratios = []
    point_times = []
    for identity, rows in groups.items():
        update = sum(float(row["update_wall_time_ms"]) for row in rows)
        duration = durations[identity]
        cumulative.append(update)
        ratios.append(update / duration if duration > 0 else math.nan)
        point_times.extend(float(row["update_wall_time_ms"]) for row in rows)
    prediction_bytes = 0
    method = sources.config["stage5"]["selected_method"]
    for report in sources.stage5_report["fold_reports"]:
        fold = int(report["fold"])
        prediction_bytes += (
            sources.stage5_root
            / "folds"
            / f"fold_{fold}"
            / report["files"][method]["predictions"]
        ).stat().st_size
    return {
        "source": sources.config["analysis"]["runtime"],
        "update_point_count": len(point_times),
        "update_wall_time_ms": {
            "mean": _finite_mean(point_times),
            **_quantiles(point_times, (0.5, 0.95, 0.99)),
        },
        "cumulative_update_wall_time_ms_per_sequence": {
            "mean": _finite_mean(cumulative),
            **_quantiles(cumulative, (0.5, 0.95, 0.99)),
        },
        "predictor_to_real_qwen_generation_duration_ratio": {
            "mean": _finite_mean(ratios),
            **_quantiles(ratios, (0.5, 0.95, 0.99)),
        },
        "peak_predictor_state_bytes": max(
            int(row["predictor_state_bytes"]) for row in selected_rows
        ),
        "compact_prediction_artifact_bytes": prediction_bytes,
        "compact_prediction_bytes_per_observation": prediction_bytes / len(selected_rows),
    }


def _ceil_quantum(value: float, *, quantum: int, cap: int) -> int:
    return min(cap, max(quantum, int(math.ceil(max(value, 1.0) / quantum) * quantum)))


def serving_replay(
    *,
    sources: Stage6Sources,
    posterior_methods: dict[str, Sequence[dict[str, Any]]],
    baseline_rows: Sequence[dict[str, Any]],
    cone_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    config = sources.config["serving_replay"]
    step = int(config["prediction_step"])
    duration = sources.duration_by_sequence
    actual = sources.total_tokens_by_sequence
    scalar_method = sources.config["stage5"]["selected_method"]
    scalar_step = {
        sequence_id(row): float(row["predicted_remaining"]) + step
        for row in posterior_methods[scalar_method]
        if int(row["step"]) == step
    }
    alps_step = {
        sequence_id(row): float(row["predicted_remaining"]) + step
        for row in posterior_methods["alps_countdown"]
        if int(row["step"]) == step
    }
    baseline_step = {
        sequence_id(row): row for row in baseline_rows if int(row["step"]) == step
    }
    q975_step = {
        sequence_id(row): float(row["posterior_q975_remaining"]) + step
        for row in cone_rows
        if int(row["step"]) == step
    }
    identities = set(actual)
    maps = (scalar_step, alps_step, baseline_step, q975_step)
    if any(set(mapping) != identities for mapping in maps):
        raise ValueError("serving replay inputs do not cover identical requests")
    predictors: dict[str, Callable[[tuple[str, float, int]], float]] = {
        "oracle_observed_length": lambda identity: float(actual[identity]),
        "max_new_tokens_4096": lambda identity: 4096.0,
        "alps_countdown_mean": lambda identity: alps_step[identity],
        "plp_terminal_zero_v3": lambda identity: step
        + float(baseline_step[identity]["plp_terminal_zero_v3"]),
        "alps_plp_concat_v1": lambda identity: step
        + float(baseline_step[identity]["alps_plp_concat_v1"]),
        "bayesian_entropy_scalar_v1_mean": lambda identity: scalar_step[identity],
        "bayesian_entropy_scalar_v1_q975": lambda identity: q975_step[identity],
    }
    if list(predictors) != config["strategies"]:
        raise ValueError("serving strategy implementation differs from the frozen order")
    batch_size = int(config["batch_size"])
    boundaries = tuple(int(value) for value in config["length_bucket_boundaries"])
    quantum = int(config["kv_allocation_quantum_tokens"])
    cap = max(boundaries)
    bytes_per_token = int(config["kv_model_contract"]["bytes_per_output_token"])
    budget = int(float(config["incremental_kv_budget_gib"]) * (1024**3))
    metrics = {}
    for name, predictor in predictors.items():
        requests = []
        for identity in identities:
            predicted = predictor(identity)
            allocated = _ceil_quantum(predicted, quantum=quantum, cap=cap)
            bucket = next(
                (index for index, boundary in enumerate(boundaries) if predicted <= boundary),
                len(boundaries),
            )
            requests.append(
                {
                    "identity": identity,
                    "actual": actual[identity],
                    "duration": duration[identity],
                    "predicted": predicted,
                    "allocated": allocated,
                    "bucket": bucket,
                }
            )
        requests.sort(key=lambda row: (row["bucket"], *row["identity"]))
        clock = 0.0
        completion = []
        peak_batch_bytes = 0
        budget_exceedance = 0
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            clock += max(float(row["duration"]) for row in batch)
            completion.extend([clock] * len(batch))
            batch_bytes = sum(int(row["allocated"]) for row in batch) * bytes_per_token
            peak_batch_bytes = max(peak_batch_bytes, batch_bytes)
            budget_exceedance += int(batch_bytes > budget)
        actual_vector = np.asarray([row["actual"] for row in requests], dtype=np.float64)
        allocated_vector = np.asarray([row["allocated"] for row in requests], dtype=np.float64)
        over = np.maximum(allocated_vector - actual_vector, 0.0)
        deficit = np.maximum(actual_vector - allocated_vector, 0.0)
        completion_vector = np.asarray(completion, dtype=np.float64)
        metrics[name] = {
            "request_count": len(requests),
            "batch_count": math.ceil(len(requests) / batch_size),
            "mean_completion_time_ms": float(completion_vector.mean()),
            "p95_completion_time_ms": float(np.quantile(completion_vector, 0.95)),
            "makespan_ms": clock,
            "throughput_tokens_per_second": float(actual_vector.sum() / (clock / 1000.0)),
            "kv_overreservation_tokens": int(over.sum()),
            "kv_overreservation_bytes": int(over.sum()) * bytes_per_token,
            "kv_overreservation_rate": float(over.sum() / actual_vector.sum()),
            "underallocation_request_count": int((deficit > 0).sum()),
            "underallocation_rate": float((deficit > 0).mean()),
            "total_underallocated_tokens": int(deficit.sum()),
            "peak_batch_incremental_kv_bytes": peak_batch_bytes,
            "incremental_kv_budget_exceedance_batch_count": budget_exceedance,
        }
    return {
        "scope": config["scope"],
        "simulator": {
            key: value
            for key, value in config.items()
            if key not in {"strategies", "metrics"}
        },
        "metrics": metrics,
        "serving_superiority_claimed": False,
        "final_holdout_accessed": False,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strict_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

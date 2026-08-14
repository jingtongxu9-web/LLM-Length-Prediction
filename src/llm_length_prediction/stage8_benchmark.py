"""One-time, no-selection evaluation of the frozen Stage-8 final holdout."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.data.bayesian_trace import BayesianTraceV1, read_bayesian_trace
from llm_length_prediction.data.stage5 import (
    stage5_bayesian_sequence,
    stage5_hybrid_samples,
    stage5_prior_summary_matrix,
)
from llm_length_prediction.evaluation.sequential import run_bayesian_sequence
from llm_length_prediction.evaluation.stage5 import (
    compact_posterior_rows,
    posterior_breakdowns,
    posterior_metrics,
    run_prior_countdown_sequence,
)
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.models.bayesian_scorer import BAYESIAN_METHOD_IDS
from llm_length_prediction.models.hybrid import (
    hybrid_feature_matrix,
    predict_progressive_head,
)
from llm_length_prediction.models.prompt_token_baseline import (
    predict_prompt_token_countdown,
)
from llm_length_prediction.stage6_analysis import convergence_metrics
from llm_length_prediction.stage8_freeze import FinalModels, load_final_models
from llm_length_prediction.stage8_holdout import (
    final_holdout_jobs,
    load_final_holdout_contract,
    validate_final_holdout_trace,
)

BENCHMARK_ID = "bayesian-sequential-v1-one-time-final-benchmark"
POINT_METHODS = (
    "prompt_token_ridge_countdown",
    "dynamic_signal_mlp_v1",
    "plp_terminal_zero_v3",
    "alps_plp_concat_v1",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty final benchmark CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_final_holdout_traces(
    config_path: str | Path,
    *,
    verify_trace_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[BayesianTraceV1], dict[Any, float]]:
    config, lock, records = load_final_holdout_contract(config_path)
    lock_path = Path(config["holdout_gate"]["benchmark_lock"])
    lock_sha256 = file_sha256(lock_path)
    manifest_sha256 = str(lock["final_holdout_manifest_sha256"])
    collection_root = Path(config["outputs"]["collection_run_root"])
    report = _read_json(collection_root / "collection_report.json")
    if (
        report.get("status") != "pass"
        or report.get("valid_trace_count") != config["final_holdout_plan"]["expected_rollout_count"]
        or report.get("stage8a_config_sha256") != config["_config_sha256"]
        or report.get("stage8b_lock_sha256") != lock_sha256
        or report.get("final_holdout_manifest_sha256") != manifest_sha256
        or report.get("final_holdout_accessed") is not True
    ):
        raise ValueError("final holdout collection report is incomplete or changed")
    index = _read_jsonl(collection_root / "collection_index.jsonl")
    jobs = final_holdout_jobs(config, records)
    by_rank = {int(row["job_rank"]): row for row in index}
    if set(by_rank) != set(range(len(jobs))):
        raise ValueError("final collection index does not cover every frozen job")
    traces = []
    durations = {}
    for job in jobs:
        row = by_rank[job.rank]
        path = Path(str(row["trace_path"]))
        if not path.is_file():
            raise ValueError(f"final trace is missing: {path}")
        if verify_trace_hashes and file_sha256(path) != row["trace_sha256"]:
            raise ValueError(f"final trace digest changed: {path}")
        trace = read_bayesian_trace(path)
        validate_final_holdout_trace(
            trace,
            job=job,
            config=config,
            lock_sha256=lock_sha256,
            manifest_sha256=manifest_sha256,
        )
        traces.append(trace)
        durations[(trace.prompt_id, trace.temperature, trace.seed)] = trace.duration_ms
    return config, lock, traces, durations


def _point_rows(
    traces: Sequence[BayesianTraceV1], models: FinalModels, config: dict[str, Any]
) -> list[dict[str, Any]]:
    samples = [sample for trace in traces for sample in stage5_hybrid_samples(trace)]
    trace_by_key = {(trace.prompt_id, trace.temperature, trace.seed): trace for trace in traces}
    trace_mu = {
        key: models.alps_prior.predict_mu(trace.prior_feature)
        for key, trace in trace_by_key.items()
    }
    prior = stage5_prior_summary_matrix(
        samples,
        trace_mu=trace_mu,
        variance=models.alps_prior.residual_variance,
    )
    prompt = predict_prompt_token_countdown(
        models.prompt_token_ridge,
        [trace_by_key[sample.trace_key].prompt_tokens for sample in samples],
        [sample.step for sample in samples],
    )
    dynamic = models.dynamic_signal_mlp.predict_remaining_many(
        [sample.dynamic_features for sample in samples]
    )
    stage5 = _read_json(Path(config["stage5"]["config"]))
    batch_size = int(stage5["baseline_training"]["progressive_heads"]["batch_size"])
    plp = predict_progressive_head(
        models.plp_head,
        np.stack([sample.plp_features for sample in samples]),
        batch_size=batch_size,
        device=models.device,
    )
    concat = predict_progressive_head(
        models.concat_head,
        hybrid_feature_matrix(samples, prior, scaler=models.concat_scaler),
        batch_size=batch_size,
        device=models.device,
    )
    output = []
    for index, sample in enumerate(samples):
        trace = trace_by_key[sample.trace_key]
        truth = None if trace.is_censored else sample.remaining_tokens
        output.append(
            {
                "prompt_id": sample.prompt_id,
                "prompt_family_id": sample.prompt_family_id,
                "task": sample.task,
                "intended_length": sample.intended_length,
                "temperature": sample.temperature,
                "seed": sample.seed,
                "step": sample.step,
                "true_remaining": truth,
                "censored_after_remaining": (
                    sample.remaining_tokens if trace.is_censored else None
                ),
                "terminal_observed": bool(
                    trace.terminal_observed and sample.step == trace.observed_tokens
                ),
                "prompt_token_ridge_countdown": float(prompt[index]),
                "dynamic_signal_mlp_v1": float(dynamic[index]),
                "plp_terminal_zero_v3": float(plp[index]),
                "alps_plp_concat_v1": float(concat[index]),
            }
        )
    return output


def _sequence_point_metrics(rows: Sequence[dict[str, Any]], method: str) -> dict[str, Any]:
    groups: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["true_remaining"] is not None:
            groups[(row["prompt_id"], float(row["temperature"]), int(row["seed"]))].append(row)
    sequences = []
    raw_truth = []
    raw_prediction = []
    for values in groups.values():
        errors = np.asarray(
            [float(row[method]) - int(row["true_remaining"]) for row in values],
            dtype=np.float64,
        )
        sequences.append(
            {
                "mae": float(np.abs(errors).mean()),
                "mse": float(np.square(errors).mean()),
                "bias": float(errors.mean()),
                "under": float(np.maximum(-errors, 0.0).mean()),
                "severe": float((errors < -100.0).mean()),
            }
        )
        raw_truth.extend(int(row["true_remaining"]) for row in values)
        raw_prediction.extend(float(row[method]) for row in values)
    if not sequences:
        return {
            "exact_sequence_count": 0,
            "censored_sequence_count": len(
                {(row["prompt_id"], row["temperature"], row["seed"]) for row in rows}
            ),
        }
    actual = np.asarray(raw_truth, dtype=np.float64)
    predicted = np.asarray(raw_prediction, dtype=np.float64)
    denominator = float(np.square(actual - actual.mean()).sum())
    all_sequences = {(row["prompt_id"], row["temperature"], row["seed"]) for row in rows}
    return {
        "observation_count": sum(len(values) for values in groups.values()),
        "exact_sequence_count": len(sequences),
        "censored_sequence_count": len(all_sequences) - len(sequences),
        "sequence_balanced_mae_tokens": float(np.mean([row["mae"] for row in sequences])),
        "sequence_balanced_rmse_tokens": math.sqrt(
            float(np.mean([row["mse"] for row in sequences]))
        ),
        "sequence_balanced_bias_tokens": float(np.mean([row["bias"] for row in sequences])),
        "raw_r_squared_tokens": 0.0
        if denominator == 0
        else 1.0 - float(np.square(actual - predicted).sum()) / denominator,
        "sequence_balanced_positive_underestimation_tokens": float(
            np.mean([row["under"] for row in sequences])
        ),
        "sequence_balanced_severe_underestimation_rate_100_tokens": float(
            np.mean([row["severe"] for row in sequences])
        ),
        "censored_point_targets_excluded_not_imputed": True,
    }


def _point_breakdowns(rows: Sequence[dict[str, Any]], method: str) -> dict[str, Any]:
    groupers: dict[str, Callable[[dict[str, Any]], str]] = {
        "by_task": lambda row: str(row["task"]),
        "by_intended_length": lambda row: str(row["intended_length"]),
        "by_task_and_intended_length": lambda row: f"{row['task']}:{row['intended_length']}",
        "by_temperature": lambda row: f"{float(row['temperature']):.3f}",
        "by_seed": lambda row: str(row["seed"]),
        "terminal_vs_nonterminal": lambda row: "terminal"
        if row["terminal_observed"]
        else "nonterminal",
    }
    output = {}
    for label, grouper in groupers.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[grouper(row)].append(row)
        output[label] = [
            {"group": name, **_sequence_point_metrics(values, method)}
            for name, values in sorted(groups.items())
        ]
    progress: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["true_remaining"] is None:
            label = "censored"
        else:
            fraction = int(row["step"]) / max(int(row["step"]) + int(row["true_remaining"]), 1)
            label = (
                "0-10%"
                if fraction < 0.1
                else "10-25%"
                if fraction < 0.25
                else "25-50%"
                if fraction < 0.5
                else "50-75%"
                if fraction < 0.75
                else "75-100%"
            )
        progress[label].append(row)
    output["by_decode_progress"] = [
        {"group": label, **_sequence_point_metrics(values, method)}
        for label, values in sorted(progress.items())
    ]
    return output


def _posterior_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    point_view = [{**row, "posterior_point_prediction": row["predicted_remaining"]} for row in rows]
    point = _sequence_point_metrics(point_view, "posterior_point_prediction")
    extras = {
        name: point[name]
        for name in (
            "exact_sequence_count",
            "censored_sequence_count",
            "raw_r_squared_tokens",
            "sequence_balanced_positive_underestimation_tokens",
            "sequence_balanced_severe_underestimation_rate_100_tokens",
            "censored_point_targets_excluded_not_imputed",
        )
        if name in point
    }
    return {**posterior_metrics(rows), **extras}


def _posterior_breakdowns(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    probability = posterior_breakdowns(rows)
    probability.pop("by_outer_fold", None)
    point_view = [{**row, "posterior_point_prediction": row["predicted_remaining"]} for row in rows]
    point = _point_breakdowns(point_view, "posterior_point_prediction")
    for breakdown, values in probability.items():
        point_by_group = {row["group"]: row for row in point[breakdown]}
        for row in values:
            extras = point_by_group[row["group"]]
            for name in (
                "exact_sequence_count",
                "censored_sequence_count",
                "raw_r_squared_tokens",
                "sequence_balanced_positive_underestimation_tokens",
                "sequence_balanced_severe_underestimation_rate_100_tokens",
                "censored_point_targets_excluded_not_imputed",
            ):
                if name in extras:
                    row[name] = extras[name]
    return probability


def _quantile(probabilities: np.ndarray, value: float, *, overflow: bool) -> tuple[float, bool]:
    index = int(np.searchsorted(np.cumsum(probabilities), value, side="left"))
    used_boundary = bool(overflow and index == len(probabilities) - 1)
    return float(len(probabilities) - 1 if used_boundary else index), used_boundary


def _convergence_with_censoring(
    exact_rows: Sequence[dict[str, Any]],
    all_rows: Sequence[dict[str, Any]],
    *,
    group: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    output = convergence_metrics(exact_rows, threshold=0.05, group=group)
    censored: dict[str, set[tuple[str, float, int]]] = defaultdict(set)
    for row in all_rows:
        if row["true_remaining"] is None:
            censored[group(row)].add(
                (row["prompt_id"], float(row["temperature"]), int(row["seed"]))
            )
    for label, identities in censored.items():
        count = len(identities)
        if label not in output:
            output[label] = {
                "sequence_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "stable_step_mean_on_success": None,
                "stable_step_quantiles_on_success": {"q50": None, "q90": None},
                "stable_progress_mean_on_success": None,
                "stable_progress_quantiles_on_success": {
                    "q50": None,
                    "q90": None,
                },
            }
        values = output[label]
        values["sequence_count"] += count
        values["failure_count"] += count
        values["success_rate"] = values["success_count"] / values["sequence_count"]
        values["right_censored_counted_as_failure"] = count
    for values in output.values():
        values.setdefault("right_censored_counted_as_failure", 0)
    return output


def _cone_rows(observations: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for observation in observations:
        lower, _ = _quantile(observation.probabilities, 0.025, overflow=observation.has_overflow)
        median, _ = _quantile(observation.probabilities, 0.5, overflow=observation.has_overflow)
        upper, boundary = _quantile(
            observation.probabilities, 0.975, overflow=observation.has_overflow
        )
        rows.append(
            {
                "method_id": "bayesian_entropy_scalar_v1",
                "prompt_id": observation.prompt_id,
                "prompt_family_id": observation.prompt_family_id,
                "task": observation.task,
                "intended_length": observation.intended_length,
                "temperature": observation.temperature,
                "seed": observation.seed,
                "step": observation.step,
                "true_remaining": observation.true_remaining,
                "posterior_mean_remaining_lower_bound": (
                    observation.summary.mean_remaining_lower_bound
                ),
                "posterior_q025_remaining": lower,
                "posterior_median_remaining": median,
                "posterior_q975_remaining": upper,
                "q975_used_overflow_boundary": boundary,
            }
        )
    return rows


def _family_bootstrap(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    left_value: Callable[[dict[str, Any]], float | None],
    right_value: Callable[[dict[str, Any]], float | None],
    config: dict[str, Any],
) -> dict[str, Any]:
    def values(
        rows: Sequence[dict[str, Any]], getter: Callable[[dict[str, Any]], float | None]
    ) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = getter(row)
            if value is not None and math.isfinite(value):
                grouped[str(row["prompt_family_id"])].append(value)
        return {family: float(np.mean(vector)) for family, vector in grouped.items()}

    left_family = values(left, left_value)
    right_family = values(right, right_value)
    if left_family.keys() != right_family.keys():
        raise ValueError("paired final comparison does not cover identical families")
    families = sorted(left_family)
    differences = np.asarray([left_family[name] - right_family[name] for name in families])
    frozen = config["benchmark"]["paired_bootstrap"]
    generator = np.random.default_rng(int(frozen["seed"]))
    draws = generator.integers(0, len(families), size=(int(frozen["replicates"]), len(families)))
    statistics = differences[draws].mean(axis=1)
    confidence = float(frozen["confidence_level"])
    alpha = 1.0 - confidence
    return {
        "estimate_left_minus_right": float(differences.mean()),
        "lower": float(np.quantile(statistics, alpha / 2)),
        "upper": float(np.quantile(statistics, 1 - alpha / 2)),
        "confidence_level": confidence,
        "replicates": int(frozen["replicates"]),
        "family_count": len(families),
        "unit": "prompt_family_id",
        "descriptive_not_selection": True,
    }


def _ceil_quantum(value: float, *, quantum: int, cap: int) -> int:
    return min(cap, max(quantum, int(math.ceil(max(value, 1.0) / quantum) * quantum)))


def _serving_replay(
    *,
    traces: Sequence[BayesianTraceV1],
    durations: Mapping[tuple[str, float, int], float],
    point_rows: Sequence[dict[str, Any]],
    posterior: Mapping[str, Sequence[dict[str, Any]]],
    cone: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    frozen = _read_json(Path(config["benchmark"]["serving_replay_config"]))["serving_replay"]
    step = int(frozen["prediction_step"])
    exact_traces = [trace for trace in traces if trace.terminal_observed]
    exact_identities = {(trace.prompt_id, trace.temperature, trace.seed) for trace in exact_traces}
    actual = {
        (trace.prompt_id, trace.temperature, trace.seed): trace.observed_tokens
        for trace in exact_traces
    }
    scalar = {
        (row["prompt_id"], row["temperature"], row["seed"]): float(row["predicted_remaining"])
        + step
        for row in posterior["bayesian_entropy_scalar_v1"]
        if int(row["step"]) == step
        and (row["prompt_id"], row["temperature"], row["seed"]) in exact_identities
    }
    alps = {
        (row["prompt_id"], row["temperature"], row["seed"]): float(row["predicted_remaining"])
        + step
        for row in posterior["alps_countdown"]
        if int(row["step"]) == step
        and (row["prompt_id"], row["temperature"], row["seed"]) in exact_identities
    }
    point = {
        (row["prompt_id"], row["temperature"], row["seed"]): row
        for row in point_rows
        if int(row["step"]) == step
        and (row["prompt_id"], row["temperature"], row["seed"]) in exact_identities
    }
    q975 = {
        (row["prompt_id"], row["temperature"], row["seed"]): float(row["posterior_q975_remaining"])
        + step
        for row in cone
        if int(row["step"]) == step
        and (row["prompt_id"], row["temperature"], row["seed"]) in exact_identities
    }
    exact_durations = {identity: durations[identity] for identity in exact_identities}
    identities = set(actual)
    if any(set(values) != identities for values in (exact_durations, scalar, alps, point, q975)):
        raise ValueError("final serving replay inputs do not cover identical requests")
    predictors: dict[str, Callable[[tuple[str, float, int]], float]] = {
        "oracle_observed_length": lambda identity: float(actual[identity]),
        "max_new_tokens_4096": lambda identity: 4096.0,
        "alps_countdown_mean": lambda identity: alps[identity],
        "plp_terminal_zero_v3": lambda identity: step
        + float(point[identity]["plp_terminal_zero_v3"]),
        "alps_plp_concat_v1": lambda identity: step + float(point[identity]["alps_plp_concat_v1"]),
        "bayesian_entropy_scalar_v1_mean": lambda identity: scalar[identity],
        "bayesian_entropy_scalar_v1_q975": lambda identity: q975[identity],
    }
    if list(predictors) != frozen["strategies"]:
        raise ValueError("final serving strategies differ from the frozen Stage-6 order")
    batch_size = int(frozen["batch_size"])
    boundaries = tuple(int(value) for value in frozen["length_bucket_boundaries"])
    quantum = int(frozen["kv_allocation_quantum_tokens"])
    cap = max(boundaries)
    bytes_per_token = int(frozen["kv_model_contract"]["bytes_per_output_token"])
    budget = int(float(frozen["incremental_kv_budget_gib"]) * 1024**3)
    metrics = {}
    for method, predictor in predictors.items():
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
                    "duration": exact_durations[identity],
                    "allocated": allocated,
                    "bucket": bucket,
                }
            )
        requests.sort(key=lambda row: (row["bucket"], *row["identity"]))
        clock = 0.0
        completion = []
        peak_bytes = 0
        budget_exceedance = 0
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            clock += max(float(row["duration"]) for row in batch)
            completion.extend([clock] * len(batch))
            batch_bytes = sum(int(row["allocated"]) for row in batch) * bytes_per_token
            peak_bytes = max(peak_bytes, batch_bytes)
            budget_exceedance += int(batch_bytes > budget)
        actual_values = np.asarray([row["actual"] for row in requests], dtype=np.float64)
        allocated_values = np.asarray([row["allocated"] for row in requests], dtype=np.float64)
        over = np.maximum(allocated_values - actual_values, 0.0)
        deficit = np.maximum(actual_values - allocated_values, 0.0)
        completion_values = np.asarray(completion, dtype=np.float64)
        metrics[method] = {
            "request_count": len(requests),
            "batch_count": math.ceil(len(requests) / batch_size),
            "mean_completion_time_ms": float(completion_values.mean()),
            "p95_completion_time_ms": float(np.quantile(completion_values, 0.95)),
            "makespan_ms": clock,
            "throughput_tokens_per_second": float(actual_values.sum() / (clock / 1000.0)),
            "kv_overreservation_tokens": int(over.sum()),
            "kv_overreservation_bytes": int(over.sum()) * bytes_per_token,
            "underallocation_rate": float((deficit > 0).mean()),
            "total_underallocated_tokens": int(deficit.sum()),
            "peak_batch_incremental_kv_bytes": peak_bytes,
            "incremental_kv_budget_exceedance_batch_count": budget_exceedance,
        }
    return {
        "scope": frozen["scope"],
        "prediction_step": step,
        "exact_sequence_count": len(exact_identities),
        "right_censored_sequence_count_excluded_not_imputed": len(traces) - len(exact_identities),
        "model_contract": frozen["kv_model_contract"],
        "metrics": metrics,
    }


def run_final_benchmark(
    config_path: str | Path,
    *,
    model_root: Path,
    output_dir: Path,
    device: str,
    verify_trace_hashes: bool,
) -> dict[str, Any]:
    config, lock, traces, durations = load_final_holdout_traces(
        config_path, verify_trace_hashes=verify_trace_hashes
    )
    models = load_final_models(config, output_dir=model_root, device=device)
    point_rows = _point_rows(traces, models, config)
    posterior: dict[str, list[dict[str, Any]]] = {
        "alps_countdown": [],
        **{method: [] for method in BAYESIAN_METHOD_IDS},
    }
    cone = []
    for trace in traces:
        sequence = stage5_bayesian_sequence(
            trace,
            prior_mu=models.alps_prior.predict_mu(trace.prior_feature),
            prior_log_variance=models.alps_prior.residual_variance,
        )
        posterior["alps_countdown"].extend(
            compact_posterior_rows(
                "alps_countdown", run_prior_countdown_sequence(sequence), outer_fold=-1
            )
        )
        for method in BAYESIAN_METHOD_IDS:
            observations = run_bayesian_sequence(
                sequence, models.bayesian_scorers[method], device=models.device
            )
            posterior[method].extend(compact_posterior_rows(method, observations, outer_fold=-1))
            if method == config["benchmark"]["primary_method"]:
                cone.extend(_cone_rows(observations))
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "point_predictions.csv", point_rows)
    _write_csv(output_dir / "uncertainty_cone.csv", cone)
    for method, rows in posterior.items():
        _write_jsonl(output_dir / f"{method}_predictions.jsonl", rows)
    metrics = {method: _posterior_metrics(rows) for method, rows in posterior.items()}
    metrics.update(
        {method: _sequence_point_metrics(point_rows, method) for method in POINT_METHODS}
    )
    breakdowns = {method: _posterior_breakdowns(rows) for method, rows in posterior.items()}
    breakdowns.update({method: _point_breakdowns(point_rows, method) for method in POINT_METHODS})
    primary = posterior[config["benchmark"]["primary_method"]]
    exact_primary = [row for row in primary if row["true_remaining"] is not None]
    convergence = {
        "overall": _convergence_with_censoring(exact_primary, primary, group=lambda row: "all"),
        "by_task": _convergence_with_censoring(
            exact_primary, primary, group=lambda row: str(row["task"])
        ),
        "by_temperature": _convergence_with_censoring(
            exact_primary,
            primary,
            group=lambda row: f"{float(row['temperature']):.3f}",
        ),
        "never_reached_including_right_censored": "failure",
    }
    primary_by_key = {
        (row["prompt_id"], row["temperature"], row["seed"], row["step"]): row for row in primary
    }
    point_primary_rows = []
    for row in point_rows:
        key = (row["prompt_id"], row["temperature"], row["seed"], row["step"])
        combined = dict(row)
        combined["primary_absolute_error"] = (
            None
            if primary_by_key[key]["error_tokens"] is None
            else abs(float(primary_by_key[key]["error_tokens"]))
        )
        point_primary_rows.append(combined)
    statistical = {
        "scalar_minus_alps_posterior_nll": _family_bootstrap(
            primary,
            posterior["alps_countdown"],
            left_value=lambda row: float(row["posterior_nll"]),
            right_value=lambda row: float(row["posterior_nll"]),
            config=config,
        ),
        "scalar_minus_hidden_posterior_nll": _family_bootstrap(
            primary,
            posterior["bayesian_entropy_hidden_delta_v1"],
            left_value=lambda row: float(row["posterior_nll"]),
            right_value=lambda row: float(row["posterior_nll"]),
            config=config,
        ),
    }
    for method in POINT_METHODS:
        statistical[f"scalar_minus_{method}_absolute_error"] = _family_bootstrap(
            point_primary_rows,
            point_primary_rows,
            left_value=lambda row: row["primary_absolute_error"],
            right_value=lambda row, name=method: (
                None
                if row["true_remaining"] is None
                else abs(float(row[name]) - int(row["true_remaining"]))
            ),
            config=config,
        )
    report = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "status": "pass",
        "stage8a_config_sha256": config["_config_sha256"],
        "stage8b_lock_sha256": file_sha256(Path(config["holdout_gate"]["benchmark_lock"])),
        "checkpoint_registry_sha256": str(lock["checkpoint_registry_sha256"]),
        "final_holdout_manifest_sha256": str(lock["final_holdout_manifest_sha256"]),
        "collection": _read_json(
            Path(config["outputs"]["collection_run_root"]) / "collection_report.json"
        ),
        "comparison_list": config["comparison_list"],
        "primary_method": config["benchmark"]["primary_method"],
        "metrics": metrics,
        "breakdowns": breakdowns,
        "uncertainty_cone": {
            "path": "uncertainty_cone.csv",
            "observation_count": len(cone),
            "q975_overflow_boundary_count": sum(
                bool(row["q975_used_overflow_boundary"]) for row in cone
            ),
        },
        "convergence": convergence,
        "serving_replay": _serving_replay(
            traces=traces,
            durations=durations,
            point_rows=point_rows,
            posterior=posterior,
            cone=cone,
            config=config,
        ),
        "statistical_comparisons": statistical,
        "model_selection_performed": False,
        "threshold_tuning_performed": False,
        "final_holdout_selects_nothing": True,
    }
    _write_json(output_dir / "final_benchmark_report.json", report)
    _write_json(
        output_dir / "file_manifest.json",
        {
            "files": {
                path.name: file_sha256(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file()
            }
        },
    )
    return report

"""Compact streaming metrics and selection for Stage-5 OOF."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from llm_length_prediction.evaluation.sequential import (
    PosteriorObservation,
    discrete_crps,
    posterior_nll,
    right_censored_posterior_nll,
)
from llm_length_prediction.models.bayesian_filter import BayesianFilterState


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def run_prior_countdown_sequence(sequence: Any) -> list[PosteriorObservation]:
    """Evaluate the ALPS prior under countdown/survival with no decode likelihood."""

    sequence.validate()
    state = BayesianFilterState(
        step=0,
        log_probabilities=np.asarray(sequence.initial_log_probabilities, dtype=np.float64),
        has_overflow=sequence.has_overflow,
    )
    observations = []
    for evidence_step in sequence.steps:
        state = state.predict_to(evidence_step.step)
        observations.append(
            PosteriorObservation(
                prompt_id=sequence.prompt_id,
                prompt_family_id=sequence.prompt_family_id,
                task=sequence.task,
                intended_length=sequence.intended_length,
                temperature=sequence.temperature,
                seed=sequence.seed,
                step=evidence_step.step,
                probabilities=state.probabilities,
                summary=state.summary(),
                true_remaining=evidence_step.true_remaining,
                censored_after_remaining=evidence_step.censored_after_remaining,
                terminal_observed=evidence_step.terminal_observed,
                update_wall_time_ms=0.0,
                predictor_state_bytes=state.log_probabilities.nbytes,
                has_overflow=sequence.has_overflow,
            )
        )
    return observations


def compact_posterior_rows(
    method_id: str,
    observations: Sequence[PosteriorObservation],
    *,
    outer_fold: int,
) -> list[dict[str, Any]]:
    if not observations:
        raise ValueError("posterior observations are required")
    rows = []
    for observation in observations:
        if observation.true_remaining is None:
            nll = right_censored_posterior_nll(
                observation.probabilities,
                int(observation.censored_after_remaining),
                has_overflow=observation.has_overflow,
            )
            crps = math.nan
            error = math.nan
        else:
            nll = posterior_nll(
                observation.probabilities,
                observation.true_remaining,
                has_overflow=observation.has_overflow,
            )
            crps = discrete_crps(
                observation.probabilities,
                observation.true_remaining,
                has_overflow=observation.has_overflow,
            )
            error = observation.summary.mean_remaining_lower_bound - observation.true_remaining
        intervals = {
            interval.level: interval for interval in observation.summary.credible_intervals
        }
        row: dict[str, Any] = {
            "method_id": method_id,
            "prompt_id": observation.prompt_id,
            "prompt_family_id": observation.prompt_family_id,
            "task": observation.task,
            "intended_length": observation.intended_length,
            "temperature": observation.temperature,
            "seed": observation.seed,
            "outer_fold": outer_fold,
            "step": observation.step,
            "true_remaining": observation.true_remaining,
            "censored_after_remaining": observation.censored_after_remaining,
            "terminal_observed": observation.terminal_observed,
            "posterior_nll": nll,
            "crps": _finite_or_none(crps),
            "predicted_remaining": observation.summary.mean_remaining_lower_bound,
            "error_tokens": _finite_or_none(error),
            "posterior_variance_lower_bound": observation.summary.variance_lower_bound,
            "posterior_entropy": observation.summary.entropy,
            "overflow_probability": observation.summary.overflow_probability,
            "update_wall_time_ms": observation.update_wall_time_ms,
            "predictor_state_bytes": observation.predictor_state_bytes,
        }
        for level in (0.5, 0.9, 0.95):
            interval = intervals[level]
            label = int(level * 100)
            covered = (
                math.nan
                if observation.true_remaining is None
                else float(interval.lower <= observation.true_remaining <= interval.upper)
            )
            row[f"interval_{label}_coverage"] = _finite_or_none(covered)
            row[f"interval_{label}_width"] = _finite_or_none(
                interval.upper - interval.lower
            )
        rows.append(row)
    return rows


def _sequence_means(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["prompt_id"], float(row["temperature"]), int(row["seed"]))].append(row)
    metrics = (
        "posterior_nll",
        "crps",
        "error_tokens",
        "posterior_variance_lower_bound",
        "posterior_entropy",
        "overflow_probability",
        "update_wall_time_ms",
        "interval_50_coverage",
        "interval_50_width",
        "interval_90_coverage",
        "interval_90_width",
        "interval_95_coverage",
        "interval_95_width",
    )
    output = []
    for sequence_rows in groups.values():
        item = {
            name: sequence_rows[0][name]
            for name in (
                "prompt_id",
                "prompt_family_id",
                "task",
                "intended_length",
                "temperature",
                "seed",
                "outer_fold",
            )
        }
        for metric in metrics:
            values = np.asarray(
                [
                    math.nan if row[metric] is None else float(row[metric])
                    for row in sequence_rows
                ],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            item[metric] = float(finite.mean()) if len(finite) else None
        errors = np.asarray(
            [
                math.nan if row["error_tokens"] is None else float(row["error_tokens"])
                for row in sequence_rows
            ],
            dtype=np.float64,
        )
        finite_errors = errors[np.isfinite(errors)]
        item["mae_tokens"] = (
            float(np.abs(finite_errors).mean()) if len(finite_errors) else None
        )
        item["squared_error_tokens"] = (
            float(np.square(finite_errors).mean()) if len(finite_errors) else None
        )
        item["bias_tokens"] = (
            float(finite_errors.mean()) if len(finite_errors) else None
        )
        item["peak_predictor_state_bytes"] = max(
            int(row["predictor_state_bytes"]) for row in sequence_rows
        )
        output.append(item)
    return output


def posterior_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("posterior metric rows are required")
    sequences = _sequence_means(rows)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sequences:
        by_family[str(row["prompt_family_id"])].append(row)

    def mean(
        name: str, values: Sequence[dict[str, Any]] = sequences
    ) -> float | None:
        vector = np.asarray(
            [math.nan if row[name] is None else float(row[name]) for row in values],
            dtype=np.float64,
        )
        finite = vector[np.isfinite(vector)]
        return float(finite.mean()) if len(finite) else None

    family_nll = {}
    for family, values in by_family.items():
        value = mean("posterior_nll", values)
        if value is None:
            raise ValueError(f"family {family} has no finite posterior NLL")
        family_nll[family] = value
    mean_squared_error = mean("squared_error_tokens")
    return {
        "observation_count": len(rows),
        "sequence_count": len(sequences),
        "family_count": len(by_family),
        "family_macro_sequence_balanced_posterior_nll": float(
            np.mean(list(family_nll.values()))
        ),
        "sequence_balanced_posterior_nll": mean("posterior_nll"),
        "sequence_balanced_crps": mean("crps"),
        "sequence_balanced_mae_tokens": mean("mae_tokens"),
        "sequence_balanced_rmse_tokens": (
            math.sqrt(mean_squared_error) if mean_squared_error is not None else None
        ),
        "sequence_balanced_bias_tokens": mean("bias_tokens"),
        "sequence_balanced_posterior_variance_lower_bound": mean(
            "posterior_variance_lower_bound"
        ),
        "sequence_balanced_posterior_entropy": mean("posterior_entropy"),
        "mean_overflow_probability": mean("overflow_probability"),
        "mean_update_wall_time_ms": mean("update_wall_time_ms"),
        "peak_predictor_state_bytes": max(
            int(row["peak_predictor_state_bytes"]) for row in sequences
        ),
        **{
            f"sequence_balanced_interval_{level}_{suffix}": mean(
                f"interval_{level}_{suffix}"
            )
            for level in (50, 90, 95)
            for suffix in ("coverage", "width")
        },
        "family_sequence_nll": family_nll,
    }


def posterior_breakdowns(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groupers: dict[str, Callable[[dict[str, Any]], str]] = {
        "by_task": lambda row: str(row["task"]),
        "by_intended_length": lambda row: str(row["intended_length"]),
        "by_task_and_intended_length": (
            lambda row: f"{row['task']}:{row['intended_length']}"
        ),
        "by_temperature": lambda row: f"{float(row['temperature']):.3f}",
        "by_seed": lambda row: str(row["seed"]),
        "by_outer_fold": lambda row: str(row["outer_fold"]),
        "terminal_vs_nonterminal": (
            lambda row: "terminal" if row["terminal_observed"] else "nonterminal"
        ),
    }
    output = {}
    for name, grouper in groupers.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[grouper(row)].append(row)
        output[name] = [
            {"group": label, **posterior_metrics(values)}
            for label, values in sorted(groups.items())
        ]
    progress_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["true_remaining"] is None:
            label = "censored"
        else:
            total = row["step"] + row["true_remaining"]
            fraction = row["step"] / max(total, 1)
            if fraction < 0.1:
                label = "0-10%"
            elif fraction < 0.25:
                label = "10-25%"
            elif fraction < 0.5:
                label = "25-50%"
            elif fraction < 0.75:
                label = "50-75%"
            else:
                label = "75-100%"
        progress_groups[label].append(row)
    output["by_decode_progress"] = [
        {"group": label, **posterior_metrics(values)}
        for label, values in progress_groups.items()
    ]
    return output


def paired_family_nll_difference(
    hidden_rows: Sequence[dict[str, Any]],
    scalar_rows: Sequence[dict[str, Any]],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    hidden = posterior_metrics(hidden_rows)["family_sequence_nll"]
    scalar = posterior_metrics(scalar_rows)["family_sequence_nll"]
    if hidden.keys() != scalar.keys():
        raise ValueError("Bayesian candidates do not cover identical families")
    families = sorted(hidden)
    differences = np.asarray([hidden[key] - scalar[key] for key in families])
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(families), size=(replicates, len(families)))
    statistics = differences[draws].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "estimate": float(differences.mean()),
        "lower": float(np.quantile(statistics, alpha / 2.0)),
        "upper": float(np.quantile(statistics, 1.0 - alpha / 2.0)),
        "confidence_level": confidence,
        "replicates": replicates,
        "family_count": len(families),
        "unit": "prompt_family_id",
    }

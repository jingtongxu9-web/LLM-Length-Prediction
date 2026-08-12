"""Frozen Stage-7 OOF-only error feedback over selected Stage-5 predictions."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.bayesian_stage5 import load_stage5_catalog
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage6_analysis import (
    Stage6Sources,
    load_posterior_rows,
    load_stage6_sources,
    sequence_id,
    strict_json_write,
)

STAGE7_ID = "bayesian-sequential-v1-stage7-oof-error-feedback"


@dataclass(frozen=True)
class Stage7Sources:
    config: dict[str, Any]
    stage6: Stage6Sources


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage7_config(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path))
    if payload.get("schema_version") != 1 or payload.get("stage7_id") != STAGE7_ID:
        raise ValueError("unsupported Stage-7 configuration")
    if payload.get("claim_scope") != "train_family_grouped_oof_failure_audit_not_final_holdout":
        raise ValueError("Stage-7 claim scope changed")
    policy = payload["data_policy"]
    if policy.get("split") != "train_family_grouped_oof_only":
        raise ValueError("Stage-7 must use Train-family OOF only")
    if any(
        policy.get(key) is not False
        for key in ("model_refit", "method_reselection", "threshold_tuning")
    ):
        raise ValueError("Stage-7 cannot refit, reselect, or tune thresholds")
    if policy.get("changes_require_new_method_id") is not True:
        raise ValueError("Stage-7 fixes must require a new method ID")
    if "forbidden" not in str(policy.get("new_final_holdout_access", "")):
        raise ValueError("Stage-7 cannot access a final holdout")
    contract = payload["scientific_contract"]
    if file_sha256(contract["path"]) != contract["sha256"]:
        raise ValueError("scientific contract changed after Stage-7 freeze")
    required = set(_read_json(Path(contract["path"]))["error_feedback"]["labels"])
    if set(payload["labels"]) != required:
        raise ValueError("Stage-7 labels differ from the scientific contract")
    stage6 = payload["stage6_source"]
    if file_sha256(stage6["config"]) != stage6["config_sha256"]:
        raise ValueError("Stage-6 source configuration changed after Stage-7 freeze")
    cohorts = payload["cohorts"]
    if cohorts["absolute_error_threshold_tokens"] != 100.0 or cohorts["worst_fraction"] != 0.05:
        raise ValueError("Stage-7 frozen error cohorts changed")
    schema = payload["report_schema"]
    if file_sha256(schema["path"]) != schema["sha256"]:
        raise ValueError("Stage-7 report schema changed after the analysis freeze")
    return payload


def load_stage7_sources(
    config_path: str | Path,
    *,
    stage4_root: str | Path,
    stage5_root: str | Path,
    verify_stage5_files: bool = False,
) -> Stage7Sources:
    config = load_stage7_config(config_path)
    stage6 = load_stage6_sources(
        config["stage6_source"]["config"],
        stage4_root=stage4_root,
        stage5_root=stage5_root,
        verify_stage5_files=verify_stage5_files,
    )
    frozen = config["stage5"]
    selected = stage6.selection["selected_method"]
    if (
        selected != frozen["selected_method"]
        or stage6.stage5_report["dataset_digest"] != frozen["dataset_digest"]
    ):
        raise ValueError("Stage-5 selection or dataset changed before Stage-7")
    rows = load_posterior_rows(stage6, selected)
    identities = {sequence_id(row) for row in rows}
    if (
        len(rows) != frozen["required_observation_count"]
        or len(identities) != frozen["required_sequence_count"]
    ):
        raise ValueError("Stage-7 source coverage differs from the freeze")
    return Stage7Sources(config=config, stage6=stage6)


def _direction_changes(values: Sequence[float], *, epsilon: float = 1e-9) -> int:
    signs: list[int] = []
    for delta in np.diff(np.asarray(values, dtype=np.float64)):
        sign = 1 if delta > epsilon else -1 if delta < -epsilon else 0
        if sign and (not signs or sign != signs[-1]):
            signs.append(sign)
    return max(0, len(signs) - 1)


def _progress_bin_medians(
    rows: Sequence[dict[str, Any]], edges: Sequence[float], field: str
) -> list[float]:
    bins: list[list[float]] = [[] for _ in range(len(edges) - 1)]
    for row in rows:
        total = int(row["step"]) + int(row["true_remaining"])
        progress = int(row["step"]) / max(total, 1)
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            if float(lower) <= progress < float(upper):
                bins[index].append(float(row[field]))
                break
    return [float(np.median(values)) for values in bins if values]


def _repetition_metrics(tokens: np.ndarray, *, ngram_size: int) -> tuple[float, int]:
    values = [int(value) for value in tokens]
    if len(values) < ngram_size:
        repeated_fraction = 0.0
    else:
        counts = Counter(
            tuple(values[index : index + ngram_size])
            for index in range(len(values) - ngram_size + 1)
        )
        repeated_fraction = sum(max(0, count - 1) for count in counts.values()) / max(
            sum(counts.values()), 1
        )
    longest = 0
    current = 0
    previous = None
    for value in values:
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
        previous = value
    return float(repeated_fraction), longest


def audit_sequences(
    sources: Stage7Sources,
    *,
    verify_trace_hashes: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = sources.config
    selected = config["stage5"]["selected_method"]
    predictions = load_posterior_rows(sources.stage6, selected)
    by_sequence: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_sequence[sequence_id(row)].append(row)
    for rows in by_sequence.values():
        rows.sort(key=lambda item: int(item["step"]))

    stage5_config = sources.stage6.config["stage5"]["config"]
    catalog = load_stage5_catalog(
        stage5_config,
        dataset_root=sources.stage6.stage4_root,
        verify_trace_hashes=verify_trace_hashes,
    )
    references = {reference.identity: reference for reference in catalog.references}
    if set(references) != set(by_sequence):
        raise ValueError("Stage-4 traces and Stage-5 OOF predictions do not align")

    peer_lengths: dict[tuple[str, float], list[int]] = defaultdict(list)
    for reference in references.values():
        peer_lengths[(reference.prompt_id, reference.temperature)].append(reference.observed_tokens)
    peer_stats = {
        key: (
            float(np.median(values)),
            max(values) - min(values),
            float(np.std(values) / max(np.mean(values), 1.0)),
        )
        for key, values in peer_lengths.items()
    }
    sequence_mae = {
        key: float(np.mean([abs(float(row["error_tokens"])) for row in rows]))
        for key, rows in by_sequence.items()
    }
    worst_threshold = float(
        np.quantile(
            np.asarray(list(sequence_mae.values())),
            1.0 - float(config["cohorts"]["worst_fraction"]),
            method=config["cohorts"]["worst_quantile_method"],
        )
    )
    label_config = config["automatic_labels"]
    output: list[dict[str, Any]] = []

    for identity, rows in sorted(by_sequence.items()):
        reference = references[identity]
        trace = catalog.load_trace(reference)
        entropy = np.asarray(trace.token_entropies, dtype=np.float64)
        late_start = int(
            math.floor(
                len(entropy) * (1.0 - float(label_config["entropy_rebound"]["late_fraction"]))
            )
        )
        late_mean = float(entropy[late_start:].mean())
        prefix_partition_count = int(label_config["entropy_rebound"]["prefix_partition_count"])
        prefix_min = float(
            np.min(
                [
                    chunk.mean()
                    for chunk in np.array_split(
                        entropy[: max(late_start, 1)],
                        min(prefix_partition_count, max(late_start, 1)),
                    )
                    if len(chunk)
                ]
            )
        )
        saved_entropy = [float(entropy[min(int(row["step"]), len(entropy)) - 1]) for row in rows]
        repeated_fraction, longest_run = _repetition_metrics(
            trace.generated_token_ids, ngram_size=int(label_config["repetition"]["ngram_size"])
        )
        peer_median, peer_range, peer_cv = peer_stats[(reference.prompt_id, reference.temperature)]
        variance_bins = _progress_bin_medians(
            rows,
            label_config["posterior_variance_increase"]["progress_bin_edges"],
            "posterior_variance_lower_bound",
        )
        variance_increase = any(
            current
            > previous
            * (
                1.0
                + float(
                    label_config["posterior_variance_increase"][
                        "minimum_adjacent_relative_increase"
                    ]
                )
            )
            for previous, current in zip(variance_bins, variance_bins[1:], strict=False)
        )
        true_total = reference.observed_tokens
        early_uncovered = sum(
            float(row["interval_95_coverage"]) == 0.0
            and float(row["interval_95_width"]) / max(true_total, 1)
            <= float(
                label_config["posterior_premature_collapse"][
                    "maximum_interval95_width_to_true_total"
                ]
            )
            for row in rows
            if int(row["step"]) / max(true_total, 1)
            <= float(label_config["posterior_premature_collapse"]["early_progress_maximum"])
        )
        total_predictions = [int(row["step"]) + float(row["predicted_remaining"]) for row in rows]
        automatic = {
            "entropy_rebound": late_mean - prefix_min
            >= float(label_config["entropy_rebound"]["minimum_increase_nats"]),
            "entropy_oscillation": _direction_changes(saved_entropy)
            >= int(label_config["entropy_oscillation"]["minimum_direction_changes"])
            and max(saved_entropy) - min(saved_entropy)
            >= float(label_config["entropy_oscillation"]["minimum_range_nats"]),
            "sampling_divergence": peer_range
            >= int(label_config["sampling_divergence"]["minimum_output_length_range_tokens"])
            and peer_cv
            >= float(label_config["sampling_divergence"]["minimum_coefficient_of_variation"]),
            "repetition": repeated_fraction
            >= float(label_config["repetition"]["minimum_repeated_ngram_fraction"])
            or longest_run
            >= int(label_config["repetition"]["minimum_consecutive_identical_tokens"]),
            "early_stop": true_total
            <= peer_median * float(label_config["early_stop"]["maximum_peer_median_ratio"])
            and peer_median - true_total
            >= int(label_config["early_stop"]["minimum_peer_median_shortfall_tokens"]),
            "posterior_variance_increase": variance_increase,
            "posterior_premature_collapse": early_uncovered
            >= int(label_config["posterior_premature_collapse"]["minimum_uncovered_saved_points"]),
            "posterior_oscillation": _direction_changes(total_predictions)
            >= int(label_config["posterior_oscillation"]["minimum_direction_changes"])
            and max(total_predictions) - min(total_predictions)
            >= float(
                label_config["posterior_oscillation"]["minimum_total_prediction_range_tokens"]
            ),
        }
        errors = [float(row["error_tokens"]) for row in rows]
        max_abs = max(abs(value) for value in errors)
        output.append(
            {
                "prompt_id": reference.prompt_id,
                "prompt_family_id": reference.prompt_family_id,
                "task": reference.task,
                "intended_length": reference.intended_length,
                "temperature": reference.temperature,
                "seed": reference.seed,
                "outer_fold": catalog.family_folds[reference.prompt_family_id],
                "observed_tokens": true_total,
                "saved_point_count": len(rows),
                "sequence_mae_tokens": sequence_mae[identity],
                "sequence_bias_tokens": float(np.mean(errors)),
                "maximum_absolute_error_tokens": max_abs,
                "maximum_underestimation_tokens": max(-min(errors), 0.0),
                "maximum_overestimation_tokens": max(max(errors), 0.0),
                "absolute_error_cohort": max_abs
                > float(config["cohorts"]["absolute_error_threshold_tokens"]),
                "worst_fraction_cohort": sequence_mae[identity] >= worst_threshold,
                "automatic_labels": automatic,
                "manual_labels": {"open_ended_prompt": "unresolved", "hallucination": "unresolved"},
                "diagnostics": {
                    "entropy_late_minus_prior_min_nats": late_mean - prefix_min,
                    "entropy_direction_changes": _direction_changes(saved_entropy),
                    "seed_output_length_range_tokens": peer_range,
                    "seed_output_length_cv": peer_cv,
                    "repeated_ngram_fraction": repeated_fraction,
                    "longest_identical_token_run": longest_run,
                    "peer_median_output_tokens": peer_median,
                    "variance_progress_bin_medians": variance_bins,
                    "early_uncovered_narrow_95_points": early_uncovered,
                    "posterior_total_direction_changes": _direction_changes(total_predictions),
                    "posterior_total_range_tokens": max(total_predictions) - min(total_predictions),
                },
            }
        )

    review = [row for row in output if row["absolute_error_cohort"] or row["worst_fraction_cohort"]]
    labels = list(label_config)
    report = {
        "sequence_count": len(output),
        "observation_count": len(predictions),
        "worst_fraction_sequence_mae_threshold_tokens": worst_threshold,
        "absolute_error_cohort_sequence_count": sum(row["absolute_error_cohort"] for row in output),
        "worst_fraction_cohort_sequence_count": sum(row["worst_fraction_cohort"] for row in output),
        "review_queue_sequence_count": len(review),
        "automatic_label_counts_all_sequences": {
            label: sum(row["automatic_labels"][label] for row in output) for label in labels
        },
        "automatic_label_counts_review_queue": {
            label: sum(row["automatic_labels"][label] for row in review) for label in labels
        },
        "manual_labels": config["manual_labels"],
        "trace_hashes_verified": verify_trace_hashes,
    }
    return output, report


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build_review_queue(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": row["prompt_id"],
            "prompt_family_id": row["prompt_family_id"],
            "temperature": row["temperature"],
            "seed": row["seed"],
            "sequence_mae_tokens": row["sequence_mae_tokens"],
            "sequence_bias_tokens": row["sequence_bias_tokens"],
            "maximum_absolute_error_tokens": row["maximum_absolute_error_tokens"],
            "maximum_underestimation_tokens": row["maximum_underestimation_tokens"],
            "maximum_overestimation_tokens": row["maximum_overestimation_tokens"],
            "cohorts": {
                "absolute_error": row["absolute_error_cohort"],
                "worst_fraction": row["worst_fraction_cohort"],
            },
            "automatic_labels": row["automatic_labels"],
            "manual_review_required": ["open_ended_prompt", "hallucination"],
        }
        for row in rows
        if row["absolute_error_cohort"] or row["worst_fraction_cohort"]
    ]


def _cohort_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    automatic = next(iter(rows))["automatic_labels"] if rows else {}
    return {
        "sequence_count": len(rows),
        "mean_sequence_mae_tokens": (
            float(np.mean([row["sequence_mae_tokens"] for row in rows])) if rows else None
        ),
        "mean_sequence_bias_tokens": (
            float(np.mean([row["sequence_bias_tokens"] for row in rows])) if rows else None
        ),
        "negative_bias_sequence_rate": (
            sum(float(row["sequence_bias_tokens"]) < 0.0 for row in rows) / len(rows)
            if rows
            else None
        ),
        "mean_maximum_underestimation_tokens": (
            float(np.mean([row["maximum_underestimation_tokens"] for row in rows]))
            if rows
            else None
        ),
        "mean_maximum_overestimation_tokens": (
            float(np.mean([row["maximum_overestimation_tokens"] for row in rows])) if rows else None
        ),
        "mean_maximum_absolute_error_tokens": (
            float(np.mean([row["maximum_absolute_error_tokens"] for row in rows])) if rows else None
        ),
        "automatic_label_counts": {
            label: sum(bool(row["automatic_labels"][label]) for row in rows) for label in automatic
        },
        "automatic_label_rates": {
            label: (
                sum(bool(row["automatic_labels"][label]) for row in rows) / len(rows)
                if rows
                else None
            )
            for label in automatic
        },
    }


def summarize_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    review = [row for row in rows if row["absolute_error_cohort"] or row["worst_fraction_cohort"]]
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "task": defaultdict(list),
        "intended_length": defaultdict(list),
        "temperature": defaultdict(list),
        "outer_fold": defaultdict(list),
    }
    for row in review:
        groups["task"][str(row["task"])].append(row)
        groups["intended_length"][str(row["intended_length"])].append(row)
        groups["temperature"][f"{float(row['temperature']):.1f}"].append(row)
        groups["outer_fold"][str(int(row["outer_fold"]))].append(row)
    label_sets = Counter(
        "+".join(label for label, value in row["automatic_labels"].items() if value)
        or "no_automatic_label"
        for row in review
    )
    return {
        "all_sequences": _cohort_summary(rows),
        "absolute_error_cohort": _cohort_summary(
            [row for row in rows if row["absolute_error_cohort"]]
        ),
        "worst_fraction_cohort": _cohort_summary(
            [row for row in rows if row["worst_fraction_cohort"]]
        ),
        "union_review_queue": _cohort_summary(review),
        "review_queue_breakdowns": {
            dimension: {
                label: _cohort_summary(values) for label, values in sorted(dimension_rows.items())
            }
            for dimension, dimension_rows in groups.items()
        },
        "most_common_automatic_label_sets": [
            {"labels": labels.split("+"), "sequence_count": count}
            for labels, count in label_sets.most_common(10)
        ],
        "semantic_label_status": {
            "open_ended_prompt": "unresolved_manual_review_required",
            "hallucination": "unresolved_manual_review_required",
        },
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    strict_json_write(path, payload)

"""Evaluate frozen PLP v2 and terminal-zero v3 once on the owned final Test."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.evaluation.hybrid import (
    family_bootstrap_interval,
    family_macro_metrics,
    family_metric_rows,
    paired_family_mae_difference,
)
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.hybrid_experiment import (
    enforce_censoring_policy,
    hybrid_samples,
    load_complete_hybrid_split,
    load_hybrid_config,
    load_hybrid_experiment,
    partition_censored,
    validate_hybrid_config,
)
from llm_length_prediction.models.hybrid import HybridSample
from llm_length_prediction.models.plp_v3 import (
    PLP_V3_METHOD_IDS,
    load_plp_v3,
    predict_plp_v3,
)

DEFAULT_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/plp_terminal_v3_protocol.json")


def _subset(
    samples: Sequence[HybridSample],
    predictions: np.ndarray,
    predicate: Callable[[HybridSample], bool],
) -> tuple[list[HybridSample], np.ndarray]:
    indices = [index for index, sample in enumerate(samples) if predicate(sample)]
    return [samples[index] for index in indices], predictions[indices]


def _group_report(
    samples: Sequence[HybridSample],
    predictions: np.ndarray,
    groups: dict[str, Callable[[HybridSample], bool]],
) -> list[dict[str, Any]]:
    rows = []
    for name, predicate in groups.items():
        selected, values = _subset(samples, predictions, predicate)
        if selected:
            trace_counts = Counter(sample.trace_key for sample in selected)
            balanced = [
                replace(sample, sequence_weight=1.0 / trace_counts[sample.trace_key])
                for sample in selected
            ]
            rows.append({"group": name, **family_macro_metrics(balanced, values)})
    return rows


def _breakdowns(
    samples: Sequence[HybridSample], predictions: np.ndarray
) -> dict[str, list[dict[str, Any]]]:
    tasks = sorted({sample.task for sample in samples})
    lengths = ("short", "medium", "long")
    seeds = sorted({sample.seed for sample in samples})
    progress_bins = (
        ("0-10%", 0.0, 0.10),
        ("10-25%", 0.10, 0.25),
        ("25-50%", 0.25, 0.50),
        ("50-75%", 0.50, 0.75),
        ("75-100%", 0.75, 1.0),
    )

    def progress(sample: HybridSample) -> float:
        return sample.step / sample.output_tokens

    progress_groups = {
        name: (
            lambda sample, lower=lower, upper=upper: progress(sample) >= lower
            and (progress(sample) < upper or (upper == 1.0 and progress(sample) <= upper))
        )
        for name, lower, upper in progress_bins
    }
    return {
        "by_task": _group_report(
            samples,
            predictions,
            {task: lambda sample, task=task: sample.task == task for task in tasks},
        ),
        "by_intended_length": _group_report(
            samples,
            predictions,
            {
                length: lambda sample, length=length: sample.intended_length == length
                for length in lengths
            },
        ),
        "by_task_x_intended_length": _group_report(
            samples,
            predictions,
            {
                f"{task}/{length}": (
                    lambda sample, task=task, length=length: sample.task == task
                    and sample.intended_length == length
                )
                for task in tasks
                for length in lengths
            },
        ),
        "by_seed": _group_report(
            samples,
            predictions,
            {str(seed): lambda sample, seed=seed: sample.seed == seed for seed in seeds},
        ),
        "by_decode_progress": _group_report(samples, predictions, progress_groups),
        "terminal_vs_nonterminal": _group_report(
            samples,
            predictions,
            {
                "terminal_remaining_zero": lambda sample: sample.remaining_tokens == 0,
                "nonterminal_positive_remaining": lambda sample: sample.remaining_tokens > 0,
            },
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--trace-root", type=Path)
    args = parser.parse_args()
    config = load_hybrid_config(args.config)
    experiment, records = load_hybrid_experiment(config)
    validate_hybrid_config(config, experiment)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    run_root = Path(protocol["outputs"]["run_root"])
    marker_path = run_root / protocol["outputs"]["final_test"] / "OPENED.json"
    registry_path = run_root / protocol["outputs"]["models"] / "model_registry.json"
    if not marker_path.is_file() or not registry_path.is_file():
        raise SystemExit("PLP final-Test gate or frozen model registry is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if marker.get("test_opened") is not True:
        raise SystemExit("PLP final-Test gate is not open")
    if marker.get("model_registry_sha256") != file_sha256(registry_path):
        raise SystemExit("PLP model registry changed after Test was opened")
    for method_id in PLP_V3_METHOD_IDS:
        item = registry["methods"][method_id]
        if file_sha256(registry_path.parent / item["file"]) != item["sha256"]:
            raise SystemExit(f"frozen model changed after gate: {item['file']}")
    loaded = load_complete_hybrid_split(
        config, experiment, records, split="test", trace_root=args.trace_root
    )
    effective, censored = partition_censored(loaded)
    censoring = enforce_censoring_policy(
        loaded_count=len(loaded),
        censored_count=censored,
        warning_rate=float(config["censoring"]["warning_rate"]),
        abort_rate=float(config["censoring"]["abort_rate"]),
    )
    samples = hybrid_samples(effective)
    fitted = load_plp_v3(registry_path.parent)
    predictions = predict_plp_v3(
        fitted, samples, batch_size=int(config["training"]["batch_size"])
    )
    evaluation = protocol["evaluation"]
    method_reports = {}
    for index, method_id in enumerate(PLP_V3_METHOD_IDS):
        values = predictions[method_id]
        family_rows = family_metric_rows(samples, values)
        family_mae = {
            str(row["prompt_family_id"]): float(row["sequence_balanced_mae_tokens"])
            for row in family_rows
        }
        method_reports[method_id] = {
            "overall": family_macro_metrics(samples, values),
            "family_macro_mae_ci_95": family_bootstrap_interval(
                family_mae,
                replicates=int(evaluation["bootstrap_replicates"]),
                confidence=float(evaluation["bootstrap_confidence_level"]),
                seed=int(evaluation["bootstrap_seed"]) + index,
            ),
            "breakdowns": _breakdowns(samples, values),
        }
    paired = paired_family_mae_difference(
        samples,
        predictions["plp_terminal_zero_v3"],
        predictions["plp_v2_frozen"],
        replicates=int(evaluation["bootstrap_replicates"]),
        confidence=float(evaluation["bootstrap_confidence_level"]),
        seed=int(evaluation["bootstrap_seed"]) + 100,
    )
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
                "output_tokens": sample.output_tokens,
                "remaining_tokens": sample.remaining_tokens,
                **{
                    method_id: float(predictions[method_id][index])
                    for method_id in PLP_V3_METHOD_IDS
                },
            }
        )
    output = run_root / protocol["outputs"]["final_test"]
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "split": "final_test",
        "test_opened": True,
        "test_access": marker,
        "censoring": censoring,
        "methods": method_reports,
        "terminal_zero_minus_v2_paired_family_mae": paired,
        "primary_claim_result": {
            "passed": paired["upper"] < 0,
            "rule": "upper bound of the paired 95% family-bootstrap MAE difference is below zero",
        },
    }
    (output / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "PLP-only final Test evaluated once; "
        f"passed={report['primary_claim_result']['passed']}; output={output}"
    )


if __name__ == "__main__":
    main()

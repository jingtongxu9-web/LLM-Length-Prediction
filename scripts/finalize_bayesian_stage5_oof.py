"""Combine five completed Stage-5 folds and apply the frozen Bayesian selection rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.bayesian_stage5 import load_stage5_catalog, validate_stage5_grid
from llm_length_prediction.evaluation.stage5 import (
    paired_family_nll_difference,
    posterior_breakdowns,
    posterior_metrics,
)
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.models.bayesian_scorer import (
    HIDDEN_DELTA_METHOD_ID,
    SCALAR_METHOD_ID,
)

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage5_oof_v1.json")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _point_metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    sequence_groups: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequence_groups[(row["prompt_id"], float(row["temperature"]), int(row["seed"]))].append(
            row
        )
    sequence_mae = []
    sequence_mse = []
    family_mae: dict[str, list[float]] = defaultdict(list)
    for values in sequence_groups.values():
        errors = np.asarray(
            [float(row[method]) - int(row["true_remaining"]) for row in values]
        )
        mae = float(np.abs(errors).mean())
        sequence_mae.append(mae)
        sequence_mse.append(float(np.square(errors).mean()))
        family_mae[str(values[0]["prompt_family_id"])].append(mae)
    return {
        "observation_count": len(rows),
        "sequence_count": len(sequence_groups),
        "family_count": len(family_mae),
        "sequence_balanced_mae_tokens": float(np.mean(sequence_mae)),
        "sequence_balanced_rmse_tokens": math.sqrt(float(np.mean(sequence_mse))),
        "family_macro_sequence_balanced_mae_tokens": float(
            np.mean([np.mean(values) for values in family_mae.values()])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    catalog = load_stage5_catalog(args.config, dataset_root=args.dataset_root)
    validate_stage5_grid(catalog)
    config = catalog.config
    run_root = args.run_root or Path(config["outputs"]["run_root"])
    fold_root = run_root / config["outputs"]["folds"]
    fold_reports = []
    probabilistic_rows = {
        "alps_countdown": [],
        SCALAR_METHOD_ID: [],
        HIDDEN_DELTA_METHOD_ID: [],
    }
    baseline_rows = []
    fold_count = int(config["data_policy"]["outer_oof_folds"])
    for fold in range(fold_count):
        directory = fold_root / f"fold_{fold}"
        report_path = directory / "fold_report.json"
        if not report_path.is_file():
            raise SystemExit(f"Stage-5 fold {fold} is incomplete: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "pass"
            or report.get("dataset_digest") != catalog.dataset_digest
            or report.get("final_holdout_accessed") is not False
        ):
            raise SystemExit(f"Stage-5 fold {fold} report failed validation")
        fold_reports.append(report)
        for method in probabilistic_rows:
            path = directory / report["files"][method]["predictions"]
            if file_sha256(path) != report["files"][method]["predictions_sha256"]:
                raise SystemExit(f"Stage-5 fold {fold} prediction digest changed: {method}")
            probabilistic_rows[method].extend(_read_jsonl(path))
        baseline_path = directory / "baseline_predictions.csv"
        if baseline_path.is_file():
            with baseline_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    converted: dict[str, Any] = dict(row)
                    for name in (
                        "temperature",
                        "prompt_token_ridge_countdown",
                        "alps_countdown",
                        "dynamic_signal_mlp_v1",
                        "plp_terminal_zero_v3",
                        "alps_plp_concat_v1",
                    ):
                        converted[name] = float(row[name])
                    for name in ("seed", "step", "true_remaining"):
                        converted[name] = int(row[name])
                    converted["outer_fold"] = fold
                    baseline_rows.append(converted)
    expected_sequences = len(catalog.references)
    for method, rows in probabilistic_rows.items():
        identities = {
            (row["prompt_id"], float(row["temperature"]), int(row["seed"])) for row in rows
        }
        if len(identities) != expected_sequences:
            raise SystemExit(f"{method} covers {len(identities)} of {expected_sequences} traces")
    primary_temperature = float(config["data_policy"]["training_temperature"])
    primary_rows = {
        method: [row for row in rows if float(row["temperature"]) == primary_temperature]
        for method, rows in probabilistic_rows.items()
    }
    evaluation = config["evaluation"]
    paired = paired_family_nll_difference(
        primary_rows[HIDDEN_DELTA_METHOD_ID],
        primary_rows[SCALAR_METHOD_ID],
        replicates=int(evaluation["bootstrap_replicates"]),
        confidence=float(evaluation["bootstrap_confidence_level"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    selected = (
        HIDDEN_DELTA_METHOD_ID
        if paired["upper"] < 0
        else config["selection"]["otherwise_select"]
    )
    methods = {}
    for method, rows in probabilistic_rows.items():
        methods[method] = {
            "primary_temperature": posterior_metrics(primary_rows[method]),
            "all_temperatures": posterior_metrics(rows),
            "breakdowns": posterior_breakdowns(rows),
        }
    baseline_methods = (
        "prompt_token_ridge_countdown",
        "alps_countdown",
        "dynamic_signal_mlp_v1",
        "plp_terminal_zero_v3",
        "alps_plp_concat_v1",
    )
    baselines = {}
    if baseline_rows:
        for method in baseline_methods:
            baselines[method] = {
                "primary_temperature": _point_metrics(
                    [
                        row
                        for row in baseline_rows
                        if float(row["temperature"]) == primary_temperature
                    ],
                    method,
                ),
                "all_temperatures": _point_metrics(baseline_rows, method),
            }
    selection = {
        "selected_method": selected,
        "hidden_delta_minus_scalar_family_paired_nll": paired,
        "rule": (
            "select hidden-delta only when the paired family-bootstrap NLL "
            "confidence interval is entirely below zero; otherwise select scalar"
        ),
        "final_holdout_selects_nothing": True,
    }
    report = {
        "schema_version": 1,
        "stage5_id": config["stage5_id"],
        "status": "pass",
        "split": "train_family_grouped_oof",
        "config_sha256": file_sha256(args.config),
        "dataset_digest": catalog.dataset_digest,
        "fold_reports": fold_reports,
        "methods": methods,
        "baselines": baselines,
        "selection": selection,
        "robustness_refit_performed": False,
        "final_holdout_accessed": False,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / config["outputs"]["report"]
    selection_path = run_root / config["outputs"]["selection"]
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Stage-5 OOF complete; selected={selected}; report={report_path}", flush=True
    )


if __name__ == "__main__":
    main()

"""Validate the frozen ALPS configuration with family-grouped cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_length_prediction.evaluation.grouped_cv import (
    cross_validate,
    flatten_results,
)
from llm_length_prediction.evaluation.trace_diagnostics import load_train_diagnostic_rows
from llm_length_prediction.experiment import load_experiment
from llm_length_prediction.models.baselines import ALL_DIAGNOSTIC_MODELS

FROZEN_FOLDS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/alps_v1_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/alps_v1/diagnostics/grouped_cv"),
    )
    parser.add_argument("--folds", type=int, default=FROZEN_FOLDS)
    args = parser.parse_args()
    if args.folds != FROZEN_FOLDS:
        raise SystemExit(
            f"ALPS v1 freezes family-grouped CV at {FROZEN_FOLDS} folds; "
            f"received --folds {args.folds}"
        )

    experiment = load_experiment(args.experiment)
    ridge = experiment["ridge"]
    if ridge.get("standardize") is not True:
        raise SystemExit("ALPS v1 requires train-only feature standardization")
    frozen_alpha = float(ridge["alpha"])
    feature_layer = int(experiment["model"]["feature_layer"])

    try:
        rows = load_train_diagnostic_rows(args.experiment)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    results = []
    for model in ALL_DIAGNOSTIC_MODELS:
        alpha = 0.0 if model == "global_mean" else frozen_alpha
        results.append(
            cross_validate(rows, model_name=model, alpha=alpha, n_splits=args.folds)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = flatten_results(results)
    validation = {
        "mode": "frozen_config_generalization_check",
        "experiment_id": experiment["experiment_id"],
        "feature_layer": feature_layer,
        "ridge_alpha": frozen_alpha,
        "folds": args.folds,
        "group_key": "prompt_family_id",
        "train_rollout_count": len(rows),
        "train_family_count": len({row.prompt_family_id for row in rows}),
        "models": list(ALL_DIAGNOSTIC_MODELS),
        "selects_hyperparameters": False,
        "fits_final_model": False,
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(
        "validated frozen "
        f"Layer {feature_layer} / Ridge alpha={frozen_alpha:g} "
        f"with {args.folds}-fold family-grouped CV on {len(rows)} train rollouts; "
        f"outputs: {args.output_dir}"
    )


if __name__ == "__main__":
    main()

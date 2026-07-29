"""Run family-grouped ALPS diagnostics using training traces only."""

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
from llm_length_prediction.models.baselines import ALL_DIAGNOSTIC_MODELS


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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    args = parser.parse_args()

    try:
        rows = load_train_diagnostic_rows(args.experiment)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    results = []
    for model in ALL_DIAGNOSTIC_MODELS:
        alphas = args.alphas if model != "global_mean" else [0.0]
        for alpha in alphas:
            results.append(
                cross_validate(rows, model_name=model, alpha=alpha, n_splits=args.folds)
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = flatten_results(results)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote grouped-CV diagnostics for {len(rows)} train rollouts: {args.output_dir}")


if __name__ == "__main__":
    main()

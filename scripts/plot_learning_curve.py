"""Generate family-level ALPS learning-curve data from frozen training traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_length_prediction.evaluation.grouped_cv import (
    cross_validate,
    flatten_results,
    select_families,
)
from llm_length_prediction.evaluation.trace_diagnostics import load_train_diagnostic_rows


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
        default=Path("artifacts/runs/alps_v1/diagnostics/learning_curve"),
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    try:
        rows = load_train_diagnostic_rows(args.experiment)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    results = []
    for fraction in args.fractions:
        for repeat in range(args.repeats):
            subset = select_families(rows, fraction, repeat=repeat)
            family_count = len({row.prompt_family_id for row in subset})
            folds = min(args.folds, family_count)
            result = cross_validate(
                subset,
                model_name="alps_hidden",
                alpha=args.alpha,
                n_splits=folds,
            )
            result["training_fraction"] = fraction
            result["repeat"] = repeat
            results.append(result)

    flat = flatten_results(results)
    for target, source in zip(flat, results, strict=True):
        target["training_fraction"] = source["training_fraction"]
        target["repeat"] = source["repeat"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "learning_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(f"wrote {len(flat)} learning-curve evaluations: {args.output_dir}")


if __name__ == "__main__":
    main()

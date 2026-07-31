"""Fit the final prompt-input-token Ridge baseline on all frozen Train traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from llm_length_prediction.comparison import (
    load_complete_split_traces,
    trace_dataset_digest,
)
from llm_length_prediction.evaluation.metrics import log1p_prior_metrics
from llm_length_prediction.experiment import load_experiment, load_frozen_prompts
from llm_length_prediction.models.prior import fit_log1p_ridge_prior

DEFAULT_EXPERIMENT = Path("configs/experiments/alps_v1_manifest.json")


def _validate_cv_baseline(path: Path, experiment: dict[str, object]) -> None:
    if not path.is_file():
        raise ValueError(
            "grouped-CV report is missing; run `python scripts/evaluate_grouped_cv.py` first"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "mode": "frozen_config_generalization_check",
        "experiment_id": experiment["experiment_id"],
        "ridge_alpha": float(experiment["ridge"]["alpha"]),  # type: ignore[index]
        "folds": 5,
        "group_key": "prompt_family_id",
        "selects_hyperparameters": False,
        "fits_final_model": False,
    }
    mismatches = [
        f"{name}: expected {value!r}, got {report.get(name)!r}"
        for name, value in expected.items()
        if report.get(name) != value
    ]
    if "prompt_tokens" not in report.get("models", []):
        mismatches.append("grouped-CV report does not contain the prompt_tokens baseline")
    if mismatches:
        raise ValueError("grouped-CV baseline contract mismatch: " + "; ".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cv-validation", type=Path)
    args = parser.parse_args()

    experiment = load_experiment(args.experiment)
    records = load_frozen_prompts(experiment)
    run_root = Path(experiment["outputs"]["run_root"])
    output_dir = args.output_dir or run_root / "comparisons" / "input_token_ridge"
    cv_validation = (
        args.cv_validation
        or run_root / "diagnostics" / "grouped_cv" / "validation.json"
    )
    _validate_cv_baseline(cv_validation, experiment)
    loaded = load_complete_split_traces(
        experiment,
        records,
        split="train",
        trace_root=args.trace_root,
    )
    features = [[float(trace.prompt_tokens)] for _, _, _, trace in loaded]
    actual = [trace.output_tokens for _, _, _, trace in loaded]
    alpha = float(experiment["ridge"]["alpha"])
    baseline = fit_log1p_ridge_prior(features, actual, alpha=alpha)
    mus = [baseline.predict_mu(feature) for feature in features]
    predicted = [baseline.predict_mean_length(feature) for feature in features]

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = baseline.to_dict()
    payload.update(
        {
            "method": "input_token_ridge",
            "experiment_id": experiment["experiment_id"],
            "model_revision": experiment["model"]["revision"],
            "tokenizer_revision": experiment["model"]["tokenizer_revision"],
            "input_feature": "prompt_tokens",
            "fit_split": "train",
            "training_count": len(loaded),
            "training_dataset_sha256": trace_dataset_digest(loaded),
            "cv_validation_path": str(cv_validation),
            "cv_validation_sha256": hashlib.sha256(cv_validation.read_bytes()).hexdigest(),
        }
    )
    model_path = output_dir / "model.json"
    model_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "train_evaluation.json").write_text(
        json.dumps(
            {
                "split": "train",
                **log1p_prior_metrics(
                    actual,
                    predicted,
                    mus,
                    baseline.residual_variance,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "train_evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "seed",
                "prompt_tokens",
                "actual_output_tokens",
                "predicted_log1p_mu",
                "predicted_mean_output_tokens",
            ],
        )
        writer.writeheader()
        for (_, seed, _, trace), mu, prediction in zip(
            loaded, mus, predicted, strict=True
        ):
            writer.writerow(
                {
                    "prompt_id": trace.prompt_id,
                    "seed": seed,
                    "prompt_tokens": trace.prompt_tokens,
                    "actual_output_tokens": trace.output_tokens,
                    "predicted_log1p_mu": mu,
                    "predicted_mean_output_tokens": prediction,
                }
            )
    print(f"fitted input-token Ridge on {len(loaded)} Train traces: {model_path}")


if __name__ == "__main__":
    main()

"""Evaluate the frozen prompt-input-token Ridge baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_length_prediction.comparison import load_complete_split_traces
from llm_length_prediction.evaluation.metrics import log1p_prior_metrics
from llm_length_prediction.experiment import load_experiment, load_frozen_prompts
from llm_length_prediction.models.prior import StandardizedRidgeLogNormalPrior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/alps_v1_manifest.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/runs/alps_v1/comparisons/input_token_ridge/model.json"),
    )
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_final_test:
        raise SystemExit("refusing final-test evaluation without --confirm-final-test")

    experiment = load_experiment(args.experiment)
    records = load_frozen_prompts(experiment)
    payload = json.loads(args.model.read_text(encoding="utf-8"))
    expected = {
        "method": "input_token_ridge",
        "experiment_id": experiment["experiment_id"],
        "model_revision": experiment["model"]["revision"],
        "tokenizer_revision": experiment["model"]["tokenizer_revision"],
        "input_feature": "prompt_tokens",
        "fit_split": "train",
        "ridge_alpha": float(experiment["ridge"]["alpha"]),
    }
    mismatches = [
        f"{name}: expected {value!r}, got {payload.get(name)!r}"
        for name, value in expected.items()
        if payload.get(name) != value
    ]
    if mismatches:
        raise SystemExit("input-token baseline contract mismatch: " + "; ".join(mismatches))
    baseline = StandardizedRidgeLogNormalPrior.from_dict(payload)
    loaded = load_complete_split_traces(
        experiment,
        records,
        split=args.split,
        trace_root=args.trace_root,
    )
    features = [[float(trace.prompt_tokens)] for _, _, _, trace in loaded]
    actual = [trace.output_tokens for _, _, _, trace in loaded]
    mus = [baseline.predict_mu(feature) for feature in features]
    predicted = [baseline.predict_mean_length(feature) for feature in features]
    output = args.output or args.model.parent / f"{args.split}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "split": args.split,
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
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
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
    print(f"evaluated input-token Ridge on {len(loaded)} {args.split} traces: {output}")


if __name__ == "__main__":
    main()

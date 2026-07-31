"""Evaluate frozen Dynamic-Signal MLP v1 on saved decode traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from llm_length_prediction.comparison import (
    load_base_experiment_for_method,
    load_complete_split_traces,
    load_method_config,
    validate_project_plp_contract,
)
from llm_length_prediction.evaluation.progressive import (
    progress_breakdown,
    progressive_metrics,
)
from llm_length_prediction.models.dynamic import (
    ProgressiveSample,
    StandardizedMLPRemainingLength,
    build_progressive_samples,
)


def _write_predictions(
    path: Path,
    samples: list[ProgressiveSample],
    predicted_mu: list[float],
    predicted_remaining: list[float],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "prompt_family_id",
                "seed",
                "step",
                "decode_progress",
                "actual_output_tokens",
                "actual_remaining_tokens",
                "predicted_log1p_remaining_mu",
                "predicted_mean_remaining_tokens",
                "predicted_mean_total_tokens",
            ],
        )
        writer.writeheader()
        for sample, mu, remaining in zip(
            samples, predicted_mu, predicted_remaining, strict=True
        ):
            writer.writerow(
                {
                    "prompt_id": sample.prompt_id,
                    "prompt_family_id": sample.prompt_family_id,
                    "seed": sample.seed,
                    "step": sample.step,
                    "decode_progress": sample.step / sample.output_tokens,
                    "actual_output_tokens": sample.output_tokens,
                    "actual_remaining_tokens": sample.remaining_tokens,
                    "predicted_log1p_remaining_mu": mu,
                    "predicted_mean_remaining_tokens": remaining,
                    "predicted_mean_total_tokens": sample.step + remaining,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/plp_v1_manifest.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/runs/alps_v1/comparisons/plp_only/model.json"),
    )
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_final_test:
        raise SystemExit("refusing final-test evaluation without --confirm-final-test")

    config = load_method_config(args.config)
    _, experiment, records = load_base_experiment_for_method(config)
    validate_project_plp_contract(config, experiment)
    payload = json.loads(args.model.read_text(encoding="utf-8"))
    expected = {
        "method": "project_plp_only",
        "method_id": config["method_id"],
        "experiment_id": experiment["experiment_id"],
        "model_revision": experiment["model"]["revision"],
        "tokenizer_revision": experiment["model"]["tokenizer_revision"],
        "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
        "fit_split": "train",
        "method_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "feature_names": config["features"],
        "hidden_sizes": config["model"]["hidden_sizes"],
        "dropout": float(config["model"]["dropout"]),
    }
    mismatches = [
        f"{name}: expected {value!r}, got {payload.get(name)!r}"
        for name, value in expected.items()
        if payload.get(name) != value
    ]
    if mismatches:
        raise SystemExit("Dynamic-Signal MLP v1 contract mismatch: " + "; ".join(mismatches))
    model = StandardizedMLPRemainingLength.from_dict(payload)
    loaded = load_complete_split_traces(
        experiment,
        records,
        split=args.split,
        trace_root=args.trace_root,
    )
    samples = [
        sample
        for record, _, _, trace in loaded
        for sample in build_progressive_samples(
            trace,
            prompt_family_id=record["prompt_family_id"],
        )
    ]
    if not samples:
        raise SystemExit(f"{args.split} traces contain no non-terminal dynamic samples")
    predicted_mu = [
        float(value)
        for value in model.predict_mu_many([sample.features for sample in samples])
    ]
    predicted_remaining = [
        float(value)
        for value in model.predict_remaining_many(
            [sample.features for sample in samples]
        )
    ]
    output = args.output or args.model.parent / f"{args.split}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "split": args.split,
                "prediction_target": "remaining_tokens",
                "overall": progressive_metrics(
                    samples,
                    predicted_remaining,
                    predicted_mu,
                    model.residual_variance,
                ),
                "by_decode_progress": progress_breakdown(
                    samples,
                    predicted_remaining,
                    predicted_mu,
                    model.residual_variance,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_predictions(
        output.with_suffix(".csv"),
        samples,
        predicted_mu,
        predicted_remaining,
    )
    print(
        f"evaluated Dynamic-Signal MLP v1 on {len(loaded)} {args.split} traces / "
        f"{len(samples)} non-terminal points: {output}"
    )


if __name__ == "__main__":
    main()
